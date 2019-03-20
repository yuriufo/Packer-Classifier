#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import adabound

import json
import time
import uuid
import matplotlib.pyplot as plt

from my_models.ODEnet import ODEfunc, ODEBlock, Flatten, norm
from gadgets import compute_accuracy, update_train_state

from myDatasets import get_datasets

# 参数
config = {
    "seed": 4396,
    "cuda": False,
    "shuffle": True,
    "experiment_id": None,
    "data_file": "names.csv",
    "split_data_file": "split_names.csv",
    "vectorizer_file": "vectorizer.json",
    "model_state_file": "model.pth",
    "save_dir": "experiments",
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "num_epochs": 60,
    "early_stopping_criteria": 5,
    "learning_rate": 1e-3,
    "batch_size": 24
}


# 生成唯一ID
def generate_unique_id():
    timestamp = int(time.time())
    unique_id = "{}_{}".format(timestamp, uuid.uuid1())
    return unique_id


# 设置随机种子
def set_seeds(seed, cuda):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


# 创建目录
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# (W-F+2P)/S
# (W-F)/S + 1
# ODEnet模型
class my_ODEnet(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, tol=1e-3):
        super(my_ODEnet, self).__init__()
        # 输入shape：(3,32,32)
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(input_dim, state_dim, 3, 1), norm(state_dim),
            nn.ReLU(inplace=True), nn.Conv2d(state_dim, state_dim, 4, 2, 1),
            norm(state_dim), nn.ReLU(inplace=True),
            nn.Conv2d(state_dim, state_dim, 4, 2, 1))

        self.feature_layers = ODEBlock(ODEfunc(state_dim), rtol=tol, atol=tol)

        self.fc_layers = nn.Sequential(
            norm(state_dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            Flatten(), nn.Linear(state_dim, output_dim))

    def forward(self, x_in, apply_softmax=False):
        out = self.downsampling_layers(x_in)
        out = self.feature_layers(out)
        out = self.fc_layers(out)

        if apply_softmax:
            out = F.softmax(out, dim=1)

        return out

    @property
    def nfe(self):
        return self.feature_layers.nfe

    @nfe.setter
    def nfe(self, value):
        self.feature_layers.nfe = value


def init_ODEnet():
    # 新建保存文件路径
    create_dirs(config["save_dir"])
    print("---->>>   Created {}".format(config["save_dir"]))
    # 设置种子
    set_seeds(seed=config["seed"], cuda=config["cuda"])
    print("---->>>   Set seeds.")
    # 检查是否有可用GPU
    config["cuda"] = True if torch.cuda.is_available() else False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")
    print("---->>>   Using CUDA: {}".format(config["cuda"]))
    # 设置当前实验ID
    config["experiment_id"] = generate_unique_id()
    print("---->>>   Generated unique id: {0}".format(config["experiment_id"]))


class Trainer(object):
    def __init__(self, dataset, model, model_file, device, shuffle, num_epochs,
                 batch_size, learning_rate, early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = adabound.AdaBound(self.model.parameters(), lr=learning_rate)  # 新的优化方法
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'done_training': False,
            'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': model_file,
            "f_nfe": [],
            "b_nfe": []
        }

    def run_train_loop(self):
        print("---->>>   Training:")
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index

            # 遍历训练集

            # 初始化批生成器, 设置为训练模式，损失和准确率归零
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            f_nfe, b_nfe = 0.0, 0.0
            self.model.train()
            self.model.nfe = 0

            for batch_index, batch_dict in enumerate(batch_generator):
                # 梯度归零
                self.optimizer.zero_grad()

                # 计算输出
                y_pred = self.model(batch_dict['image'])

                # 计算损失
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                f_nfe = (self.model.nfe - f_nfe) / (batch_index + 1)
                self.model.nfe = 0

                # 反向传播
                loss.backward()

                # 更新梯度
                self.optimizer.step()

                b_nfe = (self.model.nfe - b_nfe) / (batch_index + 1)
                self.model.nfe = 0

                # 计算准确率
                acc_t = compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)
            self.train_state['f_nfe'].append(f_nfe)
            self.train_state['b_nfe'].append(b_nfe)

            # 遍历验证集

            # 初始化批生成器, 设置为验证模式，损失和准确率归零
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # 计算输出
                y_pred = self.model(batch_dict['image'])

                # 计算损失
                loss = self.loss_func(y_pred, batch_dict['category'])
                loss_t = loss.to("cpu").item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 计算准确率
                acc_t = compute_accuracy(y_pred, batch_dict['category'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)

            self.train_state = update_train_state(
                model=self.model, train_state=self.train_state)
            self.scheduler.step(self.train_state['val_loss'][-1])
            if self.train_state['stop_early']:
                break

    def run_test_loop(self):
        # 初始化批生成器, 设置为测试模式，损失和准确率归零
        self.dataset.set_split('test')
        batch_generator = self.dataset.generate_batches(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # 计算输出
            y_pred = self.model(batch_dict['image'])

            # 计算损失
            loss = self.loss_func(y_pred, batch_dict['category'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # 计算准确率
            acc_t = compute_accuracy(y_pred, batch_dict['category'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

        # 详细信息
        print("---->>>   Test performance:")
        print("Test loss: {0:.2f}".format(self.train_state['test_loss']))
        print("Test Accuracy: {0:.1f}%".format(self.train_state['test_acc']))


def plot_performance(train_state, save_dir, show_plot=True):
    """ Plot loss and accuracy.
    """
    # Figure size
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_state["train_loss"], label="train")
    plt.plot(train_state["val_loss"], label="val")
    plt.legend(loc='upper right')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(train_state["train_acc"], label="train")
    plt.plot(train_state["val_acc"], label="val")
    plt.legend(loc='lower right')

    # Save figure
    plt.savefig(os.path.join(save_dir, "performance.png"))

    # Show plots
    if show_plot:
        print("---->>>   Metric plots:")
        plt.show()


def save_train_state(train_state, save_dir):
    train_state["done_training"] = True
    with open(os.path.join(save_dir, "train_state.json"), "w") as fp:
        json.dump(train_state, fp)
    print("---->>>  Training complete!")


def train():
    split_df, dataset = get_datasets()
    dataset.save_vectorizer(config["vectorizer_file"])
    vectorizer = dataset.vectorizer
    model = my_ODEnet(
        input_dim=3, output_dim=len(vectorizer.category_vocab), state_dim=64)
    print(model.named_modules)

    trainer = Trainer(
        dataset=dataset,
        model=model,
        model_file=config["model_state_file"],
        device=config["device"],
        shuffle=config["shuffle"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        early_stopping_criteria=config["early_stopping_criteria"])
    trainer.run_train_loop()
    plot_performance(
        train_state=trainer.train_state,
        save_dir=config["save_dir"],
        show_plot=True)
    trainer.run_test_loop()
    save_train_state(
        train_state=trainer.train_state, save_dir=config["save_dir"])


def main():
    init_ODEnet()
    train()


if __name__ == '__main__':
    main()
