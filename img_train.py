#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import adabound

import time
import uuid

from my_models.ODEnet import SE_ODEfunc, ODEBlock, Flatten, norm
# from my_models.my_transformer import ST
from gadgets.ggs import compute_accuracy, update_train_state, save_train_state, plot_performance, Confusion_matrix

from Datasets.img_datasets import get_image_datasets

# 参数
config = {
    "seed": 4396,
    "cuda": False,
    "shuffle": True,
    "train_state_file": "train_state.json",
    "vectorizer_file": "vectorizer.json",
    "model_state_file": "model.pth",
    "performance_img": "performance.png",
    "confusion_matrix_img": "confusion_matrix_img.png",
    "save_dir": Path.cwd() / "experiments" / "img",
    "state_size": [0.7, 0.15, 0.15],  # [训练, 验证, 测试]
    "batch_size": 80,
    "num_epochs": 5,
    "early_stopping_criteria": 3,
    "learning_rate": 3e-5
}


# 生成唯一ID
def generate_unique_id():
    timestamp = int(time.time())
    unique_id = "{0}_{1}".format(timestamp, uuid.uuid1())
    return unique_id


# 设置随机种子
def set_seeds(seed, cuda):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


# 创建目录
def create_dirs(dirpath):
    if not dirpath.exists():
        dirpath.mkdir(parents=True)


# (W-F+2P)/S
# (W-F)/S + 1
# ODEnet模型
class IngModel(nn.Module):
    def __init__(self,
                 input_dim,
                 state_dim,
                 output_dim,
                 reduction=16,
                 tol=1e-3):
        super(IngModel, self).__init__()
        # 输入shape：(3,32,32)
        # self.transformer = ST()

        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(input_dim, state_dim, 3, 1), norm(state_dim),
            nn.ReLU(inplace=True), nn.Conv2d(state_dim, state_dim, 4, 2, 1),
            norm(state_dim), nn.ReLU(inplace=True),
            nn.Conv2d(state_dim, state_dim, 4, 2, 1))

        self.feature_layers = ODEBlock(
            SE_ODEfunc(state_dim, reduction), rtol=tol, atol=tol)

        self.fc_layers = nn.Sequential(
            norm(state_dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
            Flatten(), nn.Linear(state_dim, output_dim))

    def forward(self, x_in, apply_softmax=False):
        # out = self.transformer(x_in)
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


def init():
    print("---->>>   PyTorch version: {}".format(torch.__version__))
    print("---->>>   Created {}".format(config["save_dir"]))
    # 设置种子
    set_seeds(seed=config["seed"], cuda=config["cuda"])
    print("---->>>   Set seeds.")
    # 检查是否有可用GPU
    config["cuda"] = True if torch.cuda.is_available() else False
    config["device"] = torch.device("cuda" if config["cuda"] else "cpu")
    print("---->>>   Using CUDA: {}".format(config["cuda"]))
    if config["cuda"] is True:
        print("---->>>   CUDA version: {}".format(torch.version.cuda))
        print("---->>>   GPU type: {}".format(torch.cuda.get_device_name(0)))
    # 设置当前实验ID
    config["experiment_id"] = generate_unique_id()
    config["save_dir"] = config["save_dir"] / config["experiment_id"]
    create_dirs(config["save_dir"])
    print("---->>>   Generated unique id: {0}".format(config["experiment_id"]))


class Trainer(object):
    def __init__(self, dataset, model, save_dir, model_file, device, shuffle,
                 num_epochs, batch_size, learning_rate,
                 early_stopping_criteria):
        self.dataset = dataset
        self.class_weights = dataset.class_weights.to(device)
        self.model = model.to(device)
        self.device = device
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss(self.class_weights)
        # self.optimizer = adabound.AdaBound(
        #    self.model.parameters(), lr=learning_rate)  # 新的优化方法
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
            'save_dir': save_dir,
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
                loss = self.loss_func(y_pred, batch_dict['packer'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                f_nfe += (self.model.nfe - f_nfe) / (batch_index + 1)
                self.model.nfe = 0

                # 反向传播
                loss.backward()

                # 更新梯度
                self.optimizer.step()

                b_nfe += (self.model.nfe - b_nfe) / (batch_index + 1)
                self.model.nfe = 0

                # 计算准确率
                acc_t = compute_accuracy(y_pred, batch_dict['packer'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)
            # self.train_state['train_loss'].append(loss_t)
            # self.train_state['train_acc'].append(acc_t)
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
                loss = self.loss_func(y_pred, batch_dict['packer'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 计算准确率
                acc_t = compute_accuracy(y_pred, batch_dict['packer'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)
            # self.train_state['val_loss'].append(loss_t)
            # self.train_state['val_acc'].append(acc_t)

            # 学习率
            self.scheduler.step(self.train_state['val_loss'][-1])
            self.train_state['learning_rate'] = float(
                list(self.optimizer.param_groups)[-1]['lr'])
            self.train_state = update_train_state(
                model=self.model, train_state=self.train_state)
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

        all_pred = []
        all_pack = []

        for batch_index, batch_dict in enumerate(batch_generator):
            # 计算输出
            y_pred = self.model(batch_dict['image'])

            # 计算损失
            loss = self.loss_func(y_pred, batch_dict['packer'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # 计算准确率
            acc_t = compute_accuracy(y_pred, batch_dict['packer'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            all_pred.extend(y_pred.max(dim=1)[1])
            all_pack.extend(batch_dict['packer'])

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

        classes_name = [
            self.dataset.vectorizer.packer_vocab.lookup_index(i)
            for i in range(
                len(set([j.cpu().numpy().tolist() for j in all_pack])))
        ]

        # 混淆矩阵
        # print("---->>>   Confusion Matrix:")
        Confusion_matrix(
            y_pred=all_pred,
            y_target=all_pack,
            classes_name=classes_name,
            save_dir=config["save_dir"] / config["confusion_matrix_img"],
            show_plot=False)

        # 详细信息
        print("---->>>   Test performance:")
        print("Test loss: {0:.2f}".format(self.train_state['test_loss']))
        print("Test Accuracy: {0:.1f}%".format(self.train_state['test_acc']))


def train():
    # 加载数据集
    dataset = get_image_datasets(
        csv_path=r"F:\my_packer\csv\train_data_20190429.pkl",
        randam_seed=config["seed"],
        state_size=config["state_size"],
        vectorize=None)
    # 保存向量器
    dataset.save_vectorizer(config["save_dir"] / config["vectorizer_file"])
    # 初始化神经网络
    model = IngModel(
        input_dim=3,
        output_dim=len(dataset.vectorizer.packer_vocab),
        state_dim=64)
    print(model.named_modules)

    # 初始化训练器
    trainer = Trainer(
        dataset=dataset,
        model=model,
        save_dir=config["save_dir"],
        model_file=config["model_state_file"],
        device=config["device"],
        shuffle=config["shuffle"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        early_stopping_criteria=config["early_stopping_criteria"])

    # 训练
    trainer.run_train_loop()

    # 训练状态图
    plot_performance(
        train_state=trainer.train_state,
        save_dir=config["save_dir"] / config["performance_img"],
        show_plot=False)

    # 测试
    trainer.run_test_loop()

    # 保存网络状态
    save_train_state(
        train_state=trainer.train_state,
        save_dir=config["save_dir"] / config["train_state_file"])

    # 清空缓存
    torch.cuda.empty_cache()


def main():
    init()
    train()


if __name__ == '__main__':
    main()
