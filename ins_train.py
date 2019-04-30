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
import copy
import numpy as np

from gadgets.ggs import compute_accuracy, update_train_state, save_train_state, plot_performance, Confusion_matrix

from Datasets.ins_datasets import get_ins_datasets

# 参数
config = {
    "seed": 7,
    "cuda": False,
    "shuffle": True,
    "train_state_file": "train_state.json",
    "vectorizer_file": "vectorizer.json",
    "model_state_file": "model.pth",
    "performance_img": "performance.png",
    "save_dir": Path.cwd() / "experiments",
    "cutoff": 25,
    "num_filters": 100,
    "embedding_dim": 100,
    "hidden_dim": 100,
    "dropout_p": 0.3,
    "state_size": [0.7, 0.15, 0.15],  # [训练, 验证, 测试]
    "batch_size": 64,
    "num_epochs": 15,
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
class InsModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 num_input_channels,
                 num_channels,
                 hidden_dim,
                 num_classes,
                 dropout_p,
                 pretrained_embeddings=None,
                 freeze_embeddings=False,
                 padding_idx=0):
        super(InsModel, self).__init__()

        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(
                pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings)

        # 卷积层权重
        self.conv = nn.ModuleList([
            nn.Conv1d(num_input_channels, num_channels, kernel_size=f)
            for f in [2, 3, 4]
        ])

        # 全连接层权重
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, x_in, channel_first=False, apply_softmax=False):

        # 嵌入
        x_in = self.embeddings(x_in)

        # 重置输入形状
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        # 卷积层输出
        z1 = self.conv[0](x_in)
        z1 = F.max_pool1d(z1, z1.size(2)).squeeze(2)
        z2 = self.conv[1](x_in)
        z2 = F.max_pool1d(z2, z2.size(2)).squeeze(2)
        z3 = self.conv[2](x_in)
        z3 = F.max_pool1d(z3, z3.size(2)).squeeze(2)

        # 拼接卷积层输出
        z = torch.cat([z1, z2, z3], 1)

        # 全连接层
        z = self.dropout(z)
        z = self.fc1(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred


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

    def pad_seq(self, seq, length):
        vector = np.zeros(length, dtype=np.int64)
        vector[:len(seq)] = seq
        vector[len(seq):] = self.dataset.vectorizer.ins_vocab.mask_index
        return vector

    def collate_fn(self, batch):

        # 深度拷贝
        batch_copy = copy.deepcopy(batch)
        processed_batch = {"ins": [], "packer": []}

        # 得到最长序列长度
        max_seq_len = max([len(sample["ins"]) for sample in batch_copy])

        # 填充
        for i, sample in enumerate(batch_copy):
            seq = sample["ins"]
            packer = sample["packer"]
            padded_seq = self.pad_seq(seq, max_seq_len)
            processed_batch["ins"].append(padded_seq)
            processed_batch["packer"].append(packer)

        # 转换为合适的tensor
        processed_batch["ins"] = torch.LongTensor(processed_batch["ins"])
        processed_batch["packer"] = torch.LongTensor(processed_batch["packer"])

        return processed_batch

    def run_train_loop(self):
        print("---->>>   Training:")
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index

            # 遍历训练集

            # 初始化批生成器, 设置为训练模式，损失和准确率归零
            self.dataset.set_split('train')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.shuffle,
                device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # 梯度归零
                self.optimizer.zero_grad()

                # 计算输出
                y_pred = self.model(batch_dict['ins'])

                # 计算损失
                loss = self.loss_func(y_pred, batch_dict['packer'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # 反向传播
                loss.backward()

                # 更新梯度
                self.optimizer.step()

                # 计算准确率
                acc_t = compute_accuracy(y_pred, batch_dict['packer'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)
            # self.train_state['train_loss'].append(loss_t)
            # self.train_state['train_acc'].append(acc_t)
            self.train_state['f_nfe'].append(0.0)
            self.train_state['b_nfe'].append(0.0)

            # 遍历验证集

            # 初始化批生成器, 设置为验证模式，损失和准确率归零
            self.dataset.set_split('val')
            batch_generator = self.dataset.generate_batches(
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.shuffle,
                device=self.device)
            running_loss = 0.0
            running_acc = 0.0
            self.model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):

                # 计算输出
                y_pred = self.model(batch_dict['ins'])

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
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            device=self.device)
        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()

        all_pred = []
        all_pack = []

        for batch_index, batch_dict in enumerate(batch_generator):
            # 计算输出
            y_pred = self.model(batch_dict['ins'])

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

        # 混淆矩阵
        print("---->>>   Confusion Matrix:")
        Confusion_matrix(all_pred, all_pack)

        # 详细信息
        print("---->>>   Test performance:")
        print("Test loss: {0:.2f}".format(self.train_state['test_loss']))
        print("Test Accuracy: {0:.1f}%".format(self.train_state['test_acc']))


def train():
    # 加载数据集
    dataset = get_ins_datasets(
        csv_path=r"F:\my_packer\csv\train_data_20190429.pkl",
        randam_seed=config["seed"],
        state_size=config["state_size"],
        vectorize=None)
    # 保存向量器
    dataset.save_vectorizer(config["save_dir"] / config["vectorizer_file"])
    # 初始化神经网络
    vectorizer = dataset.vectorizer
    model = InsModel(
        embedding_dim=config["embedding_dim"],
        num_embeddings=len(vectorizer.ins_vocab),
        num_input_channels=config["embedding_dim"],
        num_channels=config["num_filters"],
        hidden_dim=config["hidden_dim"],
        num_classes=len(vectorizer.packer_vocab),
        dropout_p=config["dropout_p"],
        pretrained_embeddings=None,
        padding_idx=vectorizer.ins_vocab.mask_index)
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
        show_plot=True)

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
