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
    "num_layers": 1,
    "embedding_dim": 50,
    "kernels": [3, 5],
    "num_filters": 50,
    "rnn_hidden_dim": 64,
    "hidden_dim": 36,
    "dropout_p": 0.25,
    "bidirectional": False,
    "state_size": [0.7, 0.15, 0.15],  # [训练, 验证, 测试]
    "batch_size": 28,
    "num_epochs": 60,
    "early_stopping_criteria": 5,
    "learning_rate": 2e-5
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


def gather_last_relevant_hidden(hiddens, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)


# 编码器
class InsEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_word_embeddings,
                 num_char_embeddings,
                 kernels,
                 num_input_channels,
                 num_output_channels,
                 rnn_hidden_dim,
                 num_layers,
                 bidirectional,
                 word_padding_idx=0,
                 char_padding_idx=0):
        super(InsEncoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 嵌入层
        self.word_embeddings = nn.Embedding(
            num_embeddings=num_word_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=word_padding_idx)
        self.char_embeddings = nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=char_padding_idx)

        # 卷积层权重
        self.conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                kernel_size=kernel) for kernel in kernels
        ])

        # GRU层权重
        self.gru = nn.GRU(
            input_size=embedding_dim * (len(kernels) + 1),
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)

    def initialize_hidden_state(self, batch_size, rnn_hidden_dim, device):
        num_directions = 2 if self.bidirectional else 1
        hidden_t = torch.zeros(self.num_layers * num_directions, batch_size,
                               rnn_hidden_dim).to(device)
        return hidden_t

    def get_char_level_embeddings(self, x):
        # x: (N, seq_len, word_len)
        batch_size, seq_len, word_len = x.size()
        x = x.view(-1, word_len)  # (N * seq_len, word_len)

        # 嵌入层
        x = self.char_embeddings(x)  # (N * seq_len, word_len, embedding_dim)

        # 重排使得num_input_channels在第一维 (N, embedding_dim, word_len)
        x = x.transpose(1, 2)

        # 卷积层
        z = [F.relu(conv(x)) for conv in self.conv]

        # 池化
        z = [F.max_pool1d(zz, zz.size(2)).squeeze(2) for zz in z]
        z = [zz.view(batch_size, seq_len, -1)
             for zz in z]  # (N, seq_len, embedding_dim)

        # 连接卷积输出得到字符级嵌入
        z = torch.cat(z, 2)

        return z

    def forward(self, x_word, x_char, x_lengths, device):
        # x_word: (N, seq_size)
        # x_char: (N, seq_size, word_len)

        # 词级嵌入层
        z_word = self.word_embeddings(x_word)

        # 字符级嵌入层
        z_char = self.get_char_level_embeddings(x=x_char)

        # 连接结果
        z = torch.cat([z_word, z_char], 2)

        # 向RNN输入
        initial_h = self.initialize_hidden_state(
            batch_size=z.size(0),
            rnn_hidden_dim=self.gru.hidden_size,
            device=device)
        out, h_n = self.gru(z, initial_h)

        return out


# 解码器
class InsDecoder(nn.Module):
    def __init__(self, rnn_hidden_dim, hidden_dim, output_dim, dropout_p):
        super(InsDecoder, self).__init__()

        # 注意力机制的全连接模型
        self.fc_attn = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)
        self.v = nn.Parameter(torch.rand(rnn_hidden_dim))

        # 全连接参数
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, apply_softmax=False):

        # 注意力机制
        z = torch.tanh(self.fc_attn(encoder_outputs))
        z = z.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        z = torch.bmm(v, z).squeeze(1)  # [B*T]
        attn_scores = F.softmax(z, dim=1)
        context = torch.matmul(
            encoder_outputs.transpose(-2, -1),
            attn_scores.unsqueeze(dim=2)).squeeze()
        if len(context.size()) == 1:
            context = context.unsqueeze(0)

        # 全连接层
        z = self.dropout(context)
        z = self.fc1(z)
        z = self.dropout(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return attn_scores, y_pred


class InsModel(nn.Module):
    def __init__(self, embedding_dim, num_word_embeddings, num_char_embeddings,
                 kernels, num_input_channels, num_output_channels,
                 rnn_hidden_dim, hidden_dim, output_dim, num_layers,
                 bidirectional, dropout_p, word_padding_idx, char_padding_idx):
        super(InsModel, self).__init__()

        self.encoder = InsEncoder(
            embedding_dim, num_word_embeddings, num_char_embeddings, kernels,
            num_input_channels, num_output_channels, rnn_hidden_dim,
            num_layers, bidirectional, word_padding_idx, char_padding_idx)
        self.decoder = InsDecoder(rnn_hidden_dim, hidden_dim, output_dim,
                                  dropout_p)

    def forward(self, x_word, x_char, x_lengths, device, apply_softmax=False):
        encoder_outputs = self.encoder(x_word, x_char, x_lengths, device)
        y_pred = self.decoder(encoder_outputs, apply_softmax)
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

    def pad_word_seq(self, seq, length):
        vector = np.zeros(length, dtype=np.int64)
        vector[:len(seq)] = seq
        vector[len(seq):] = self.dataset.vectorizer.ins_word_vocab.mask_index
        return vector

    def pad_char_seq(self, seq, seq_length, word_length):
        vector = np.zeros((seq_length, word_length), dtype=np.int64)
        vector.fill(self.dataset.vectorizer.ins_char_vocab.mask_index)
        for i in range(len(seq)):
            char_padding = np.zeros(word_length - len(seq[i]), dtype=np.int64)
            vector[i] = np.concatenate((seq[i], char_padding), axis=None)
        return vector

    def collate_fn(self, batch):

        # 深度拷贝
        batch_copy = copy.deepcopy(batch)
        processed_batch = {
            'ins_word_vector': [],
            'ins_char_vector': [],
            'ins_length': [],
            'packer': []
        }

        # 得到最长序列长度
        max_seq_length = max(
            [len(sample["ins_word_vector"]) for sample in batch_copy])
        max_word_length = max(
            [len(sample["ins_char_vector"][0]) for sample in batch_copy])

        # 填充
        for i, sample in enumerate(batch_copy):
            padded_word_seq = self.pad_word_seq(sample["ins_word_vector"],
                                                max_seq_length)
            padded_cahr_seq = self.pad_char_seq(
                sample["ins_char_vector"], max_seq_length, max_word_length)
            processed_batch["ins_word_vector"].append(padded_word_seq)
            processed_batch["ins_char_vector"].append(padded_cahr_seq)
            processed_batch["ins_length"].append(sample["ins_length"])
            processed_batch["packer"].append(sample["packer"])

        # 转换为合适的tensor
        processed_batch["ins_word_vector"] = torch.LongTensor(
            processed_batch["ins_word_vector"])
        processed_batch["ins_char_vector"] = torch.LongTensor(
            processed_batch["ins_char_vector"])
        processed_batch["ins_length"] = torch.LongTensor(
            processed_batch["ins_length"])
        processed_batch["packer"] = torch.LongTensor(processed_batch["packer"])

        return processed_batch

    def run_train_loop(self):
        print("---->>>   Training:")
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index

            # 遍历训练集

            torch.cuda.empty_cache()

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
                _, y_pred = self.model(
                    x_word=batch_dict['ins_word_vector'],
                    x_char=batch_dict['ins_char_vector'],
                    x_lengths=batch_dict['ins_length'],
                    device=self.device)

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

            torch.cuda.empty_cache()

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
                _, y_pred = self.model(
                    x_word=batch_dict['ins_word_vector'],
                    x_char=batch_dict['ins_char_vector'],
                    x_lengths=batch_dict['ins_length'],
                    device=self.device)

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
        torch.cuda.empty_cache()

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
            _, y_pred = self.model(
                x_word=batch_dict['ins_word_vector'],
                x_char=batch_dict['ins_char_vector'],
                x_lengths=batch_dict['ins_length'],
                device=self.device)

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
        num_word_embeddings=len(vectorizer.ins_word_vocab),
        num_char_embeddings=len(vectorizer.ins_char_vocab),
        kernels=config["kernels"],
        num_input_channels=config["embedding_dim"],
        num_output_channels=config["num_filters"],
        rnn_hidden_dim=config["rnn_hidden_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=len(vectorizer.packer_vocab),
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout_p=config["dropout_p"],
        word_padding_idx=vectorizer.ins_word_vocab.mask_index,
        char_padding_idx=vectorizer.ins_char_vocab.mask_index)
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
