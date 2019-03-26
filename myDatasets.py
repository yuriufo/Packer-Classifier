#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gadgets import img_to_array
import pandas as pd
import collections
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch
import pre_data.settings as sts

# Classes
classes = {
    0: 'ACProtect',
    1: 'Armadillo',
    2: 'ASProtect',
    3: 'ExeCryptor',
    4: 'ExeShield',
    5: 'Obsidium',
    6: 'PeCompact',
    7: 'PELock',
    8: 'Themida',
    9: 'UPX'
}


# 词汇表：原始输入和数字形式的转换字典，用于壳类型
class Vocabulary(object):
    def __init__(self, token_to_idx=None):

        # 令牌到索引
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx

        # 索引到令牌
        self.idx_to_token = {
            idx: token
            for token, idx in self.token_to_idx.items()
        }

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token[token] for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self.token_to_idx)


# 序列化词汇表：实际图像的词汇表，存储标准差与平均值
class SequenceVocabulary():
    def __init__(self, train_means, train_stds):

        self.train_means = train_means
        self.train_stds = train_stds

    def to_serializable(self):
        contents = {
            'train_means': self.train_means,
            'train_stds': self.train_stds
        }
        return contents

    @classmethod
    def from_dataframe(cls, df):
        train_data = df[df.split == "train"]
        means = {0: [], 1: [], 2: []}
        stds = {0: [], 1: [], 2: []}
        for image in train_data.image:
            for dim in range(3):
                means[dim].append(np.mean(image[:, :, dim]))
                stds[dim].append(np.std(image[:, :, dim]))
        train_means = np.array(
            (np.mean(means[0]), np.mean(means[1]), np.mean(means[2])),
            dtype="float64").tolist()
        train_stds = np.array(
            (np.mean(stds[0]), np.mean(stds[1]), np.mean(stds[2])),
            dtype="float64").tolist()

        return cls(train_means, train_stds)

    def __str__(self):
        return "<SequenceVocabulary(train_means: {0}, train_stds: {1}>".format(
            self.train_means, self.train_stds)


# 向量器：输入和输出的词汇表类实例，使用词汇表标准化图像
class ImageVectorizer(object):
    def __init__(self, image_vocab, packer_vocab):
        self.image_vocab = image_vocab
        self.packer_vocab = packer_vocab

    def vectorize(self, image):

        # 防止改变实际的df
        image = np.copy(image)

        # 正则化
        for dim in range(3):
            mean = self.image_vocab.train_means[dim]
            std = self.image_vocab.train_stds[dim]
            image[:, :, dim] = ((image[:, :, dim] - mean) / std)

        # 把输入shape (a, a, b) 变为 (b, a, a)
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)

        return image

    @classmethod
    def from_dataframe(cls, df):
        # 创建壳类别词汇表
        packer_vocab = Vocabulary()
        for packer in sorted(set(df.packer)):
            packer_vocab.add_token(packer)
        # 创建图像词汇表
        image_vocab = SequenceVocabulary.from_dataframe(df)
        return cls(image_vocab, packer_vocab)

    @classmethod
    def from_serializable(cls, contents):
        image_vocab = SequenceVocabulary.from_serializable(
            contents['image_vocab'])
        packer_vocab = Vocabulary.from_serializable(contents['packer_vocab'])
        return cls(image_vocab=image_vocab, packer_vocab=packer_vocab)

    def to_serializable(self):
        return {
            'image_vocab': self.image_vocab.to_serializable(),
            'packer_vocab': self.packer_vocab.to_serializable()
        }


# 数据集：提供向量化数据
class ImageDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

        # 数据分割
        self.train_df = self.df[self.df.split == 'train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split == 'val']
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split == 'test']
        self.test_size = len(self.test_df)
        self.lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }
        self.set_split('train')

        # 分类权重（防止类别不平衡）
        class_counts = df.packer.value_counts().to_dict()

        def sort_key(item):
            return self.vectorizer.packer_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(
            frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df):
        train_df = df[df.split == 'train']
        return cls(df, ImageVectorizer.from_dataframe(train_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with vectorizer_filepath.open() as fp:
            return ImageVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with vectorizer_filepath.open("w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(self.target_split,
                                                      self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        image_vector = self.vectorizer.vectorize(row.image)
        packer_index = self.vectorizer.packer_vocab.lookup_token(row.packer)
        return {'image': image_vector, 'packer': packer_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self,
                         batch_size,
                         shuffle=True,
                         drop_last=True,
                         device="cpu"):
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


def get_datasets(img_path=sts.SAVE_IMAGES_PATH):
    data = []
    for i, _class in enumerate(classes.values()):
        for file in (img_path/_class).glob("*.png"):
            full_filepath = img_path/_class/file
            data.append({
                "image": img_to_array(full_filepath),
                "packer": _class
            })

    df = pd.DataFrame(data)

    by_packer = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_packer[row.packer].append(row.to_dict())

    final_list = []
    for _, item_list in sorted(by_packer.items()):
        if True:
            np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        # 给数据点一个切分属性
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train + n_val]:
            item['split'] = 'val'
        for item in item_list[n_train + n_val:]:
            item['split'] = 'test'

        final_list.extend(item_list)

    split_df = pd.DataFrame(final_list)
    # print(split_df.head())

    # 数据库实例
    dataset = ImageDataset.load_dataset_and_make_vectorizer(split_df)
    return dataset


if __name__ == '__main__':
    print(get_datasets().class_weights)
