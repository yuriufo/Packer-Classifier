#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import pandas as pd
import collections
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch
import pre_data.settings as sts
from collections import Counter
from tqdm import tqdm

from Datasets.img_datasets import Vocabulary, IMG_SequenceVocabulary
from Datasets.ins_datasets import INS_SequenceVocabulary

__all__ = ["get_datasets"]

# Classes
classes = sts.PACKERS_LANDSPACE


# 向量器：输入和输出的词汇表类实例
class Vectorizer(object):
    def __init__(self, image_vocab, ins_word_vocab, ins_char_vocab,
                 packer_vocab):
        self.image_vocab = image_vocab
        self.ins_word_vocab = ins_word_vocab
        self.ins_char_vocab = ins_char_vocab
        self.packer_vocab = packer_vocab

    # 向量化
    def vectorize(self, image, ins_s):
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

        # 向量化每一个文件的汇编指令集序列
        indices = [
            self.ins_word_vocab.lookup_token(token)
            for token in ins_s.split(" ")
        ]

        indices = [self.ins_word_vocab.begin_seq_index
                   ] + indices + [self.ins_word_vocab.end_seq_index]

        # 词级向量
        ins_length = len(indices)
        word_vector = np.zeros(ins_length, dtype=np.int64)
        word_vector[:ins_length] = indices

        # 字符级向量
        word_length = max([len(word) for word in ins_s.split(" ")])
        char_vector = np.zeros((len(word_vector), word_length), dtype=np.int64)
        char_vector[0, :] = self.ins_word_vocab.mask_index  # <BEGIN>
        char_vector[-1, :] = self.ins_word_vocab.mask_index  # <END>
        for i, word in enumerate(ins_s.split(" ")):
            char_vector[i + 1, :len(word)] = [
                self.ins_char_vocab.lookup_token(char) for char in word
            ]

        return image, word_vector, char_vector, ins_length

    # 词级反向量化
    def unvectorize_word_vector(self, word_vector):
        tokens = [
            self.ins_word_vocab.lookup_index(index) for index in word_vector
        ]
        ins_ = " ".join(token for token in tokens)
        return ins_

    # 字符级反向量化
    def unvectorize_char_vector(self, char_vector):
        ins_ = ""
        for word_vector in char_vector:
            for index in word_vector:
                if index == self.ins_char_vocab.mask_index:
                    break
                ins_ += self.ins_char_vocab.lookup_index(index)
            ins_ += " "
        return ins_

    @classmethod
    def from_dataframe(cls, df, cutoff):

        # 创建壳类别词汇表
        packer_vocab = Vocabulary()
        for packer in sorted(set(df.packer)):
            packer_vocab.add_token(packer)

        # 创建图像词汇表
        image_vocab = IMG_SequenceVocabulary.from_dataframe(df)

        # 获取指令数目
        word_counts = Counter()
        for ins_ in df.ins:
            for token in ins_.split(" "):
                word_counts[token] += 1

        # 创建反汇编指令的词汇表实例(word)
        ins_word_vocab = INS_SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                ins_word_vocab.add_token(word)

        # 创建反汇编指令的词汇表实例(char)
        ins_char_vocab = INS_SequenceVocabulary()
        for ins_ in df.ins:
            for token in ins_:
                ins_char_vocab.add_token(token)

        return cls(image_vocab, ins_word_vocab, ins_char_vocab, packer_vocab)

    @classmethod
    def from_serializable(cls, contents):
        image_vocab = IMG_SequenceVocabulary.from_serializable(
            contents['image_vocab'])
        ins_word_vocab = INS_SequenceVocabulary.from_serializable(
            contents['ins_word_vocab'])
        ins_char_vocab = INS_SequenceVocabulary.from_serializable(
            contents['ins_char_vocab'])
        packer_vocab = Vocabulary.from_serializable(contents['packer_vocab'])
        return cls(image_vocab, ins_word_vocab, ins_char_vocab, packer_vocab)

    def to_serializable(self):
        return {
            'image_vocab': self.image_vocab.to_serializable(),
            'ins_word_vocab': self.ins_word_vocab.to_serializable(),
            'ins_char_vocab': self.ins_char_vocab.to_serializable(),
            'packer_vocab': self.packer_vocab.to_serializable()
        }


# 数据集：提供向量化数据
class Dataset(Dataset):
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
    def load_dataset_and_make_vectorizer(cls, df, cutoff):
        train_df = df[df.split == 'train']
        return cls(df, Vectorizer.from_dataframe(train_df, cutoff))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with vectorizer_filepath.open() as fp:
            return Vectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with vectorizer_filepath.open("w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})>".format(self.target_split,
                                                       self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        image_vector, ins_word_vector, ins_char_vector, ins_length = self.vectorizer.vectorize(
            row.image, row.ins)
        packer_index = self.vectorizer.packer_vocab.lookup_token(row.packer)
        return {
            'image_vector': image_vector,
            'ins_word_vector': ins_word_vector,
            'ins_char_vector': ins_char_vector,
            'ins_length': ins_length,
            'packer': packer_index
        }

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self,
                         batch_size,
                         collate_fn,
                         shuffle=True,
                         drop_last=False,
                         device="cpu"):
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last)
        for data_dict in tqdm(dataloader):
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


def get_datasets(csv_path=sts.SAVE_CSV_PATH / "train_data_20190429.pkl",
                 randam_seed=None,
                 state_size=[0.7, 0.15, 0.15],
                 vectorize=None):

    if np.sum(state_size) != 1.0 or any([i < 0 for i in state_size]):
        raise Exception("np.sum({0}) != 1 or not integer".format(state_size))
    if randam_seed is not None:
        np.random.seed(randam_seed)

    df = pd.read_pickle(csv_path)

    by_packer = collections.defaultdict(list)
    for _, row in df.iterrows():
        by_packer[row.packer].append(row.to_dict())

    # print("---->>>   packer:")
    # for packer in by_packer:
    #     print("{0}: {1}".format(packer, len(by_packer[packer])))

    final_list = []
    for _, item_list in sorted(by_packer.items()):
        np.random.shuffle(item_list)
        n = len(item_list)
        n_train = int(state_size[0] * n)
        n_val = int(state_size[1] * n)

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
    if vectorize is None:
        dataset = Dataset.load_dataset_and_make_vectorizer(split_df, 5)
    else:
        dataset = Dataset.load_dataset_and_load_vectorizer(split_df, vectorize)
    return dataset


if __name__ == '__main__':
    datasets = get_datasets(randam_seed=22)
    vector = datasets.vectorizer

    print(vector.image_vocab)
    print(vector.ins_word_vocab)
    print(vector.ins_char_vocab)
    print(vector.packer_vocab)

    print()

    print(datasets)
    print(datasets.class_weights)
    input_ = datasets[10]  # __getitem__
    print(input_['image_vector'])
    print(input_['ins_word_vector'])
    print(input_['ins_char_vector'])
    print(input_['ins_length'])
    print(input_['packer'])
