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

__all__ = ["InsVectorizer", "get_ins_datasets"]

# Classes
classes = sts.PACKERS_LANDSPACE


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
            raise KeyError(
                "the index {0} is not in the Vocabulary".format(index))
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={0})>".format(len(self))

    def __len__(self):
        return len(self.token_to_idx)


# 序列化词汇表：反汇编指令的词汇表，存储反汇编指令
class INS_SequenceVocabulary(Vocabulary):
    def __init__(self,
                 token_to_idx=None,
                 unk_token="<UNK>",
                 mask_token="<MASK>",
                 begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(INS_SequenceVocabulary, self).__init__(token_to_idx)

        self.mask_token = mask_token
        self.unk_token = unk_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)

        # 索引到令牌
        self.idx_to_token = {
            idx: token
            for token, idx in self.token_to_idx.items()
        }

    def to_serializable(self):
        contents = super(INS_SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self.unk_token,
            'mask_token': self.mask_token,
            'begin_seq_token': self.begin_seq_token,
            'end_seq_token': self.end_seq_token
        })
        return contents

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_index)

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError(
                "the index ({0}) is not in the INS_SequenceVocabulary".format(
                    index))
        return self.idx_to_token[index]

    def __str__(self):
        return "<INS_SequenceVocabulary(size={0})>".format(
            len(self.token_to_idx))

    def __len__(self):
        return len(self.token_to_idx)


# 向量器：输入和输出的词汇表类实例，使用词汇表标准化图像
class InsVectorizer(object):
    def __init__(self, ins_vocab, packer_vocab):
        self.ins_vocab = ins_vocab
        self.packer_vocab = packer_vocab

    def vectorize(self, ins_s):

        # 向量化每一个文件的汇编指令集序列
        indices = [
            self.ins_vocab.lookup_token(token) for token in ins_s.split(" ")
        ]

        indices = [self.ins_vocab.begin_seq_index
                   ] + indices + [self.ins_vocab.end_seq_index]

        # 创建向量
        ins_length = len(indices)
        vector = np.zeros(ins_length, dtype=np.int64)
        vector[:len(indices)] = indices

        return vector, ins_length

    # 反向量化
    def unvectorize(self, vector):
        tokens = [self.ins_vocab.lookup_index(index) for index in vector]
        ins_ = " ".join(token for token in tokens)
        return ins_

    @classmethod
    def from_dataframe(cls, df, cutoff):

        # 创建壳类别词汇表
        packer_vocab = Vocabulary()
        for packer in sorted(set(df.packer)):
            packer_vocab.add_token(packer)

        # 获取长度
        word_counts = Counter()
        for ins_ in df.ins:
            for token in ins_.split(" "):
                word_counts[token] += 1

        # 创建反汇编指令的词汇表实例
        ins_vocab = INS_SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                ins_vocab.add_token(word)

        return cls(ins_vocab, packer_vocab)

    @classmethod
    def from_serializable(cls, contents):
        ins_vocab = INS_SequenceVocabulary.from_serializable(
            contents['ins_vocab'])
        packer_vocab = Vocabulary.from_serializable(contents['packer_vocab'])
        return cls(ins_vocab=ins_vocab, packer_vocab=packer_vocab)

    def to_serializable(self):
        return {
            'ins_vocab': self.ins_vocab.to_serializable(),
            'packer_vocab': self.packer_vocab.to_serializable()
        }


# 数据集：提供向量化数据
class ImageDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

        # 最大汇编指令集长度
        self.max_seq_length = max(
            map(lambda ins_s: len(ins_s.split(" ")),
                df.ins)) + 2  # (<BEGIN> + <END>)

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
        return cls(df, InsVectorizer.from_dataframe(train_df, cutoff))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with vectorizer_filepath.open() as fp:
            return InsVectorizer.from_serializable(json.load(fp))

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
        ins_vector, ins_length = self.vectorizer.vectorize(row.ins)
        packer_index = self.vectorizer.packer_vocab.lookup_token(row.packer)
        return {
            'ins': ins_vector,
            'ins_length': ins_length,
            'packer': packer_index
        }

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self,
                         batch_size,
                         collate_fn,
                         shuffle=True,
                         drop_last=True,
                         device="cpu"):
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            drop_last=drop_last)
        for data_dict in tqdm(dataloader, desc=str(self)):
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


def get_ins_datasets(csv_path=sts.SAVE_CSV_PATH / "train_data_20190429.pkl",
                     randam_seed=None,
                     state_size=[0.7, 0.15, 0.15],
                     vectorize=None):

    if np.sum(state_size) != 1.0 or any([i < 0 for i in state_size]):
        raise Exception("np.sum({0}) != 1 or not integer".format(state_size))
    if randam_seed is not None:
        np.random.seed(randam_seed)

    train_df = pd.read_pickle(csv_path)
    df = train_df[['ins', 'packer']]

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
        dataset = ImageDataset.load_dataset_and_make_vectorizer(split_df, 5)
    else:
        dataset = ImageDataset.load_dataset_and_load_vectorizer(
            split_df, vectorize)
    return dataset


if __name__ == '__main__':
    datasets = get_ins_datasets(randam_seed=22)
    vector = datasets.vectorizer

    print(vector.ins_vocab)
    print(vector.packer_vocab)
    vectorized_ins, ins_length = vector.vectorize(
        "mov add ret retn jmp call or")
    print(np.shape(vectorized_ins))
    print(vectorized_ins)
    print(ins_length)
    print(vector.unvectorize(vectorized_ins))
    print(datasets)
    input_ = datasets[10]['ins']  # __getitem__
    print(input_)
    print(datasets.class_weights)
