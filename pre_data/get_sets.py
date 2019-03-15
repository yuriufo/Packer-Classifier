#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import multiprocessing as mp
import settings as sts
from binascii import unhexlify


def get_sub_path_list(Path):
    sub_path_list = []
    for root, dirs, files in os.walk(Path):
        for d in dirs:
            sub_path_list.append(os.path.join(root, d))
    return sub_path_list


def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = text.replace(",", "")
    text = text.strip()
    return text


def opc_2_list(text):
    text = text[2:-2].replace(" [", " ")
    text = [list(map(eval, word.split(", "))) for word in text.split("], ")]
    return text


def disa_2_list(text):
    text = text[2:-2].replace(" [", " ")
    text = [[i[1:-1] for i in word.split(", ")] for word in text.split("], ")]
    return text


def get_csv_sets(file_path):
    sub_file_path = file_path.replace(sts.PACKER_SAVE_CSV_PATH, '')
    save_csv_sets_path = sts.SAVE_CSV_SETS_PATH + sub_file_path
    if not os.path.exists(save_csv_sets_path):
        os.makedirs(save_csv_sets_path)

    path_name = sub_file_path.replace("\\", "")
    save_file_name1 = os.path.join(save_csv_sets_path, path_name + "_opc.csv")
    save_file_name2 = os.path.join(save_csv_sets_path, path_name + "_disa.csv")
    if os.path.exists(save_file_name1) and os.path.exists(save_file_name2):
        return

    with open(save_file_name1, "w", newline='') as csv_file1:
        with open(save_file_name2, "w", newline='') as csv_file2:
            csv_write1 = csv.writer(csv_file1)
            csv_write2 = csv.writer(csv_file2)
            csv_write1.writerow(["opc", "packer"])
            csv_write2.writerow(["disa", "packer"])
            for root, _, files in os.walk(file_path):
                for name in files:
                    abs_file_path = os.path.join(root, name)
                    with open(abs_file_path, "r") as csv_file3:
                        csv_reader = csv.reader(csv_file3)
                        opc_, disa_ = [], []
                        for i, row in enumerate(csv_reader):
                            if i == 0:
                                continue
                            str_ = unhexlify(row[1].ljust(16, "0")[:16])
                            opc_.append([c for c in str_])
                            sentences = preprocess_text(row[2]).split(" ")
                            disa_.append(sentences)
                        csv_write1.writerow([opc_, path_name])
                        csv_write2.writerow([disa_, path_name])


if __name__ == '__main__':
    mp.freeze_support()
    sub_path_list = get_sub_path_list(sts.PACKER_SAVE_CSV_PATH)
    print(sub_path_list)
    p = mp.Pool(sts.CPU_COUNT)
    p.map(get_csv_sets, sub_path_list)

    # import pandas as pd
    # import numpy as np
    # file_csv = pd.read_csv(r"C:\Users\msi\Desktop\packer\3\PEC\PEC_disa.csv")
    # print(disa_2_list(file_csv.disa[0]))
