#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import multiprocessing as mp
import settings as sts
from binascii import unhexlify
import pandas as pd


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


def get_csv_sets(file_path):
    sub_file_path = file_path.replace(sts.PACKER_SAVE_CSV_PATH, '')
    save_pkl_sets_path = sts.SAVE_PKL_SETS_PATH + sub_file_path
    if not os.path.exists(save_pkl_sets_path):
        os.makedirs(save_pkl_sets_path)

    path_name = sub_file_path.replace("\\", "")
    save_file_name1 = os.path.join(save_pkl_sets_path, path_name + "_opc.pkl")
    save_file_name2 = os.path.join(save_pkl_sets_path, path_name + "_disa.pkl")
    if os.path.exists(save_file_name1) and os.path.exists(save_file_name2):
        return

    datadict_opc = {"opc": [], "packer": []}
    datadict_disa = {"disa": [], "packer": []}
    for root, _, files in os.walk(file_path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            file_csv = pd.read_csv(abs_file_path)
            opc_, disa_ = [], []
            for opc, disa in zip(file_csv.opc, file_csv.disa):
                str_ = unhexlify(opc.ljust(16, "0")[:16])
                opc_.append([c for c in str_])
                disa_.append(preprocess_text(disa).split(" "))
            datadict_opc["opc"].append(opc_)
            datadict_disa["disa"].append(disa_)
            datadict_opc["packer"].append(path_name)
            datadict_disa["packer"].append(path_name)
    df1 = pd.DataFrame(datadict_opc)
    df1.to_pickle(save_file_name1)
    df2 = pd.DataFrame(datadict_disa)
    df2.to_pickle(save_file_name2)


if __name__ == '__main__':
    mp.freeze_support()
    sub_path_list = get_sub_path_list(sts.PACKER_SAVE_CSV_PATH)
    print(sub_path_list)
    p = mp.Pool(sts.CPU_COUNT)
    p.map(get_csv_sets, sub_path_list)

    # file_csv = pd.read_pickle(
    #               r"C:\Users\msi\Desktop\packer\3\UPX\UPX_opc.pkl")
    # print(file_csv)
