#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import settings as sts
from binascii import unhexlify
import pandas as pd
import numpy as np
import re


def get_sub_path_list(path):
    sub_path_list = []
    for dir_ in path.glob("*"):
        if dir_.is_dir():
            sub_path_list.append(dir_)
    return sub_path_list


def get_df_opc_disa(file_path):
    datadict = {"opc": [], "disa": []}
    with open(str(file_path), "r") as file_read:
        for line in file_read:
            matchObj = re.match(r'(.*) (.*) --> (.*)', line, re.I)
            if matchObj:
                datadict["opc"].append(matchObj.group(2))
                datadict["disa"].append(matchObj.group(3))
    return datadict


def get_csv(file_path, data_dict):
    sub_file_path = file_path.relative_to(sts.PACKER_SAVE_YURI_PATH)
    label = str(sub_file_path)

    for abs_file_path in file_path.rglob("*"):
        if not abs_file_path.is_file():
            continue
        file_dict = get_df_opc_disa(abs_file_path)

        opc_ = np.zeros((32, 32, 3))
        num = 0
        for opc in file_dict['opc']:
            str_ = unhexlify(
                opc.ljust(sts.DISA_SIZE * 2, "0")[:sts.DISA_SIZE * 2])
            opc_[num // 32][num % 32] = [c for c in str_]
            num = num + 1

        ins_ = []
        for disa in file_dict['disa']:
            ins_.append(disa.split(" ")[0].strip())

        data_dict["image"].append(np.asarray(opc_, dtype="float32"))
        data_dict["ins"].append(" ".join(ins_))
        data_dict["packer"].append(label)


def main():
    sub_path_list = get_sub_path_list(sts.PACKER_SAVE_YURI_PATH)
    # print(sub_path_list)
    data_dict = {"image": [], "ins": [], "packer": []}
    for sub_path in sub_path_list:
        get_csv(sub_path, data_dict)
    df = pd.DataFrame(data_dict)
    pd.to_pickle(df, sts.SAVE_CSV_PATH / 'train_data_20190429.pkl')


if __name__ == '__main__':
    main()
