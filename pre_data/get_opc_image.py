#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import multiprocessing as mp
import settings as sts
from binascii import unhexlify
import pandas as pd
import numpy as np
import re
from PIL import Image


def get_sub_path_list(Path):
    sub_path_list = []
    for root, dirs, files in os.walk(Path):
        for d in dirs:
            sub_path_list.append(os.path.join(root, d))
    return sub_path_list


def get_csv_opc_disa(file_path):
    datadict = {"addr": [], "opc": [], "disa": []}
    with open(file_path, "r") as file_read:
        for line in file_read:
            matchObj = re.match(r'(.*) (.*) --> (.*)', line, re.I)
            if matchObj:
                datadict["addr"].append(matchObj.group(1))
                datadict["opc"].append(matchObj.group(2))
                datadict["disa"].append(matchObj.group(3))
    df = pd.DataFrame(datadict)
    return df


def array_to_img(fp, opc):
    img = Image.fromarray(np.uint8(opc).reshape(32, 32, 3)).convert('RGB')
    img.save(fp)


def get_csv_sets(file_path):
    sub_file_path = file_path.replace(sts.PACKER_SAVE_CSV_PATH, '')
    save_pkl_sets_path = sts.SAVE_PKL_SETS_PATH + sub_file_path
    if not os.path.exists(save_pkl_sets_path):
        os.makedirs(save_pkl_sets_path)

    for root, _, files in os.walk(file_path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            file_pd = get_csv_opc_disa(abs_file_path)
            opc_ = []
            for opc in file_pd.opc:
                str_ = unhexlify(
                    opc.ljust(sts.IMAGES_Channel * 2,
                              "0")[:sts.IMAGES_Channel * 2])
                opc_.append([c for c in str_])
            while len(opc_) < 1024:
                opc_.append([0, 0, 0])
            array_to_img(os.path.join(save_pkl_sets_path, name+".png"), opc_)


def main():
    mp.freeze_support()
    sub_path_list = get_sub_path_list(sts.PACKER_SAVE_CSV_PATH)
    print(sub_path_list)
    p = mp.Pool(sts.CPU_COUNT)
    p.map(get_csv_sets, sub_path_list)


if __name__ == '__main__':
    main()
