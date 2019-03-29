#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import settings as sts
from binascii import unhexlify
import pandas as pd
import numpy as np
import re
from PIL import Image


def get_sub_path_list(path):
    sub_path_list = []
    for dir_ in path.rglob("*"):
        if dir_.is_dir():
            sub_path_list.append(dir_)
    return sub_path_list


def get_df_opc_disa(file_path):
    datadict = {"addr": [], "opc": [], "disa": []}
    with open(str(file_path), "r") as file_read:
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


def get_disa(file_path):
    sub_file_path = file_path.relative_to(sts.PACKER_SAVE_YURI_PATH)
    save_disa_path = sts.SAVE_DISA_PATH / sub_file_path
    if not save_disa_path.exists():
        save_disa_path.mkdir(parents=True)

    for abs_file_path in file_path.glob("*"):
        if not abs_file_path.is_file():
            continue
        file_pd = get_df_opc_disa(abs_file_path)
        opc_ = []
        for opc in file_pd.opc:
            str_ = unhexlify(
                opc.ljust(sts.DISA_SIZE * 2, "0")[:sts.DISA_SIZE * 2])
            opc_.append([c for c in str_])
        while len(opc_) < 1024:
            opc_.append([0, 0, 0])
        array_to_img(save_disa_path / abs_file_path.with_suffix(".png").name,
                     opc_)


def main():
    mp.freeze_support()
    sub_path_list = get_sub_path_list(sts.PACKER_SAVE_YURI_PATH)
    # print(sub_path_list)
    p = mp.Pool(sts.CPU_COUNT)
    p.map(get_disa, sub_path_list)


if __name__ == '__main__':
    main()
