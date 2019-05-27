#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import multiprocessing as mp
import settings as sts

import pandas as pd


def get_files_list(Path):
    file_list = []
    for root, _, files in os.walk(Path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            file_list.append(abs_file_path)
    return file_list


def get_csv_opc_disa(file_path, save_csv_file):
    datadict = {"addr": [], "opc": [], "disa": []}
    with open(file_path, "r") as file_read:
        for line in file_read:
            matchObj = re.match(r'(.*) (.*) --> (.*)', line, re.I)
            if matchObj:
                datadict["addr"].append(matchObj.group(1))
                datadict["opc"].append(matchObj.group(2))
                datadict["disa"].append(matchObj.group(3))
    df = pd.DataFrame(datadict)
    df.to_csv(save_csv_file, index=False)


def get_addr_opc_disa(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    sub_file_path = file_path.replace(sts.PACKER_RAW_PATH, '').replace(
        os.path.basename(file_path), '')
    save_csv_path = sts.PACKER_SAVE_CSV_PATH + sub_file_path

    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)

    dst_path = os.path.join(save_csv_path, file_name)
    if os.path.exists(dst_path + '.csv'):
        return

    command = 'pin -t {pt} -o {fn}.yuri -- {pp}'.format(
        pt=sts.PINTOOL86_PATH, pp=file_path, fn=file_name)
    pwd = os.getcwd() + "\\"

    try:
        os.system(command)
        if os.path.exists(pwd + file_name + '.yuri'):
            print("{} completed!".format(file_name))
            get_csv_opc_disa(pwd + file_name + '.yuri', dst_path + '.csv')
            os.remove(pwd + file_name + '.yuri')
        else:
            print("{} failed.".format(file_name))
    except Exception:
        print("{} get worry.".format(file_name))


if __name__ == '__main__':
    mp.freeze_support()
    files_list = get_files_list(sts.PACKER_RAW_PATH)
    print("Total Packer Count : {}".format(len(files_list)))
    p = mp.Pool(sts.CPU_COUNT)
    p.map(get_addr_opc_disa, files_list)
