#!/usr/bin/env python 
# -*- coding: utf-8 -*- 

import os
import csv, re
import multiprocessing as mp
from .settings import *

# 获取路径下所有文件的绝对路径
def get_files_list(Path):
    file_list = []
    for root, _, files in os.walk(Path) :
        for name in files :
            abs_file_path = os.path.join(root, name)
            file_list.append(abs_file_path)
    return file_list

# 将数据转为CSV
def get_csv_opc_disa(file_path,save_csv_file):
    with open(save_csv_file, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["addr","opc","disa"])
        with open(file_path, "r") as file_read:
            for line in file_read:
                matchObj = re.match(r'(.*) (.*) --> (.*)', line, re.I)
                if matchObj:
                    writer.writerow([matchObj.group(1),matchObj.group(2),matchObj.group(3)])

def get_addr_opc_disa(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    sub_file_path = file_path.replace(PACKER_RAW_PATH,'').replace(os.path.basename(file_path), '')
    save_csv_path = PACKER_SAVE_CSV_PATH + sub_file_path

    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)

    dst_path = os.path.join(save_csv_path, file_name)
    if os.path.exists(dst_path + '.csv'):
        return

    command = 'pin -t {pintool_path} -- {packer_path} {file_name}.yuri'.format(pintool_path=PINTOOL_PATH, packer_path=file_path, file_name=file_name)
    pwd = os.getcwd() + "\\"

    try:
        os.system(command)
        if os.path.exists(pwd+file_name+'.yuri'):
            print("{} completed!".format(file_name))
            get_csv_opc_disa(pwd+file_name+'.yuri',dst_path + '.csv')
            os.remove(pwd+file_name+'.yuri')
        else:
            print("{} failed.".format(file_name))
    except:
        print("{} get worry.".format(file_name))

if __name__ == '__main__' :
    mp.freeze_support()
    files_list = get_files_list(PACKER_RAW_PATH)
    print("Total Packer Count : {}".format(len(files_list)))
    p = mp.Pool(CPU_COUNT)
    p.map(get_addr_opc_disa, files_list)