#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import settings as sts
import subprocess


def get_files_list(Path):
    for root, _, files in os.walk(Path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            yield root, abs_file_path


def main():
    for path_, file_path in get_files_list(os.path.join(sts.PACKER_RAW_PATH, "UPX")):

        sub_path = path_.replace(sts.PACKER_RAW_PATH, '')
        save_csv_path = sts.PACKER_SAVE_CSV_PATH + sub_path
        if not os.path.exists(save_csv_path):
            os.makedirs(save_csv_path)

        command = '{bat} "{fp}" "{sp}"'.format(
            bat=sts.BATCH_PATH, fp=file_path, sp=save_csv_path)

        try:
            p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
            p.wait()
            print(p.stdout.read())
        except Exception:
            print("{} get worry.".format(file_path))


if __name__ == '__main__':
    main()
