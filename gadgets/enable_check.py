#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os


def get_files_list(path):
    p = Path(path)
    for abs_file_path in p.rglob("*"):
        if abs_file_path.is_file():
            yield abs_file_path


if __name__ == '__main__':
    file_list = set()
    num = 0
    for file_path in get_files_list(r"F:\my_packer\yuri\Nopack"):
        file_list.add(str(file_path.name))

    for file_path in get_files_list(r"F:\my_packer\packed\Nopack"):
        if str(file_path.name) not in file_list:
            num = num + 1
            os.remove(str(file_path))

    print(num)
