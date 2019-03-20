#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import settings as sts
from hashlib import md5


def get_files_list(Path):
    for root, _, files in os.walk(Path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            yield root, abs_file_path


def calc_md5(str_):
    return md5(str_).hexdigest()


if __name__ == '__main__':
    md5_sets = set()
    for path_, file_path in get_files_list(r"F:\Unpackme\PE32"):
        s = ""
        with open(file_path, "rb") as f:
            s = calc_md5(f.read())
            if s in md5_sets:
                print("{} gets wrong.".format(file_path))
                print("md5: {}".format(s))
                exit(0)
            md5_sets.add(s)
        os.rename(file_path, os.path.join(path_, s))
    else:
        print("ok")
