#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from hashlib import md5


def get_files_list(path):
    p = Path(path)
    for abs_file_path in p.rglob("*"):
        if abs_file_path.is_file():
            yield abs_file_path


def calc_md5(str_):
    return md5(str_).hexdigest()


if __name__ == '__main__':
    md5_sets = set()
    for file_path in get_files_list(r'F:\my_packer\packed\PeSpin\other'):
        s = ""
        with file_path.open("rb") as f:
            s = calc_md5(f.read())
            if s in md5_sets:
                print("{} gets wrong.".format(file_path))
                print("md5: {}".format(s))
                exit(0)
            md5_sets.add(s)

        file_path.replace(file_path.with_name(s))
    else:
        print("ok")
