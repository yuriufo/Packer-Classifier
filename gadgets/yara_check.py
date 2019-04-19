#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yara import compile
from pathlib import Path
import os


class YaraCheck(object):
    def __init__(self, rules_path):
        super(YaraCheck, self).__init__()
        self.Rules = self.setRules(rules_path)

    def setRules(self, path):
        yaraRule = compile(path)
        return yaraRule

    def scan(self, filename):
        with filename.open("rb") as fin:
            bdata = fin.read()
        matches = self.Rules.match(data=bdata)
        # print matches
        # for i in matches:
        #    print(i.rule, i.tags)
        return [i.rule for i in matches]


def get_files_list(path):
    p = Path(path)
    for abs_file_path in p.rglob("*"):
        if abs_file_path.is_file():
            yield abs_file_path


def clas_file(file_path, yc):
    try:
        return yc.scan(file_path)
    except Exception:
        return None


if __name__ == '__main__':
    num = 0
    yc = YaraCheck(r"./gadgets/packer.yar")
    for file_path in get_files_list(r"F:\my_packer\packed\Molebox\Molebox_2.63"):
        ans = clas_file(file_path, yc)
        # print(ans)
        if len(ans) == 0:
            os.remove(str(file_path))
            num = num + 1
            continue
        ams_l = " ".join(ans).lower()
        # if 'mole' not in ams_l:
            # num = num + 1
            # os.remove(str(file_path))
    print(num)
