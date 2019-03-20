#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yara import compile
import os
import shutil
import settings as sts


class YaraCheck(object):
    def __init__(self, rules_path):
        super(YaraCheck, self).__init__()
        self.Rules = self.setRules(os.getcwd() + rules_path)

    def setRules(self, path):
        yaraRule = compile(path)
        return yaraRule

    def scan(self, filename):
        with open(filename, "rb") as fin:
            bdata = fin.read()
        matches = self.Rules.match(data=bdata)
        # print matches
        # for i in matches:
        #    print(i.rule, i.tags)
        return [i.rule for i in matches]


def get_files_list(Path):
    for root, _, files in os.walk(Path):
        for name in files:
            abs_file_path = os.path.join(root, name)
            yield abs_file_path, name


def clas_file(file_path, yc):
    try:
        return yc.scan(file_path)
    except Exception:
        return None


def mycopyfile(srcfile, save_path, file_name):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copyfile(srcfile, os.path.join(save_path, file_name))
    except Exception:
        pass


if __name__ == '__main__':
    yc = YaraCheck(r"\pre_data\packer.yara")
    for file_path, file_name in get_files_list(r"F:\spider"):
        ans = clas_file(file_path, yc)
        # print(ans)
        if len(ans) == 0:
            continue
        ams_l = " ".join(ans).lower()
        for packer in sts.PACKERS_LANDSPACE:
            if packer.lower() in ams_l:
                if packer == "ASProtect" and "yoda" in ams_l:
                    continue
                save_path = os.path.join(sts.CAL_MAL_PATH, packer)
                mycopyfile(file_path, save_path, file_name)
                break
