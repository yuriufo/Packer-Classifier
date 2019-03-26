#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yara import compile
from pathlib import Path
import shutil
import settings as sts


class YaraCheck(object):
    def __init__(self, rules_path):
        super(YaraCheck, self).__init__()
        self.Rules = self.setRules(str(Path(r"\pre_data\packer.yar")))

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


def mycopyfile(srcfile, save_path):
    try:
        if not save_path.exists():
            save_path.mkdir(parents=True)
        shutil.copyfile(srcfile, save_path / (srcfile.name))
    except Exception:
        pass


if __name__ == '__main__':
    yc = YaraCheck()
    for file_path in get_files_list(r"F:\spider"):
        ans = clas_file(file_path, yc)
        # print(ans)
        if len(ans) == 0:
            continue
        ams_l = " ".join(ans).lower()
        for packer in sts.PACKERS_LANDSPACE:
            if packer.lower() in ams_l:
                if packer == "ASProtect" and "yoda" in ams_l:
                    continue
                save_path = sts.CAL_MAL_PATH / packer
                mycopyfile(file_path, save_path)
                break
