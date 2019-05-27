#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pre_data.settings as sts
from binascii import unhexlify
import numpy as np
import re


class Preproce(object):
    def __init__(self):
        pass

    def _check(self, features):
        if any([
                len(features) < 5, "this is dll" in features,
                "this is not PE" in features, "error" in features
        ]):
            return 1
        else:
            return 0

    def _get_df_opc_disa(self, features):
        datadict = {"opc": [], "disa": []}
        for line in features.split("\n"):
            matchObj = re.match(r'(.*) (.*) --> (.*)', line, re.I)
            if matchObj:
                datadict["opc"].append(matchObj.group(2))
                datadict["disa"].append(matchObj.group(3))
        return datadict

    def preprocess(self, features):
        if self._check(features):
            print("get error! " + features)
            return None
        file_dict = self._get_df_opc_disa(features)

        data_dict = {"image": [], "ins": []}

        # 机器码预处理
        opc_ = np.zeros((32, 32, 3))
        num = 0
        for opc in file_dict['opc']:
            str_ = unhexlify(
                opc.ljust(sts.DISA_SIZE * 2, "0")[:sts.DISA_SIZE * 2])
            opc_[num // 32][num % 32] = [c for c in str_]
            num = num + 1

        # 反汇编指令预处理
        ins_ = []
        for disa in file_dict['disa']:
            ins_.append(disa.split(" ")[0].strip())

        data_dict["image"].append(np.asarray(opc_, dtype="float32"))
        data_dict["ins"].append(" ".join(ins_))

        return data_dict
