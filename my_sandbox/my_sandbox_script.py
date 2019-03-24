#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import argparse
import sys
import time
import os
import pefile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeout', default=10, type=int)
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('-o', '--output', default=r"C:\Users\msi\Desktop")
    args = parser.parse_args()

    message = "error"

    file_path = "C:\\Malware\\" + args.filename
    try:
        pe = pefile.PE(file_path, fast_load=True)
        if pe.FILE_HEADER.IMAGE_FILE_DLL is True:
            message = "this is dll"
            command = ""
        elif pe.FILE_HEADER.IMAGE_FILE_32BIT_MACHINE is True:
            command = "pin -t {pt32} -o {fn}.yuri -- {fn}".format(
                pt32=r"C:\pin-3.7\itrace_x86.dll", fn=file_path)
        else:
            command = "pin -t {pt64} -o {fn}.yuri -- {fn}".format(
                pt64=r"C:\pin-3.7\itrace_x64.dll", fn=file_path)
    except Exception:
        message = "this is not PE"

    try:
        p = subprocess.Popen(command)
        for i in range(args.timeout):
            if p.poll() is not None:
                break
            progress = (100 / args.timeout) * i
            sys.stdout.write('\r%d%% complete' % progress)
            sys.stdout.flush()
            time.sleep(1)
        else:
            p.terminate()
            sys.stdout.write('timeout')
            sys.stdout.flush()
    except Exception:
        pass

    output_path = os.path.join(r"C:\Malware", args.filename + ".yuri")
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write(message)
    sys.exit(0)


if __name__ == '__main__':
    main()
