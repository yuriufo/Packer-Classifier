#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from pre_data import settings as sts
import subprocess

# 参数
config = {
    "timueout":
    50,
    "vmrun_path":
    Path(r"E:\VMware\vmrun.exe"),
    "vmx_path":
    Path(r"D:\虚拟机\Win10\Windows 10 x64.vmx"),
    "vm_snapshot":
    r"real",
    "vm_user":
    r"msi",
    "vm_pass":
    r"123456",
    "script_path":
    Path(r"C:\Users\msi\Desktop\my_sandbox_script.py"),
    "python_path":
    Path(r"C:\Users\msi\AppData\Local\Programs\Python\Python35\python.exe"),
    "malware_path":
    Path(r"C:\Malware")
}


def get_files_list(path):
    for dir_ in path.rglob("*"):
        if dir_.is_file():
            yield dir_, dir_.name


def revertToSnapshot():
    command = '{0} -T ws revertToSnapshot "{1}" {2}'.format(
        config["vmrun_path"], config["vmx_path"], config["vm_snapshot"])
    p = subprocess.Popen(command, shell=True)
    p.wait()


def start():
    command = '{0} -T ws start "{1}" nogui'.format(config["vmrun_path"],
                                                   config["vmx_path"])
    p = subprocess.Popen(command, shell=True)
    p.wait()


def copyFileFromHostToGuest(file_path, file_name):
    command = '{0} -gu {1} -gp {2} copyFileFromHostToGuest "{3}" "{4}" "{5}"'.format(
        config["vmrun_path"], config["vm_user"], config["vm_pass"],
        config["vmx_path"], file_path, config["malware_path"] / file_name)
    p = subprocess.Popen(command, shell=True)
    p.wait()


def runProgramInGuest(file_name):
    command = '{0} -T ws -gu {1} -gp {2} runProgramInGuest "{3}" "{4}" "{5}" -t {6} -f {7}'.format(
        config["vmrun_path"], config["vm_user"], config["vm_pass"],
        config["vmx_path"], config["python_path"], config["script_path"],
        config["timueout"], file_name)
    p = subprocess.Popen(command, shell=True)
    p.wait()


def copyFileFromGuestToHost(save_path, file_name):
    command = '{0} -gu {1} -gp {2} copyFileFromGuestToHost "{3}" "{4}" "{5}"'.format(
        config["vmrun_path"], config["vm_user"], config["vm_pass"],
        config["vmx_path"], config["malware_path"] / (file_name + ".yuri"),
        save_path / file_name)
    p = subprocess.Popen(command, shell=True)
    p.wait()


def stop():
    command = '{0} -T ws stop "{1}"'.format(config["vmrun_path"],
                                            config["vmx_path"])
    p = subprocess.Popen(command, shell=True)
    p.wait()


def main():
    for file_path, file_name in get_files_list(
            sts.PACKER_RAW_PATH / r"PE32\FSG"):

        sub_path = file_path.parents.relative_to(sts.PACKER_RAW_PATH)
        save_path = sts.PACKER_SAVE_YURI_PATH / sub_path
        if not save_path.exists():
            save_path.mkdir(parents=True)

        try:
            revertToSnapshot()
            start()
            copyFileFromHostToGuest(file_path, file_name)
            runProgramInGuest(file_name)
            copyFileFromGuestToHost(save_path, file_name)
            print("{} completed!".format(file_path))
        except Exception:
            print("{} get worry.".format(file_path))
    else:
        try:
            stop()
        except Exception:
            print("exit worry.")
        else:
            print("all completed!")


if __name__ == '__main__':
    main()
