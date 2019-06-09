#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import os


class TimeDes(object):
    def __init__(self, value):
        self.value = value

    def __get__(self, instance, owner):
        if not isinstance(self.value, int):
            raise TypeError("timeout must be integer")
        return self.value

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError("timeout must be integer")
        if value < 5:
            raise ValueError("timeout must be >= 5")
        self.value = value


class StrDes(object):
    def __init__(self, name, a_string):
        self.name = name
        self.a_string = a_string

    def __get__(self, instance, owner):
        if not isinstance(self.a_string, str):
            raise TypeError("{} must be string".format(self.name))
        return self.a_string

    def __set__(self, instance, a_string):
        if not isinstance(a_string, str):
            raise TypeError("{} must be string".format(self.name))
        self.a_string = a_string


class Sandbox(object):
    timeout = TimeDes(5)
    vmrun_path = StrDes("vmrun_path", None)
    vmx_path = StrDes("vmx_path", None)
    vm_snapshot = StrDes("vm_snapshot", None)
    vm_user = StrDes("vm_user", None)
    vm_pass = StrDes("vm_pass", None)
    script_path = StrDes("script_path", None)
    python_path = StrDes("python_path", None)
    malware_path = StrDes("malware_path", None)

    def __init__(self,
                 vmrun_path,
                 vmx_path,
                 vm_snapshot,
                 vm_user,
                 vm_pass,
                 script_path,
                 python_path,
                 malware_path,
                 timeout=10):
        self.timeout = timeout
        self.vmrun_path = vmrun_path
        self.vmx_path = vmx_path
        self.vm_snapshot = vm_snapshot
        self.vm_user = vm_user
        self.vm_pass = vm_pass
        self.script_path = script_path
        self.python_path = python_path
        self.malware_path = malware_path

    # 恢复快照
    def _revertToSnapshot(self):
        print("--> revertToSnapshot")
        command = '{0} -T ws revertToSnapshot "{1}" {2}'.format(
            self.vmrun_path, self.vmx_path, self.vm_snapshot)
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    # 启动虚拟机
    def _start(self, no_gui):
        print("--> start")
        command = '{0} start "{1}"'.format(self.vmrun_path, self.vmx_path)
        if no_gui:
            command = command + ' nogui'
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    # 把文件从主机复制到虚拟机
    def _copyFileFromHostToGuest(self, file_path, file_name):
        print("--> copyFileFromHostToGuest")
        command = '{0} -gu {1} -gp {2} copyFileFromHostToGuest "{3}" "{4}" "{5}"'.format(
            self.vmrun_path, self.vm_user, self.vm_pass, self.vmx_path,
            file_path, os.path.join(self.malware_path, file_name))
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    # 运行虚拟机中的脚本
    def _runProgramInGuest(self, file_name):
        print("--> runProgramInGuest")
        command = '{0} -T ws -gu {1} -gp {2} runProgramInGuest "{3}" "{4}" "{5}" -t {6} -f {7}'.format(
            self.vmrun_path, self.vm_user, self.vm_pass, self.vmx_path,
            self.python_path, self.script_path, self.timeout, file_name)
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    # 把预处理内容从虚拟机复制到主机
    def _copyFileFromGuestToHost(self, save_path, file_name):
        print("--> copyFileFromGuestToHost")
        command = '{0} -gu {1} -gp {2} copyFileFromGuestToHost "{3}" "{4}" "{5}"'.format(
            self.vmrun_path, self.vm_user, self.vm_pass, self.vmx_path,
            os.path.join(self.malware_path, (file_name + ".yuri")),
            os.path.join(save_path, file_name + ".yuri"))
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    # 关闭虚拟机
    def _stop(self):
        print("--> stop")
        command = '{0} -T ws stop "{1}" hard'.format(self.vmrun_path,
                                                     self.vmx_path)
        p = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        print(stdout.decode('utf-8'), stderr.decode('utf-8'))

    def get_features(self, packer_file_path, no_gui=True, stop=True):

        file_path = Path(packer_file_path)
        if not file_path.is_file():
            return "{0} is not file.".format(packer_file_path)
        file_name = file_path.name
        file_path = file_path.parent

        try:
            self._revertToSnapshot()
            self._start(no_gui)
            self._copyFileFromHostToGuest(packer_file_path, file_name)
            self._runProgramInGuest(file_name)
            self._copyFileFromGuestToHost(str(file_path), file_name)
            if stop:
                self._stop()
            print("--> get features")
            with open(packer_file_path + ".yuri", "r") as file_read:
                features = file_read.read()
            os.remove(packer_file_path + ".yuri")
            print("--> completed!")
        except Exception as e:
            print(e)
            print("{0} get worry.".format(packer_file_path))
            return None
        else:
            return features


if __name__ == '__main__':
    sb = Sandbox(
        vmrun_path=r"E:\VMware\vmrun.exe",
        vmx_path=r"D:\虚拟机\Win10\Windows 10 x64.vmx",
        vm_snapshot="real",
        vm_user="msi",
        vm_pass="123456",
        script_path=r"C:\Users\msi\Desktop\my_sandbox_script.py",
        python_path=r"C:\Users\msi\AppData\Local\Programs\Python\Python35\python.exe",
        malware_path=r"C:\Malware",
        timeout=10)
    features = sb.get_features(r"C:\Users\msi\Desktop\aspack变形.exe")
    print(features)
