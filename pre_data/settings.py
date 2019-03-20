#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Packer Raw Path
PACKER_RAW_PATH = os.path.normpath(
    os.path.abspath(r'F:\Unpackme\PE32'))

# Packer Save CSV Path
PACKER_SAVE_CSV_PATH = os.path.normpath(
    os.path.abspath(r'F:\train'))

# Save PKI Sets Path
SAVE_PKL_SETS_PATH = os.path.normpath(
    os.path.abspath(r'F:\train_image'))

# Pintool Path
PINTOOL86_PATH = os.path.normpath(
    os.path.abspath(r'F:\test\ximo_packer\ximo\PECompact2.X\itrace_x86.dll'))
PINTOOL64_PATH = os.path.normpath(
    os.path.abspath(r'F:\test\ximo_packer\ximo\PECompact2.X\itrace_x64.dll'))

# Classifier Mal Path
CAL_MAL_PATH = os.path.normpath(os.path.abspath(r'F:\Unpackme\cla'))

# Batch Path
BATCH_PATH = os.path.normpath(
    os.path.abspath(r'F:\course\gp\Packer-Classifier\pre_data\my_batch.bat'))

# Images Channel
IMAGES_Channel = 3

# CPU COUNT
CPU_COUNT = 2

# Packers Landspace
# ASProtect && Yoda
PACKERS_LANDSPACE = (
    "ASPacK",
    "UPX",
    "PECompact",
    "Upack",
    "PEtite",
    "NsPack",
    "FSG",
    "ASProtect",
    "Armadillo",
    "EXECryptor",
    "Themida",
    "VMProtect",
    "ACProtect",
    "DBPE",
    "Enigma",
    "Stealth",
    "FoxLock",
    "Krypton",
    "Obsidium",
    "Armor",
    "PElock",
    "PESpin",
    "tElock",
    "VFP",
    "Yoda",
    "RCryptor",
    "ReCrypt",
    "RLPack",
    "Thinstall",
    "VProtector",
    "ZProtect",
    "nPack",
    "MPress_",
    "MoleBox",
    "FishPE",
    "eXPressor",
    "ExeShield",
)
