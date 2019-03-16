#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

__all__ = [
    "PACKER_RAW_PATH", "PACKER_SAVE_CSV_PATH", "PINTOOL86_PATH",
    "PINTOOL64_PATH", "CPU_COUNT", "SAVE_PKL_SETS_PATH", "CAL_MAL_PATH",
    "PACKERS_LANDSPACE"
]

# Packer Raw Path
PACKER_RAW_PATH = os.path.normpath(
    os.path.abspath(r'C:\Users\msi\Desktop\packer\1'))

# Packer Save CSV Path
PACKER_SAVE_CSV_PATH = os.path.normpath(
    os.path.abspath(r'C:\Users\msi\Desktop\packer\2'))

# Save PKI Sets Path
SAVE_PKL_SETS_PATH = os.path.normpath(
    os.path.abspath(r'C:\Users\msi\Desktop\packer\3'))

# Pintool Path
PINTOOL86_PATH = os.path.normpath(
    os.path.abspath(r'F:\test\ximo_packer\ximo\PECompact2.X\itrace2_x86.dll'))
PINTOOL64_PATH = os.path.normpath(
    os.path.abspath(r'F:\test\ximo_packer\ximo\PECompact2.X\itrace2_x64.dll'))

# Classifier Mal Path
CAL_MAL_PATH = os.path.normpath(
    os.path.abspath(r'C:\Users\msi\Desktop\packer\4'))

# CPU COUNT
CPU_COUNT = 2

# Packers Landspace
# ASProtect && Yoda
PACKERS_LANDSPACE = ("ASPacK", "UPX", "PECompact", "Upack", "PEtite", "NsPack",
                     "FSG", "ASProtect", "Armadillo", "EXECryptor", "Themida",
                     "VMProtect", "ACProtect", "DBPE", "Enigma", "Stealth",
                     "FoxLock", "Krypton", "Obsidium", "Armor", "PElock",
                     "PESpin", "tElock", "VFP", "Yoda")
