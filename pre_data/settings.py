#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# Packer Raw Path
PACKER_RAW_PATH = Path(r'F:\my_packer\packed')

# Packer Save Yuri Path
PACKER_SAVE_YURI_PATH = Path(r'F:\my_packer\yuri')

# Save Images Path
SAVE_IMAGES_PATH = Path(r'F:\my_packer\train_image')

# Save csv Path
SAVE_CSV_PATH = Path(r'F:\my_packer\csv')

# Classifier Mal Path
CAL_MAL_PATH = Path(r'F:\Unpackme\cla')

# Images Channel
IMAGES_Channel = 3

# Disa Size
DISA_SIZE = 3

# CPU COUNT
CPU_COUNT = 1

# Packers Landspace
# PACKERS_LANDSPACE = ("ACProtect", "Armadillo", "ASProtect", "ASPacK", "Enigma",
#                      "EXECryptor", "FSG", "Molebox", "NsPack", "Obsidium",
#                     "PECompact", "PeSpin", "PEtite", "RLPack", "UPX")

PACKERS_LANDSPACE = {
    0: 'ASPacK',
    1: 'Molebox',
    2: 'NsPack',
    3: 'PECompact',
    4: 'PeSpin',
    5: 'UPX',
    6: 'Nopack'
}