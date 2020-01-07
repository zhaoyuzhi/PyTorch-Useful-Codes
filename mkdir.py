# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:05:54 2019
@author: zhaoyuzhi
"""

import os

save_folder = './'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# single-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

# multi-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
