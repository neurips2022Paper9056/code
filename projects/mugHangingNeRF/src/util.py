import os
import torch
import numpy as np

def makeDirs(path):
    root, ext = os.path.splitext(path)
    if not ext:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir) and dir != '':
        os.makedirs(dir)
