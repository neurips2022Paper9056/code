import torch
import mugEnvironment
import numpy as np
import h5py
from src.util import makeDirs
import matplotlib.pyplot as plt

#dataName = "data/withAlternativeViews_1"
dataName = "data/withAlternativeViews_test_1"

#B = 10000
B = 1000
V = 8
H = 180
W = 180
maxM = 2
margin = 0.1

#np.random.seed(0)
np.random.seed(12345678) # for test_1

fileName = dataName + '.hdf5'
makeDirs(fileName)
hdf5File = h5py.File(fileName, mode='w')
hdf5File.create_dataset("I", (B, V, 3, H, W), np.uint8)    # B, V, C, H, W
hdf5File.create_dataset("M", (B, V, maxM, H, W), np.uint8) # B, V, M, H, W
hdf5File.create_dataset("KT", (B, V, 4, 3), np.float32)    # B, V, 4, 3
hdf5File.create_dataset("KInvT", (B, V, 4, 3), np.float32) # B, V, 4, 3
hdf5File.create_dataset("KBounds", (B, V, 2), np.float32)  # B, V, 2

env = mugEnvironment.MugEnvironment()
i = 0
while True:
    if i >= B:
        break
    env.initEnvironment(i)
    env.addAlternativeCameraObs()

    xLow, xHi, yLow, yHi, zLow, zHi = env.getBoundingBoxLimits()
    xLow += margin; yLow += margin; zLow += margin
    xHi -= margin;  yHi -= margin;  zHi -= margin
    x = np.random.uniform(xLow, xHi)
    y = np.random.uniform(yLow, yHi)
    z = np.random.uniform(zLow, zHi)
    env.setState(torch.Tensor([x, y, z]))
    success, coll = env.checkStateInSim()
    if not coll:
        I, M, KT, KInvT, KBounds = env.getObservation()

        hdf5File["I"][i] = (I[0]*255.0).to(torch.uint8).cpu().numpy()
        hdf5File["M"][i] = (M[0]*255.0).to(torch.uint8).cpu().numpy()
        hdf5File["KT"][i] = KT[0].cpu().numpy()
        hdf5File["KInvT"][i] = KInvT[0].cpu().numpy()
        hdf5File["KBounds"][i] = KBounds[0].cpu().numpy()
        i += 1
        print(i)

hdf5File.close()
