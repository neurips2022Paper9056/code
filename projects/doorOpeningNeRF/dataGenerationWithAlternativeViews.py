import torch
import doorEnvironment
import numpy as np
import h5py
from src.util import makeDirs
import matplotlib.pyplot as plt

#dataName = "data/withAlternativeViews_1"
dataName = "data/withAlternativeViews_test_1"


#B = 50000
B = 5000
V = 8
H = 150
W = 200
maxM = 3

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

env = doorEnvironment.DoorEnvironment()

i = 0
while True:
    if i >= B:
        break
    env.initEnvironment(i)
    env.addAlternativeCameraObs()
    
    stateBounds = env.getStateBounds()
    doorOpening = np.random.uniform(stateBounds[0][0], stateBounds[0][1])
    x0 = np.random.uniform(stateBounds[1][0], stateBounds[1][1])
    y0 = -0.1
    z0 = np.random.uniform(stateBounds[3][0], stateBounds[3][1])
   
    env.setState(torch.Tensor([doorOpening, x0, y0, z0]))
    for _ in range(40):
        if i >= B:
            break
        env.simulateStep([-0.01, 0.01, 0.0])
        if not env.checkWithinBounds(): break
        I, M, KT, KInvT, KBounds = env.getObservation()
        
        #plt.imshow((I[0]*255.0).to(torch.uint8).cpu().permute(0,2,3,1)[3].numpy())
        #plt.show()
        hdf5File["I"][i] = (I[0]*255.0).to(torch.uint8).cpu().numpy()
        hdf5File["M"][i] = (M[0]*255.0).to(torch.uint8).cpu().numpy()
        hdf5File["KT"][i] = KT[0].cpu().numpy()
        hdf5File["KInvT"][i] = KInvT[0].cpu().numpy()
        hdf5File["KBounds"][i] = KBounds[0].cpu().numpy()
        
        i += 1
        print(i)

hdf5File.close()



