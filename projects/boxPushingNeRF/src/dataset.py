import torch
import h5py

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filePath):
        DH5Py = h5py.File(filePath + '.hdf5', mode='r')
        self.I = DH5Py["I"]              # B, V, C, H, W images
        self.M = DH5Py["M"]              # B, V, nM, H, W masks with nM maximum number of objects in dataset
        #self.nM = DH5Py["nM"]            # B, V, 1 number of objects in each view #TODO important to determine number of latent vectors in total for each batch
        self.K = DH5Py["K"]              # B, V, 3, 4 camera projection matrix, including intrinsics and extrinsics
                                         # such that x_c = P\bar{x}_w
                                         # will be transposed to be used as x^T*K for multiple x (rows)
        self.KBounds = DH5Py["KBounds"]  # B, V, 2

        self.len = self.I.shape[0]

        self.I = self.I[:]
        print('image')
        self.M = self.M[:]
        print('mask')
        self.K = torch.from_numpy(self.K[:]).transpose(2,3) # transposed KT B, V, 4, 3
        self.KBounds = torch.from_numpy(self.KBounds[:])
        self.KInvT = self.computeInverseCameraMatrix(self.K) # inverse of KT B, V, 4, 3

        print('Loading data in memory complete')

    def computeInverseCameraMatrix(self, KT):
        B, V = KT.shape[0:2]
        E = torch.eye(4).reshape(1,1,4,4).repeat(B, V, 1, 1)
        E[:,:,:4,0:3] = KT
        KInvT = torch.linalg.inv(E)
        KInvT = KInvT[:,:,:4,:3]
        return KInvT


    def __getitem__(self, index):
        return {
            'I': torch.from_numpy(self.I[index]).to(torch.float32)/255.0,
            'M': torch.from_numpy(self.M[index]).to(torch.float32)/255.0,
            'KT': self.K[index],
            'KInvT': self.KInvT[index],
            'KBounds': self.KBounds[index]
        }

    def __len__(self):
        return self.len


