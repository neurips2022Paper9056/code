import torch
import h5py

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filePath):
        DH5Py = h5py.File(filePath + '.hdf5', mode='r')
        self.I = DH5Py["I"]              # B, V, C, H, W images
        self.M = DH5Py["M"]              # B, V, nM, H, W masks with nM maximum number of objects in dataset
        self.KT = DH5Py["KT"]            # B, V, 4, 3 transposed camera matrix
        self.KInvT = DH5Py["KInvT"]      # B, V, 4, 3 transposed inverse camera matrix
        self.KBounds = DH5Py["KBounds"]  # B, V, 2

        self.len = self.I.shape[0]

        self.I = self.I[:]
        print('image')
        self.M = self.M[:]
        print('mask')
        self.KT = torch.from_numpy(self.KT[:])
        self.KInvT = torch.from_numpy(self.KInvT[:])
        self.KBounds = torch.from_numpy(self.KBounds[:])
        print('Loading data in memory complete')


    def __getitem__(self, index):
        return {
            'I': torch.from_numpy(self.I[index]).to(torch.float32)/255.0,
            'M': torch.from_numpy(self.M[index]).to(torch.float32)/255.0,
            'KT': self.KT[index],
            'KInvT': self.KInvT[index],
            'KBounds': self.KBounds[index]
        }

    def __len__(self):
        return self.len


