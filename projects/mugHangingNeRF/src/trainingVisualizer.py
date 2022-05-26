import torch
import numpy as np
from src.util import makeDirs
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TrainingVisualizer:

    def renderAndSaveImageMultipleViews(self, name, NSplit):
        with torch.no_grad():
            bInd = np.random.randint(0, self.comp3DDyn.dataset.len)
            bD = self.comp3DDyn.dataset[bInd:bInd + 1]
            bD = {key: tensor.to(self.comp3DDyn.device) for key, tensor in bD.items()}
            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3
            KInvT = bD['KInvT']  # B, V, 4, 3
            KBounds = bD['KBounds']  # B, V, 2
            B, V, C, H, W = I.shape
            
            if self.comp3DDyn.C['global']:
                M = (torch.sum(M, dim=2, keepdim=True) > 0) * 1.0 # B, V, 1, H, W

            z = self.comp3DDyn.objectEncoder(I=I, M=M, KT=KT)  # B, nM, k
            _, nM, k = z.shape

            maxV = 4
            KInvT = KInvT[:, 0:maxV].view(B * maxV, 4, 3)
            KBounds = KBounds[:, 0:maxV].view(B * maxV, 2)
            z = z.unsqueeze(1).expand(-1, maxV, -1, -1).view(B * maxV, nM, k)
            rgb = self.comp3DDyn.render(KInvT=KInvT, KBounds=KBounds, z=z, H=H, W=W, NSplit=NSplit)  # B*maxV, H, W, 3

            I = I.permute(0, 1, 3, 4, 2).view(-1, H, W, C)
            M = (torch.sum(M, dim=2) > 0).view(-1, H, W, 1)
            I = I * M
            fig, axs = plt.subplots(2, 4, figsize=(20, 7))
            for i in range(maxV):
                axs[0, i].imshow(rgb[i].cpu())
                axs[1, i].imshow(I[i].cpu())

            name = name + '.png'
            makeDirs(name)
            fig.savefig(name, dpi='figure', format='png', metadata=None, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            

    def renderAndSaveImageMultipleiewsCNNDecoder(self, name):
        with torch.no_grad():
            bInd = np.random.randint(0, self.comp3DDyn.dataset.len)
            bD = self.comp3DDyn.dataset[bInd:bInd + 1]
            bD = {key: tensor.to(self.comp3DDyn.device) for key, tensor in bD.items()}
            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3
            B, V, C, H, W = I.shape

            z = self.comp3DDyn.objectEncoder(I=I, M=M, KT=KT)  # B, nM, k
            _, nM, k = z.shape

            maxV = 4
            KT = KT[:, 0:maxV].reshape(B * maxV, 4, 3)
            z = z.unsqueeze(1).expand(-1, maxV, -1, -1).view(B * maxV, nM, k)

            zAggr = self.comp3DDyn.latentStateAggreator(z)  # B*maxV, k_imageDec
            rgb = self.comp3DDyn.imageDecoder(zAggr, KT)  # B*maxV, 3, H, W
            rgb = rgb.permute(0, 2, 3, 1)

            I = I.permute(0, 1, 3, 4, 2).view(-1, H, W, C)
            M = (torch.sum(M, dim=2) > 0).view(-1, H, W, 1)
            I = I * M
            fig, axs = plt.subplots(2, 4, figsize=(20, 7))
            for i in range(maxV):
                axs[0, i].imshow(rgb[i].cpu())
                axs[1, i].imshow(I[i].cpu())

            name = name + '.png'
            makeDirs(name)
            fig.savefig(name, dpi='figure', format='png', metadata=None, bbox_inches='tight', pad_inches=0.1)



    def renderAndSaveImageMultipleiewsCNNCNN(self, name):
        with torch.no_grad():
            bInd = np.random.randint(0, self.comp3DDyn.dataset.len)
            bD = self.comp3DDyn.dataset[bInd:bInd + 1]
            bD = {key: tensor.to(self.comp3DDyn.device) for key, tensor in bD.items()}
            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3
            B, V, C, H, W = I.shape

            if self.comp3DDyn.C['global']:
                M = (torch.sum(M, dim=2, keepdim=True) > 0)*1.0  # B, V, 1, H, W
                # nM = 1 if global is True!

            z = self.comp3DDyn.objectEncoder(I=I, M=M, KT=KT)  # B, nM, k
            _, nM, k = z.shape

            maxV = 4
            KT = KT[:, 0:maxV].reshape(B * maxV, 4, 3)
            z = z.unsqueeze(1).expand(-1, maxV, -1, -1).view(B * maxV, nM, k)

            zAggr = self.comp3DDyn.latentStateAggreator(z)  # B*maxV, k_imageDec
            rgb = self.comp3DDyn.imageDecoder(zAggr, KT)  # B*maxV, 3, H, W
            rgb = rgb.permute(0, 2, 3, 1)

            I = I.permute(0, 1, 3, 4, 2).view(-1, H, W, C)
            if not self.comp3DDyn.C['global']:
                M = (torch.sum(M, dim=2) > 0) # B, V, H, W
            M = M.view(-1, H, W, 1) # B*V, H, W, 1
            I = I * M
            fig, axs = plt.subplots(2, 4, figsize=(20, 7))
            for i in range(maxV):
                axs[0, i].imshow(rgb[i].cpu())
                axs[1, i].imshow(I[i].cpu())

            name = name + '.png'
            makeDirs(name)
            fig.savefig(name, dpi='figure', format='png', metadata=None, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

    def __init__(self, comp3DDyn):
        self.comp3DDyn = comp3DDyn
