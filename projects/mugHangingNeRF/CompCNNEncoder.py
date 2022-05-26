import torch
import numpy as np
import json
from src.dataset import Dataset
from src.objectEncoder2D import ObjectEncoder2D, ResnetImageEncoder_1, ConvImageEncoder_1
from src.nerfRender import *
import src.nerfs
from src.util import makeDirs
from src.trainingVisualizer import TrainingVisualizer
import sys

class CompCNNEncoderNeRF:

    def render(self, KInvT, KBounds, z, H, W, NSplit = None):
        """
        :param KInvT: B, 4, 3
        :param KBounds: B, 2
        :param z: B, nM, k
        :param H:
        :param W:
        :param NSplit: number of pixels per render batch. If None, no split.
        :return: B, H, W, 3
        """
        B = KInvT.shape[0]
        uv = generateUVGridFlat(w=W, h=H, device=self.device)  # H*W, 2
        uv = uv.unsqueeze(0).expand(B, -1, -1)  # B, H*W, 2

        if NSplit is None:
            NSplit = B

        uvChunked = torch.split(uv, NSplit, dim=1)
        RGB = []
        for uvChunk in uvChunked:   # uvChunk: B, N, 2 with N <= NSplit
            N = uvChunk.shape[1]
            NP = self.C['NP']

            t, x_w = get_x_c_rayPointsFrom_uv(KInvT, KBounds, uvChunk, NP) # t: B*N, NP, 1;    x_w: B*N, NP, 3

            #B*N*NP       B*N          (B*N)'*NP
            insideIndsNP, insideIndsN, insideIndsNPN = getRayPointInsideBoundingBoxIndices(x_w, self.boundingBoxLimitLow, self.boundingBoxLimitHigh)

            if insideIndsNPN is None:
                RGB.append(torch.zeros(B, N, 3, device=x_w.device))
                continue

            t = t[insideIndsN]  # (B*N)', NP, 1
            x_w = x_w.view(B * N * NP, 3)[insideIndsNP]  # (B*N*NP)', 3

            _, nM, k = z.shape
            zChunk = z.unsqueeze(1).unsqueeze(1).expand(-1, N, -1, -1, -1).expand(-1, -1, NP, -1, -1).reshape(B * N * NP, nM, k)
            zChunk = zChunk[insideIndsNP]  # (B*N*NP)', nM, k

            sigma, c = self.nerf(x_w, zChunk)  # sigma: (B*N*NP)', nM, 1; c: (B*N*NP)', nM, 3
            sigmaTotal, cTotal = compositeMultipleNerfs(sigma, c) # sigma: (B*N*NP)', 1; c: (B*N*NP)', 3

            sigmaTotal, cTotal = scatterSparseNerfResultIntoRayNPChunk(sigmaTotal, cTotal, insideIndsNPN, NP) # (B*N)', NP, 1;   (B*N)', NP, 3

            w = calcWeights(t, sigmaTotal)  # (B*N)', NP, 1
            rgb = torch.sum(w * cTotal, dim=1)  # (B*N)', 3

            rgbM = torch.zeros(B * N, 3, device=rgb.device)
            rgbM[insideIndsN] = rgb
            rgb = rgbM.view(B, N, 3)
            RGB.append(rgb)

        return torch.cat(RGB, dim=1).view(B, H, W, 3)



    def trainEpochChunkedMasked(self, epoch):
        bTot = len(self.trainLoader)
        for b, bD in enumerate(self.trainLoader):
            bD = {key: tensor.to(self.device) for key, tensor in bD.items()}
            self.optimizer.zero_grad()

            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3
            KInvT = bD['KInvT']  # B, V, 4, 3
            KBounds = bD['KBounds']  # B, V, 2
            B, V, C, H, W = I.shape

            if self.C['global']:
                M = (torch.sum(M, dim=2, keepdim=True) > 0) * 1.0 # B, V, 1, H, W

            viewIndex = np.random.randint(0, V)

            I_R = I[:, viewIndex]  # B, C, H, W
            I_R = I_R.permute(0, 2, 3, 1).view(B, H*W, C) # B, H*W, C

            M_R = M[:, viewIndex]  # B, nM, H, W
            M_R = torch.sum(M_R, dim=1) > 0  # B, H, W "union" of masks

            M_R_enlarged = enlargeMasks(M_R.unsqueeze(1), 29).reshape(B, H*W) # B, H*W

            M_R = M_R.view(B, H*W, 1) # B, H*W, 1

            KInvT_R = KInvT[:, viewIndex]  # B, 4, 3
            KBounds_R = KBounds[:, viewIndex]  # B, 2

            uv = generateUVGridFlat(w=W, h=H, device=self.device)  # H*W, 2
            uv = uv.unsqueeze(0).expand(B, -1, -1)  # B, H*W, 2

            NSplit = self.C['NSplit']
            uvChunked = torch.split(uv, NSplit, dim=1)
            I_R_chunked = torch.split(I_R, NSplit, dim=1)
            M_R_chunked = torch.split(M_R, NSplit, dim=1)
            M_R_enlarged_chunked = torch.split(M_R_enlarged, NSplit, dim=1)
            batchLoss = 0.0

            for uvChunk, I_R_chunk, M_R_chunk, M_R_enlarged_chunk in zip(uvChunked, I_R_chunked, M_R_chunked, M_R_enlarged_chunked):
                N = uvChunk.shape[1]
                NP = self.C['NP']

                t, x_w = get_x_c_rayPointsFrom_uv(KInvT_R, KBounds_R, uvChunk, NP)  # t: B*N, NP, 1;    x_w: B*N, NP, 3

                # B*N*NP       B*N          (B*N)'*NP
                insideIndsNP, insideIndsN, insideIndsNPN = getRayPointInsideBoundingBoxIndices(x_w, self.boundingBoxLimitLow, self.boundingBoxLimitHigh, M_R_enlarged_chunk)

                if insideIndsNPN is None:
                    continue

                t = t[insideIndsN] # (B*N)', NP, 1
                x_w = x_w.view(B*N*NP,3)[insideIndsNP] # (B*N*NP)', 3

                z = self.objectEncoder(I=I, M=M, KT=KT)  # B, nM, k
                _, nM, k = z.shape
                z = z.unsqueeze(1).unsqueeze(1).expand(-1, N, -1, -1, -1).expand(-1, -1, NP, -1, -1).reshape(B * N * NP, nM, k)
                z = z[insideIndsNP] # (B*N*NP)', nM, k

                sigma, c = self.nerf(x_w, z)  # sigma: (B*N*NP)', nM, 1; c: (B*N*NP)', nM, 3
                sigmaTotal, cTotal = compositeMultipleNerfs(sigma, c)  # sigma: (B*N*NP)', 1; c: (B*N*NP)', 3

                sigmaTotal, cTotal = scatterSparseNerfResultIntoRayNPChunk(sigmaTotal, cTotal, insideIndsNPN, NP)  # (B*N)', NP, 1;   (B*N)', NP, 3

                w = calcWeights(t, sigmaTotal)  # (B*N)', NP, 1
                rgb = torch.sum(w * cTotal, dim=1)  # (B*N)', 3

                rgbM = torch.zeros(B * N, 3, device=rgb.device)
                rgbM[insideIndsN] = rgb
                rgb = rgbM.view(B, N, 3)

                target = I_R_chunk * M_R_chunk # I_R_chunk B, N, 3      M_R_chunk B, N, 1
                loss = (target - rgb).square()
                loss = loss.mean()
                loss = loss / len(uvChunked)
                batchLoss += loss.item()
                loss.backward()

            self.optimizer.step()
            print(epoch, b/bTot*100.0, batchLoss)
            sys.stdout.flush()


    def trainNetwork(self, numberOfEpochs):
        self.setTrain()
        for epoch in range(numberOfEpochs):
            self.trainEpochChunkedMasked(epoch)
            self.saveNetwork(self.C['expPath'] + '/' + str(epoch))
            self.setEval()
            self.trainingVisualizer.renderAndSaveImageMultipleViews(self.C['expPath'] + "/imgs/" + str(epoch), self.C['NSplit'])
            self.setTrain()

    def prepareForTraining(self):
        self.setTrain()
        params = list(self.objectEncoder.parameters())
        params += list(self.nerf.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.C['learningRate'])

        self.dataset = Dataset(self.C['dataPath'])
        self.trainLoader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.C['batchSize'])

        self.trainingVisualizer = TrainingVisualizer(self)

    def saveNetwork(self, name):
        name = name + '.pt'
        makeDirs(name)
        stateDict = {
            'nerfStateDict': self.nerf.state_dict(),
            'objectEncoderStateDict': self.objectEncoder.state_dict()
        }
        torch.save(stateDict, name)

    def loadNetwork(self, name):
        checkpoint = torch.load(name + '.pt', map_location=self.device)
        self.nerf.load_state_dict(checkpoint['nerfStateDict'])
        self.objectEncoder.load_state_dict(checkpoint['objectEncoderStateDict'])

    def saveConfig(self):
        makeDirs(self.C['expPath'])
        with open(self.C['expPath'] + '/config.json', 'w') as outfile:
            json.dump(self.C, outfile, indent=2, sort_keys=True)

    def setRandomSeed(self):
        if self.C['randomSeed'] >= 0:
            torch.manual_seed(self.C['randomSeed'])
            np.random.seed(self.C['randomSeed'])

    def buildModel(self):
        if self.C['imageEncoderNetwork'] == 'ResnetImageEncoder_1':
            CNN = ResnetImageEncoder_1(F_I=self.C['F_I']) 
        elif self.C['imageEncoderNetwork'] == 'ConvImageEncoder_1':
            CNN = ConvImageEncoder_1(F_I=self.C['F_I'], fcDim=self.C['imageEncoder_fCDim'])
        
        self.objectEncoder = ObjectEncoder2D(imageEncoder=CNN, k=self.C['k'], F=self.C['F'])

        self.nerf = getattr(src.nerfs, self.C['nerfNetworkSettings']['networkModule'])(**self.C['nerfNetworkSettings'])

        self.objectEncoder.to(self.device)
        self.nerf.to(self.device)

        self.boundingBoxLimitLow = torch.Tensor([self.C['xLim'][0], self.C['yLim'][0], self.C['zLim'][0]]).to(device=self.device).reshape(1, 1, 3)
        self.boundingBoxLimitHigh = torch.Tensor([self.C['xLim'][1], self.C['yLim'][1], self.C['zLim'][1]]).to(device=self.device).reshape(1, 1, 3)

    def setEval(self):
        self.nerf.eval()
        self.objectEncoder.eval()

    def setTrain(self):
        self.nerf.train()
        self.objectEncoder.train()

    def __init__(self, **C):
        self.C = {}

        self.C['dataPath'] = None
        self.C['expPath'] = None

        self.C['xLim'] = [-0.125 - 0.25, -0.125 + 0.25]
        self.C['yLim'] = [-0.2 - 0.25, -0.2 + 0.25]
        self.C['zLim'] = [0.5 - 0.25, 0.5 + 0.25]
        self.C['XW'] = 40
        self.C['XH'] = 40
        self.C['XD'] = 40

        self.C['F_I'] = 128
        self.C['F'] = 128
        self.C['k'] = 64

        self.C['NP'] = 100
        
        self.C['global'] = False
        
        self.C['imageEncoderNetwork'] = 'ResnetImageEncoder_1'

        self.C['nerfNetworkSettings'] = {
            'networkModule': 'NeRF_2',
            'sigmaActivation' : 'softplus'
        }

        self.C['learningRate'] = 0.0001
        self.C['batchSize'] = 3
        self.C['NSplit'] = 16200

        self.C['randomSeed'] = 0

        self.C.update(C)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




if __name__ == "__main__":
    ae = CompCNNEncoderNeRF(dataPath="data/1", expPath="exp/NeRF-RL-image")
    ae.setRandomSeed()
    ae.buildModel()
    ae.prepareForTraining()
    ae.saveConfig()
    ae.trainNetwork(100000)
