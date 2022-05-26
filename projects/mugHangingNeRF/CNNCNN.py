import torch
import numpy as np
import json
from src.dataset import Dataset
from src.objectEncoder2D import ObjectEncoder2D, ResnetImageEncoder_1
from src.nerfRender import *
from src.CNNDecoder import *
from src.util import makeDirs
from src.trainingVisualizer import TrainingVisualizer

class CNNCNN:

    def render(self, KT, z):
        """
        :param KT: B, 4, 3
        :param z: B, nM, k
        :return: B, H, W, 3
        """
        zAggr = self.latentStateAggreator(z)  # B*maxV, k_imageDec
        rgb = self.imageDecoder(zAggr, KT)  # B*maxV, 3, H, W
        rgb = rgb.permute(0, 2, 3, 1)
        return rgb

    def trainEpoch(self, epoch):
        bTot = len(self.trainLoader)
        for b, bD in enumerate(self.trainLoader):
            bD = {key: tensor.to(self.device) for key, tensor in bD.items()}
            self.optimizer.zero_grad()

            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3
            B, V, C, H, W = I.shape

            if self.C['global']:
                M = (torch.sum(M, dim=2, keepdim=True) > 0)*1.0  # B, V, 1, H, W
                # nM = 1 if global is True!

            z = self.objectEncoder(I=I, M=M, KT=KT)  # B, nM, k
            _, nM, k = z.shape

            viewIndex = np.random.randint(0, V)

            I_R = I[:, viewIndex]  # B, C, H, W

            M_R = M[:, viewIndex]  # B, nM, H, W
            if not self.C['global']:
                M_R = torch.sum(M_R, dim=1, keepdim=True) > 0  # B, 1, H, W "union" of masks

            KT_R = KT[:, viewIndex] # B, 4, 3

            batchLoss = 0.0

            zAggr = self.latentStateAggreator(z) # B, k_imageDec
            rgb = self.imageDecoder(zAggr, KT_R) # B, 3, H, W

            if self.C['lossOnMaskOnly']:
                M_R_enlarged = enlargeMasks(M_R, 9)  # B, 1, H, W
                rgb = rgb*M_R_enlarged

            target = I_R * M_R # I_R B, 3, H, W      M_R B, 1, H, W
            loss = (target - rgb).square()
            loss = loss.mean()
            batchLoss += loss.item()
            loss.backward()

            self.optimizer.step()
            print(epoch, b/bTot*100.0, batchLoss)

    def trainNetwork(self, numberOfEpochs):
        for epoch in range(numberOfEpochs):
            self.trainEpoch(epoch)
            self.saveNetwork(self.C['expPath'] + '/' + str(epoch))
            self.trainingVisualizer.renderAndSaveImageMultipleiewsCNNCNN(self.C['expPath'] + "/imgs/" + str(epoch))


    def prepareForTraining(self):
        params = list(self.objectEncoder.parameters())
        params += list(self.latentStateAggreator.parameters())
        params += list(self.imageDecoder.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.C['learningRate'])

        self.dataset = Dataset(self.C['dataPath'])
        self.trainLoader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.C['batchSize'])

        self.trainingVisualizer = TrainingVisualizer(self)

    def saveNetwork(self, name):
        name = name + '.pt'
        makeDirs(name)
        stateDict = {
            'latentStateAggreatorStateDict': self.latentStateAggreator.state_dict(),
            'imageDecoderStateDict': self.imageDecoder.state_dict(),
            'objectEncoderStateDict': self.objectEncoder.state_dict()
        }
        torch.save(stateDict, name)

    def loadNetwork(self, name):
        checkpoint = torch.load(name + '.pt', map_location=self.device)
        self.latentStateAggreator.load_state_dict(checkpoint['latentStateAggreatorStateDict'])
        self.imageDecoder.load_state_dict(checkpoint['imageDecoderStateDict'])
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
        CNN = ResnetImageEncoder_1(F_I=self.C['F_I'])
        self.objectEncoder = ObjectEncoder2D(imageEncoder=CNN, k=self.C['k'], F=self.C['F'])

        self.latentStateAggreator = LatentStateAggregator_1(k=self.C['k'], k_imageDec=self.C['k_imageDec'])
        self.imageDecoder = ImageDecoder_1(self.C['k_imageDec'], H=self.C['H'], W=self.C['W'])

        self.objectEncoder.to(self.device)
        self.latentStateAggreator.to(self.device)
        self.imageDecoder.to(self.device)

        self.boundingBoxLimitLow = torch.Tensor([self.C['xLim'][0], self.C['yLim'][0], self.C['zLim'][0]]).to(device=self.device).reshape(1, 1, 3)
        self.boundingBoxLimitHigh = torch.Tensor([self.C['xLim'][1], self.C['yLim'][1], self.C['zLim'][1]]).to(device=self.device).reshape(1, 1, 3)

    def __init__(self, **C):
        self.C = {}

        self.C['dataPath'] = None
        self.C['expPath'] = None
        self.C['trainingImgSavePath'] = None

        self.C['xLim'] = [-0.125-0.25, -0.125+0.25]
        self.C['yLim'] = [-0.2-0.25, -0.2+0.25]
        self.C['zLim'] = [0.5-0.25, 0.5+0.25]
        self.C['XW'] = 40
        self.C['XH'] = 40
        self.C['XD'] = 40

        self.C['H'] = 180
        self.C['W'] = 180

        self.C['F_I'] = 128
        self.C['F'] = 128
        self.C['k'] = 128

        self.C['k_imageDec'] = 128

        self.C['lossOnMaskOnly'] = False
        self.C['global'] = True

        self.C['learningRate'] = 0.0001
        self.C['batchSize'] = 32

        self.C['randomSeed'] = 0

        self.C.update(C)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    cnncnn = CNNCNN(dataPath="data/1", expPath="exp/ConvAutoencoder-global")
    cnncnn.setRandomSeed()
    cnncnn.buildModel()
    cnncnn.prepareForTraining()
    cnncnn.saveConfig()
    cnncnn.trainNetwork(100000)
