import torch
from src.contrastiveLoss import computeContrastiveLoss, soft_update_params
import numpy as np
import json
from src.dataset import Dataset
from src.imageEncoder import *
from src.featureVolumeEncoder import FeatureVolumeEncoder_1
from src.objectEncoder import ObjectEncoder
from src.util import makeDirs

class TimeConPre:
    def eval(self, dataPath):
        dataset = Dataset(dataPath)
        testLoader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.C['batchSize'])
        totalLoss = 0.0
        bTot = len(testLoader)
        batchCounter = 0
        for b, bD in enumerate(testLoader):
            bD = {key: tensor.to(self.device) for key, tensor in bD.items()}

            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3

            with torch.no_grad():
                zQuery = self.objectEncoder(I=I[:, 0:self.C['viewIndexSplit']],
                                            M=M[:, 0:self.C['viewIndexSplit']],
                                            KT=KT[:, 0:self.C['viewIndexSplit']])  # B, nM, k

                zTarget = self.objectEncoderTarget(I=I[:, self.C['viewIndexSplit']:],
                                                   M=M[:, self.C['viewIndexSplit']:],
                                                   KT=KT[:, self.C['viewIndexSplit']:])  # B, nM, k

                loss = computeContrastiveLoss(zQuery, zTarget, self.CURL_W)
                loss = loss.mean()
            totalLoss += loss.item()
            batchCounter += 1
            print(b / bTot * 100.0, totalLoss / batchCounter)
        return totalLoss / batchCounter

    def trainEpoch(self, epoch):
        totalLoss = 0.0
        bTot = len(self.trainLoader)
        batchCounter = 0
        for b, bD in enumerate(self.trainLoader):
            bD = {key: tensor.to(self.device) for key, tensor in bD.items()}
            self.optimizer.zero_grad()

            I = bD['I']  # B, V, C, H, W
            M = bD['M']  # B, V, nM, H, W
            KT = bD['KT']  # B, V, 4, 3

            zQuery = self.objectEncoder(I=I[:,0:self.C['viewIndexSplit']],
                                        M=M[:,0:self.C['viewIndexSplit']],
                                        KT=KT[:,0:self.C['viewIndexSplit']])  # B, nM, k
            with torch.no_grad():
                zTarget = self.objectEncoderTarget(I=I[:,self.C['viewIndexSplit']:],
                                                   M=M[:,self.C['viewIndexSplit']:],
                                                   KT=KT[:,self.C['viewIndexSplit']:])  # B, nM, k

            loss = computeContrastiveLoss(zQuery, zTarget, self.CURL_W)
            loss = loss.mean()

            loss.backward()
            self.optimizer.step()

            soft_update_params(self.objectEncoder, self.objectEncoderTarget, self.C['encoderUpdateTau'])

            totalLoss += loss.item()
            batchCounter += 1
            print(epoch, b/bTot*100.0, totalLoss/batchCounter)
        return totalLoss/batchCounter

    def trainNetwork(self, numberOfEpochs):
        Loss = []
        for epoch in range(numberOfEpochs):
            loss = self.trainEpoch(epoch)
            Loss.append(loss)
            self.saveNetwork(self.C['expPath'] + '/' + str(epoch))
            json.dump(Loss, open(self.C['expPath'] + '/' + str(epoch) + '_loss.json', 'w'))

    def prepareForTraining(self):
        params = list(self.objectEncoder.parameters())
        params += [self.CURL_W]

        self.optimizer = torch.optim.Adam(params, lr=self.C['learningRate'])

        self.dataset = Dataset(self.C['dataPath'])
        self.trainLoader = torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.C['batchSize'])

    def saveNetwork(self, name):
        name = name + '.pt'
        makeDirs(name)
        stateDict = {
            'objectEncoderStateDict': self.objectEncoder.state_dict(),
            'objectEncoderTargetStateDict': self.objectEncoderTarget.state_dict(),
            'CURL_W': self.CURL_W
        }
        torch.save(stateDict, name)

    def loadNetwork(self, name=None, networkNumber=None):
        if networkNumber is not None:
            assert name is None, "name and network number cannot be both specified"
            name = self.C['expPath'] + '/' + str(networkNumber)
        checkpoint = torch.load(name + '.pt', map_location=self.device)
        self.objectEncoder.load_state_dict(checkpoint['objectEncoderStateDict'])
        self.objectEncoderTarget.load_state_dict(checkpoint['objectEncoderTargetStateDict'])
        self.CURL_W = checkpoint['CURL_W']

    def saveConfig(self):
        makeDirs(self.C['expPath'])
        with open(self.C['expPath'] + '/config.json', 'w') as outfile:
            json.dump(self.C, outfile, indent=2, sort_keys=True)


    def makeImplictEncoder(self):
        pixelSceneEncoder = PixelSceneEncoder(imageEncoderNetwork=IdentityImageEncoder(),
                                              F_x_c=self.C['encoderNetworkSettings']['F_x_c'],
                                              finalEncoderHiddenLayers=self.C['encoderNetworkSettings']['pixelSceneEncoder_finalEncoderHiddenLayers'],
                                              F=self.C['encoderNetworkSettings']['F'])

        CNN = FeatureVolumeEncoder_1(F=self.C['encoderNetworkSettings']['F'], k=self.C['k'])
        return ObjectEncoder(pixelSceneEncoder=pixelSceneEncoder,
                                           featureVolumeEncoder=CNN,
                                           xLim=self.C['xLim'],
                                           yLim=self.C['yLim'],
                                           zLim=self.C['zLim'],
                                           w=self.C['XW'], h=self.C['XH'], d=self.C['XD'])

    def makeEncoder(self):
        if self.C['encoderNetwork'] == 'implicitEncoder':
            return self.makeImplictEncoder()
        else:
            raise NotImplementedError

    def buildModel(self):
        if self.C['randomSeed'] >= 0:
            torch.manual_seed(self.C['randomSeed'])
            np.random.seed(self.C['randomSeed'])

        self.objectEncoder = self.makeEncoder()
        self.objectEncoderTarget = self.makeEncoder()

        self.objectEncoder.to(self.device)
        self.objectEncoderTarget.to(self.device)

        self.objectEncoderTarget.load_state_dict(self.objectEncoder.state_dict())

        self.CURL_W = torch.rand(self.C['max_nM_duringTraining'], self.C['k'], self.C['k'], device=self.device).requires_grad_()

    def __init__(self, configPath = None, **C):
        self.C = {}

        self.C['dataPath'] = None
        self.C['expPath'] = None

        self.C['xLim'] = [-0.125-0.25, -0.125+0.25]
        self.C['yLim'] = [-0.2-0.25, -0.2+0.25]
        self.C['zLim'] = [0.5-0.25, 0.5+0.25]
        self.C['XW'] = 40
        self.C['XH'] = 40
        self.C['XD'] = 40

        self.C['viewIndexSplit'] = 4
        self.C['encoderUpdateTau'] = 0.05
        self.C['max_nM_duringTraining'] = 2

        self.C['encoderNetwork'] = 'implicitEncoder'
        self.C['encoderNetworkSettings'] = {
            'F_x_c': 32,
            'pixelSceneEncoder_finalEncoderHiddenLayers': [128, 128],
            'F': 64
        }

        self.C['k'] = 64

        self.C['learningRate'] = 0.0001
        self.C['batchSize'] = 10

        self.C['randomSeed'] = -1

        if configPath is not None:
            C = json.load(open(configPath + '/config.json'))

        self.C.update(C)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    conPre = TimeConPre(dataPath="data/withAlternativeViews_1", expPath="exp/Multi-CURL")
    conPre.buildModel()
    conPre.prepareForTraining()
    conPre.saveConfig()
    conPre.trainNetwork(100000)
