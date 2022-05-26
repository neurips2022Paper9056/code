import torch
from src.contrastiveLoss import computeContrastiveLoss, soft_update_params, randomCrop
import numpy as np
import json
from src.dataset import Dataset
from src.objectEncoder2D import ResnetImageEncoder_1, ConvImageEncoder_1
from src.util import makeDirs
import sys

class ConPre:

    def eval(self, dataPath=None, dataset=None):
        # YOU HAVE TO SET EVAL MODE MANUALLY IF DESIRED!
        if dataPath is not None:
            dataset = Dataset(dataPath)
        else:
            assert dataset is not None, "dataset must be specified"
            
        testLoader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.C['batchSize'])
        totalLoss = 0.0
        bTot = len(testLoader)
        batchCounter = 0
        for b, bD in enumerate(testLoader):
            with torch.no_grad():
                loss = self.computeLoss(bD)

            totalLoss += loss.item()
            batchCounter += 1
            print(b / bTot * 100.0, totalLoss / batchCounter)
            sys.stdout.flush()

        return totalLoss / batchCounter


    def computeLoss(self, bD):
        I = bD['I'][:, self.C['viewIndex']].to(self.device)  # B, C, H, W

        if self.multipleLatentVectors:
            M = bD['M'][:, self.C['viewIndex']].to(self.device)  # B, nM, H, W
            tmp = torch.cat((I, M), dim=1)
            tmpQuery = randomCrop(tmp, self.C['HC'], self.C['WC'])
            tmpTarget = randomCrop(tmp, self.C['HC'], self.C['WC'])
            B, _, HC, WC = tmpQuery.shape
            C = I.shape[1]
            nM = M.shape[1]
            IQuery = tmpQuery[:, 0:C].unsqueeze(1).expand(-1, nM, -1, -1, -1).reshape(B * nM, C, HC, WC)
            MQuery = tmpQuery[:, C:].unsqueeze(2).reshape(B * nM, 1, HC, WC)
            IQuery = IQuery * MQuery  # B*nM, C, HC, WC

            ITarget = tmpTarget[:, 0:C].unsqueeze(1).expand(-1, nM, -1, -1, -1).reshape(B * nM, C, HC, WC)
            MTarget = tmpTarget[:, C:].unsqueeze(2).reshape(B * nM, 1, HC, WC)
            ITarget = ITarget * MTarget  # B*nM, C, HC, WC
        else:
            IQuery = randomCrop(I, self.C['HC'], self.C['WC'])  # B, C, H, W
            ITarget = randomCrop(I, self.C['HC'], self.C['WC'])  # B, C, H, W
            B = IQuery.shape[0]
            nM = 1

        zQuery = self.objectEncoder(IQuery).view(B, nM, self.C['k'])
        with torch.no_grad():
            zTarget = self.objectEncoderTarget(ITarget).view(B, nM, self.C['k'])

        loss = computeContrastiveLoss(zQuery, zTarget, self.CURL_W)
        return loss.mean()

    def trainEpoch(self, epoch):
        totalLoss = 0.0
        bTot = len(self.trainLoader)
        batchCounter = 0
        for b, bD in enumerate(self.trainLoader):
            self.optimizer.zero_grad()

            loss = self.computeLoss(bD)

            loss.backward()
            self.optimizer.step()

            soft_update_params(self.objectEncoder, self.objectEncoderTarget, self.C['encoderUpdateTau'])

            totalLoss += loss.item()
            batchCounter += 1
            print(epoch, b/bTot*100.0, totalLoss/batchCounter)
            sys.stdout.flush()

        return totalLoss/batchCounter

    def trainNetwork(self, numberOfEpochs):
        self.setTrain()
        Loss = []
        for epoch in range(numberOfEpochs):
            loss = self.trainEpoch(epoch)
            Loss.append(loss)
            self.saveNetwork(self.C['expPath'] + '/' + str(epoch))
            json.dump(Loss, open(self.C['expPath'] + '/' + str(epoch) + '_loss.json', 'w'))

    def prepareForTraining(self):
        self.setTrain()
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

    def makeEncoder(self):
        if self.C['imageEncoderNetwork'] == 'ResnetImageEncoder_1':
            return ResnetImageEncoder_1(F_I=self.C['k'], fcDim=self.C['fcDim'])
        elif self.C['imageEncoderNetwork'] == 'ConvImageEncoder_1':
            return ConvImageEncoder_1(F_I=self.C['k'], fcDim=self.C['fcDim'])

    def setTrain(self):
        self.objectEncoder.train()
        self.objectEncoderTarget.train()

    def setEval(self):
        self.objectEncoder.eval()
        self.objectEncoderTarget.eval()

    def setRandomSeed(self):
        if self.C['randomSeed'] >= 0:
            torch.manual_seed(self.C['randomSeed'])
            np.random.seed(self.C['randomSeed'])

    def buildModel(self):
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

        self.C['viewIndex'] = 0

        self.C['encoderUpdateTau'] = 0.05
        self.C['max_nM_duringTraining'] = 1

        self.C['H'] = 150
        self.C['W'] = 200
        self.C['imageCropFactor'] = 0.84
        
        self.C['imageEncoderNetwork'] = 'ResnetImageEncoder_1'
        self.C['fcDim'] = 12288

        self.C['k'] = 128

        self.C['learningRate'] = 0.0001
        self.C['batchSize'] = 10

        self.C['randomSeed'] = 0

        if configPath is not None:
            C = json.load(open(configPath + '/config.json'))

        self.C.update(C)

        self.C['HC'] = round(self.C['H']*self.C['imageCropFactor'])
        self.C['WC'] = round(self.C['W']*self.C['imageCropFactor'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.multipleLatentVectors = self.C['max_nM_duringTraining'] > 1



if __name__ == "__main__":
    conPre = ConPre(dataPath="dataGeneration/data/nM_1", expPath="exp/CURL", max_nM_duringTraining=1, imageEncoderNetwork='ConvImageEncoder_1', fcDim=8512)
    conPre.setRandomSeed()
    conPre.buildModel()
    conPre.prepareForTraining()
    conPre.saveConfig()
    conPre.trainNetwork(100000)
