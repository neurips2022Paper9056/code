import torch
from src.resnet import resnet18

class ObjectEncoder2D(torch.nn.Module):
    # similar architecture as in https://github.com/YunzhuLi/comp_nerf_dy/blob/1da9eeb63115cb4219efffa1c0a99b76af6cb0c7/nerf_dy/models.py#L123
    # but adapted to work with multiple objects and their masks
    def __init__(self, imageEncoder, k, F):
        super().__init__()
        self.imageEncoder = imageEncoder

        self.combinedImageCameraEncodingNetwork = torch.nn.Sequential(
            torch.nn.Linear(self.imageEncoder.F_I + 12, 2*F),
            torch.nn.ReLU(),
            torch.nn.Linear(2*F, F),
            torch.nn.ReLU()
        )
        self.finalEncodingNetwork = torch.nn.Sequential(
            torch.nn.Linear(F, 2*k),
            torch.nn.ReLU(),
            torch.nn.Linear(2*k, 2*k),
            torch.nn.ReLU(),
            torch.nn.Linear(2*k, k)
        )

    def forward(self, I, M, KT):
        """
        :param I: B, V, C, H, W
        :param M: B, V, nM, H, W
        :param KT: B, V, 4, 3
        :return: B, nM, k
        """
        B, V, C, H, W = I.shape
        nM = M.shape[2]
        I = I.unsqueeze(2).expand(-1, -1, nM, -1, -1, -1) # B, V, nM, C, H, W
        M = M.unsqueeze(3)
        I = I*M # B, V, nM, C, H, W
        E_I = self.imageEncoder(I.view(B*V*nM, C, H, W)) # B*V*nM, F_I
        KT = KT.reshape(B*V, 4*3).unsqueeze(1).expand(-1, nM, -1).reshape(B*V*nM, 4*3) # B*V*nM, 4*3

        E_K = torch.cat([E_I, KT], dim=1) # B*V*nM, F_I + 4*3

        E = self.combinedImageCameraEncodingNetwork(E_K)
        F = E.shape[1]
        E = E.view(B, V, nM, F) # B, V, nM, F
        E = torch.mean(E, dim=1).view(B*nM, F) # B*nM, F
        z = self.finalEncodingNetwork(E) # B*nM, k
        k = z.shape[1]
        return z.view(B, nM, k) # B, nM, k


class ObjectEncoder2DSingleView(torch.nn.Module):
    # similar architecture as in https://github.com/YunzhuLi/comp_nerf_dy/blob/1da9eeb63115cb4219efffa1c0a99b76af6cb0c7/nerf_dy/models.py#L123
    def __init__(self, imageEncoder, k, F):
        super().__init__()
        self.imageEncoder = imageEncoder
        F = self.imageEncoder.F_I

        self.finalEncodingNetwork = torch.nn.Sequential(
            torch.nn.Linear(F, 2*k),
            torch.nn.ReLU(),
            torch.nn.Linear(2*k, 2*k),
            torch.nn.ReLU(),
            torch.nn.Linear(2*k, k)
        )

    def forward(self, I, M, KT):
        """
        :param I: B, V, C, H, W
        :param M: B, V, nM, H, W
        :param KT: not used, just for compatibility
        :return: B, nM, k
        """
        B, V, C, H, W = I.shape
        nM = M.shape[2]
        I = I.unsqueeze(2).expand(-1, -1, nM, -1, -1, -1) # B, V, nM, C, H, W
        M = M.unsqueeze(3)
        I = I*M # B, V, nM, C, H, W
        E = self.imageEncoder(I.view(B*V*nM, C, H, W)) # B*V*nM, F

        F = E.shape[1]
        E = E.view(B, V, nM, F) # B, V, nM, F
        E = torch.mean(E, dim=1).view(B*nM, F) # B*nM, F
        z = self.finalEncodingNetwork(E) # B*nM, k
        k = z.shape[1]
        return z.view(B, nM, k) # B, nM, k


class ResnetImageEncoder_1(torch.nn.Module):
    #adapted from https://github.com/YunzhuLi/comp_nerf_dy/blob/1da9eeb63115cb4219efffa1c0a99b76af6cb0c7/nerf_dy/models.py
    def __init__(self, F_I, fcDim=17920):
        super().__init__()
        self.F_I = F_I
        self.encoder = resnet18()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fcDim, 2*F_I),
            torch.nn.ReLU(),
            torch.nn.Linear(2*F_I, F_I))

    def forward(self, I):
        """
        :param I: B, C, H, W
        :return: B, k
        """
        B, C, H, W = I.shape

        x = self.encoder.conv1(I)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = x.view(B, -1)
        x = self.fc(x) # B, F_I

        return x


class ConvImageEncoder_1(torch.nn.Module):
    def __init__(self, F_I, fcDim):
        super().__init__()
        self.F_I = F_I
        self.convNet = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(fcDim, 2*F_I),
            torch.nn.ReLU(),
            torch.nn.Linear(2*F_I, F_I))

    def forward(self, I):
        """
        :param I: B, C, H, W
        :return: B, k
        """
        B = I.shape[0]
        x = self.convNet(I)
        x = x.view(B, -1)
        x = self.fc(x) # B, F_I

        return x
