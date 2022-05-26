import torch

def generateVoxelGrid(xLim, yLim, zLim, w, h, d):
    xRange = torch.linspace(*xLim, w)
    yRange = torch.linspace(*yLim, h)
    zRange = torch.linspace(*zLim, d)
    grid = torch.meshgrid(zRange, yRange, xRange)
    gridFlat = torch.stack((grid[2].reshape(-1), grid[1].reshape(-1), grid[0].reshape(-1)), 1)
    return gridFlat

class ObjectEncoder(torch.nn.Module):
    def __init__(self, pixelSceneEncoder, featureVolumeEncoder, xLim, yLim, zLim, w, h, d):
        super(ObjectEncoder, self).__init__()
        self.XW = w
        self.XH = h
        self.XD = d
        self.X = generateVoxelGrid(xLim, yLim, zLim, w, h, d)
        self.pixelSceneEncoder = pixelSceneEncoder
        self.featureVolumeEncoder = featureVolumeEncoder

    def forward(self, I, M, KT):
        B = I.shape[0]
        nM = M.shape[2]
        X = self.X.unsqueeze(0).expand(B, -1, -1) # B, xD*xH*xW, 3

        # compute aggregated feature volume for each object
        Y = self.pixelSceneEncoder(I, M, KT, X) # B, nM, xD*xH*xW, F
        F = Y.shape[-1]
        Y = Y.view((B*nM, self.XD, self.XH, self.XW, F)) # B*nM, XD, XH, XW, F
        Y = Y.permute(0, 4, 1, 2, 3) # B*nM, F, XD, XH, XW, TODO I think we can get rid of this permute by changing the image encoder, although it should for sure not hurt performance at all

        # latent vectors for each object
        z = self.featureVolumeEncoder(Y) # B*nM, k
        z = z.view((B,nM,-1)) # B, nM, k
        return z

    def predict_z_for_q(self, I, M, KT, q):
        """
        :param I: B, V, C, H, W
        :param M: B, V, nM, H, W
        :param KT: B, V, 4, 3
        :param q: B, 3
        :return: B, nM, k
        """
        B = I.shape[0]
        nM = M.shape[2]
        X = self.X.unsqueeze(0).repeat(B, 1, 1)  # B, xD*xH*xW, 3
        X = X - q.unsqueeze(1) # B, xD*xH*xW, 3
        # compute aggregated feature volume for each object
        Y = self.pixelSceneEncoder(I, M, KT, X)  # B, nM, xD*xH*xW, F
        F = Y.shape[-1]
        Y = Y.view((B * nM, self.XD, self.XH, self.XW, F))  # B*nM, XD, XH, XW, F
        Y = Y.permute(0, 4, 1, 2, 3)  # B*nM, F, XD, XH, XW

        # latent vectors for each object
        z = self.featureVolumeEncoder(Y)  # B*nM, k
        z = z.view((B, nM, -1))  # B, nM, k
        return z

    def to(self, *args, **kwargs):
        super(ObjectEncoder, self).to(*args, **kwargs)
        self.X = self.X.to(*args, **kwargs)
        return self
