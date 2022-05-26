import torch

class LatentStateAggregator_1(torch.nn.Module):
    def __init__(self, k, k_imageDec):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(k, k_imageDec),
            torch.nn.ReLU(),
            torch.nn.Linear(k_imageDec, k_imageDec),
            torch.nn.ReLU(),
            torch.nn.Linear(k_imageDec, k_imageDec),
        )

    def forward(self, z):
        # B, nM, k = z.shape
        z = torch.mean(z, dim=1) # B, k
        return self.network(z) # B, k_imageDec


class ImageDecoder_1(torch.nn.Module):
    # adapted from https://github.com/YunzhuLi/comp_nerf_dy/blob/1da9eeb63115cb4219efffa1c0a99b76af6cb0c7/nerf_dy/models.py#L83
    def __init__(self, k_imageDec, H, W):
        super().__init__()
        self.H = H
        self.W = W
        nf = k_imageDec
        self.network = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nf + 12, nf, 4, 1, 0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm2d(nf),
            torch.nn.ConvTranspose2d(nf, nf, 4, 2, 0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm2d(nf),
            torch.nn.ConvTranspose2d(nf, nf // 2, 3, 2, 0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm2d(nf // 2),
            torch.nn.ConvTranspose2d(nf // 2, nf // 2, 4, 2, 0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm2d(nf // 2),
            torch.nn.ConvTranspose2d(nf // 2, nf // 4, 3, 2, 0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm2d(nf // 4),
            torch.nn.ConvTranspose2d(nf // 4, 3, 4, 2, 0)
        )

    def forward(self, zAggr, KT):
        B, k_imageDec = zAggr.shape
        KT = KT.reshape(B, 3*4)
        z = torch.cat([zAggr, KT], dim=1) # B, k_imageDec + 12
        return self.network(z.reshape(B, k_imageDec+12, 1, 1))#[:,:,0:self.H,0:self.W]
