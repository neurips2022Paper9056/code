import torch

def projectWorldToCameraCoord(KT, x_w):
    """
    :param KT: transposed camera matrix B, 4, 3
    :param x_w: query point in world coordinates B, N, 3
    :return: points in camera coordinates B, N, 3 (range [-1, 1] for uv and true depth)
    """
    tmp = torch.bmm(x_w, KT[:,0:3,:]) + KT[:,3:,:]
    tmp[:,:,0:2] /= tmp[:,:,2:]
    return tmp


class PixelSceneEncoder(torch.nn.Module):
    def __init__(self, imageEncoderNetwork, F_x_c, finalEncoderHiddenLayers, F):
        super(PixelSceneEncoder, self).__init__()
        self.imageEncoderNetwork = imageEncoderNetwork

        self.cameraCoordinateEncoderNetwork = torch.nn.Sequential(
            torch.nn.Linear(3, F_x_c),
            torch.nn.ReLU()
        )

        finalEncoderLayerList = [torch.nn.Linear(F_x_c + imageEncoderNetwork.F_I, finalEncoderHiddenLayers[0]), torch.nn.ReLU()]
        for i in range(1, len(finalEncoderHiddenLayers)):
            finalEncoderLayerList.extend([torch.nn.Linear(finalEncoderHiddenLayers[i-1], finalEncoderHiddenLayers[i]), torch.nn.ReLU()])
        finalEncoderLayerList.append(torch.nn.Linear(finalEncoderHiddenLayers[-1], F))
        self.combinedFeatureEncoderNetwork = torch.nn.Sequential(*finalEncoderLayerList)

    def forward(self, I, M, KT, x):
        B, V, C, H, W = I.shape
        nM = M.shape[2]
        N = x.shape[1]
        M = M.reshape(B*V, nM, H, W)
        KT = KT.reshape(B * V, 4, 3)

        I = I.reshape(B * V, C, H, W)
        E = self.imageEncoderNetwork(I)  # B*V, F_I, H, W
        F_I = E.shape[1]

        # x = torch.repeat_interleave(x, V, dim=0) # B*V, N, 3 ## repeat_interleave is slow
        x = x.unsqueeze(1).expand(-1, V, -1, -1).reshape(B*V, N, 3) # B*V, N, 3
        x_c = projectWorldToCameraCoord(KT, x) # B*V, N, 3

        uv_c = x_c[:, :, 0:2].unsqueeze(2) # B*V, N, 1, 2
        E_x_c = torch.nn.functional.grid_sample(E, uv_c,
                                                mode='bilinear',
                                                padding_mode='zeros',
                                                align_corners=False) # B*V, F_I, N, 1

        E_x_c = E_x_c.squeeze(-1).transpose(1,2).reshape(B*V*N, F_I) # B*V*N, F_I
        f_x_c = self.cameraCoordinateEncoderNetwork(x_c.view(B*V*N, 3)) # B*V*N, F_x_c

        F_x_c = self.combinedFeatureEncoderNetwork(torch.cat([E_x_c, f_x_c], dim=1)) # B*V*N, F


        # get masks at x_c (or uv_c to be precise)
        M_x_c = torch.nn.functional.grid_sample(M, uv_c,
                                                mode='nearest',
                                                padding_mode='zeros',
                                                align_corners=False)  # B*V, nM, N, 1

        M_x_c = M_x_c.view(B, V, nM, N, 1)
        F = F_x_c.shape[-1]
        F_x_c = F_x_c.view(B, V, 1, N, F).expand(-1, -1, nM, -1, -1)  # B, V, nM, N, F

        MSum = torch.sum(M_x_c, dim=1) # B, nM, N, 1
        MSum[MSum == 0.0] = 1e-4 # value should not matter, since it is multiplied with zero anyway
        F_x_c = torch.sum(F_x_c * M_x_c, dim=1) / MSum # B, nM, N, F

        #E_x_c = E_x_c.squeeze(-1).transpose(1,2).view([B, V, 1, N, F]).expand(-1, -1, nM, -1, -1) # B, V, nM, N, F
        #M_x_c = M_x_c.view([B, V, nM, N, 1])
        #F_x_c = torch.sum(E_x_c*M_x_c, dim=1)# / torch.sum(M_x_c, dim=1) # B, nM, N, F

        return F_x_c



class IdentityImageEncoder(torch.nn.Module):
    """
    Image encoder that returns the original image.
    This is just for testing purposes
    """
    def __init__(self):
        super(IdentityImageEncoder, self).__init__()
        self.F_I = 3

    def forward(self, x):
        return x


