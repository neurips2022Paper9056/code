import torch

def enlargeMasks(mask, kernelSize):
    """
    :param mask: B, 1, H, W
    """
    dilationKernel = torch.ones(1, 1, kernelSize, kernelSize, device=mask.device)
    size = (kernelSize-1)//2
    return torch.clamp(torch.nn.functional.conv2d(mask.float(), dilationKernel, padding=(size, size)), 0.0, 1.0).bool() # B, 1, H, W


def generateUVGridFlat(w, h, device=None):
    """
    generate flat grid of normalized pixel coords [-1,1] of size h*w
    :return: h*w, 2
    """
    xRange = torch.linspace(-1.0, 1.0, w, device=device)
    yRange = torch.linspace(-1.0, 1.0, h, device=device)
    grid = torch.meshgrid(yRange, xRange)
    gridFlat = torch.stack((grid[1].reshape(-1), grid[0].reshape(-1)), 1) # h*w, 2
    return gridFlat

def generateRayParametersFromInverseCamMatrix(KInvT, uv, normalize=False):
    """
    generate normalized ray
    :param KInvT: transposed inverse camera matrix B, 4, 3
    :param uv: image plane coordinate through which ray should go in [-1, 1] B, N, 2
    :param normalize: normalize rayDirection, default False
    :return: rayOrigin B, N, 3 and rayDirection B, N, 3
    """
    rayOrigin = KInvT[:,3:,:] # corresponds to KInv*[0;0;0;1]
    rayOrigin = rayOrigin.expand(-1, uv.shape[1], -1)
    rayDirection = torch.bmm(uv, KInvT[:,:2,:3]) + KInvT[:,2:3,:3] # corresponds to KInv*[u;v;1;0]
    if normalize:
        rayDirection /= torch.linalg.norm(rayDirection, ord=2, dim=2, keepdim=True)
    return rayOrigin, rayDirection

def generateRaysFromRayParameters(rayOrigins, rayDirections, t):
    """
    :param rayOrigins: N, 3
    :param rayDirections: N, 3
    :param t: N, NP, 1
    :return: N, NP, 3
    """
    return rayOrigins.unsqueeze(1) + t * rayDirections.unsqueeze(1)

def sampleRayTsCoarse(rayBounds, NP):
    """
    computes scalar ray factors uniformly in NP bins between near/far rayBounds
    :param rayBounds: N, 2 near/far bounds
    :param NP: number of points
    :return: N, NP, 1
    """
    step = 1.0/NP
    t = torch.linspace(0, 1 - step, NP, device=rayBounds.device) # NP
    N = rayBounds.shape[0]
    t = t.unsqueeze(0).repeat(N, 1)  # N, NP
    t += torch.rand_like(t) * step
    near = rayBounds[:, 0:1]
    far = rayBounds[:, 1:]
    return (near * (1.0 - t) + far * t).unsqueeze(-1)  # N, NP, 1


def get_x_c_rayPointsFrom_uv(KInvT, KBounds, uv, NP):
    """
    :param KInvT: B, 4, 3
    :param KBounds: B, 2
    :param uv: B, N, 2
    :param NP: number of points along each ray
    :return: t: B*N, NP, 1;    x_w: B*N, NP, 3
    """
    B, N, _ = uv.shape
    rayOrigin, rayDirection = generateRayParametersFromInverseCamMatrix(KInvT, uv, normalize=True)  # rayOrigin B, N, 3 and rayDirection B, N, 3
    rayOrigin = rayOrigin.reshape(B * N, 3)
    rayDirection = rayDirection.reshape(B * N, 3)
    KBounds = KBounds.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 2)
    t = sampleRayTsCoarse(rayBounds=KBounds, NP=NP)  # B*N, NP, 1
    x_w = generateRaysFromRayParameters(rayOrigin, rayDirection, t)  # B*N, NP, 3
    return t, x_w

def getRayPointInsideBoundingBoxIndices(x_w, boundingBoxLimitLow, boundingBoxLimitHigh, M_enlarged = None):
    """
    :param x_w: B*N, NP, 3
    :param boundingBoxLimitLow: # 1, 1, 3
    :param boundingBoxLimitHigh: # 1, 1, 3
    :return: B*N*NP, B*N, (B*N)'*NP
    """
    # mask of ray points that are within the bounding box
    insideInds_BN_NP = torch.all(torch.logical_and(x_w > boundingBoxLimitLow, x_w < boundingBoxLimitHigh), dim=2)  # B*N, NP

    if M_enlarged is not None:
        #M_enlarged # B, N
        B, N = M_enlarged.shape
        M_enlarged = M_enlarged.reshape(B*N)
        insideInds_BN_NP[~M_enlarged] = False

    # mask of rows, i.e. pixels, that contain at least one point within the bounding box
    insideInds_BN = torch.any(insideInds_BN_NP, dim=1)  # B*N

    if not torch.any(insideInds_BN):
        return None, None, None

    # remove pixel rays of ray mask that do not contain a point at all
    insideInds_BNPrime_NP = insideInds_BN_NP[insideInds_BN].view(-1)  # (B*N)'*NP
    return insideInds_BN_NP.view(-1), insideInds_BN, insideInds_BNPrime_NP

def scatterSparseNerfResultIntoRayNPChunk(sigmaTotal, cTotal, insideIndsNPN, NP):
    """
    :param sigmaTotal: (B*N*NP)', nM, 1
    :param cTotal: (B*N*NP)', nM, 3
    :param insideIndsNPN: (B*N)'*NP
    :param NP: number of points along original ray
    :return: Sigma: (B*N)', NP, 1;   C: (B*N)', NP, 3
    """
    Sigma = torch.zeros(insideIndsNPN.shape[0], 1, device=sigmaTotal.device)  # (B*N)'*NP, 1
    C = torch.zeros(insideIndsNPN.shape[0], 3, device=sigmaTotal.device)  # (B*N)'*NP, 3
    Sigma[insideIndsNPN] = sigmaTotal
    C[insideIndsNPN] = cTotal
    sigmaTotal = Sigma.view(-1, NP, 1)  # (B*N)', NP, 1
    cTotal = C.view(-1, NP, 3)  # (B*N)', NP, 3
    return  sigmaTotal, cTotal

def compositeMultipleNerfs(sigma, c):
    """
    :param sigma: N, nM, 1
    :param c: N, nM, 3
    :return: sigma, c of composite B, N, 1   B, N, 3
    """
    sigmaTotal = torch.sum(sigma, dim=1, keepdim=False)  # N, 1
    sigmaTotal[sigmaTotal < 1e-6] = 1e-6  # prevent nan
    cTotal = torch.sum(sigma * c, dim=1, keepdim=False) / sigmaTotal  # N, 3
    return sigmaTotal, cTotal


def calcWeights(t, sigma):
    """
    calculate volume rendering weights
    :param t: N, NP, 1
    :param sigma: N, NP, 1
    :return: weights N, NP, 1
    """
    delta_t = t[:,1:,:] - t[:,:-1,:] # N, NP-1, 1
    #delta_t = torch.cat([delta_t, torch.ones_like(t[:,:1,:])*1e10], dim=1) # N, NP, 1
    #tmp = torch.exp(-sigma*delta_t) # N, NP, 1
    tmp = torch.exp(-sigma[:,:-1,:]*delta_t) # N, NP-1, 1
    tmp = torch.cat([tmp, torch.zeros_like(t[:,:1,:])], dim=1) # N, NP, 1
    w = (1.0-tmp)*torch.cumprod(torch.cat([torch.ones_like(tmp[:,:1,:]), tmp + 1e-10], dim=1), dim=1)[:,:-1,:] #N, NP, 1
    return w