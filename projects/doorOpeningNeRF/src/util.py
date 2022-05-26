import os

def makeDirs(path):
    root, ext = os.path.splitext(path)
    if not ext:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir) and dir != '':
        os.makedirs(dir)
        
        
        
def perturbMasks(M, nH, sH):
    """
    :param M: B, V, nM, H, W
    :param nH: number of box perturbations
    :param sH: size of box perturbation in each direction
    :return: B, V, nM, H, W
    """
    for maskBatch in M:
        for maskView in maskBatch:
            for maskLatent in maskView:
                a = torch.nonzero(maskLatent == 1)
                inds = np.random.randint(0, a.shape[0], size=nH)
                a = a[inds]
                for b in a:
                    maskLatent[b[0] - sH:b[0] + sH, b[1] - sH:b[1] + sH] = 0.0
