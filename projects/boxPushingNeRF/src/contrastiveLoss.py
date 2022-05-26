import torch

def compute_logits(zQuery, zTarget, W):
    """
    :zQuery: B, nM, k
    :zTarget: B, nM, k
    :W: nM, k, k
    inspired by https://github.com/MishaLaskin/curl/blob/master/curl_sac.py#L217
    """
    Wz = torch.bmm(W, zTarget.permute(1, 2, 0))  # nM, k, B
    logits = torch.bmm(zQuery.permute(1, 0, 2), Wz)  # nM, B, B
    return logits

def computeContrastiveLoss(zQuery, zTarget, W):
    """
    :param zQuery: B, nM, k
    :param zTarget: B, nM, k
    :param W: nM, k, k
    :return: B, nM
    """
    logits = compute_logits(zQuery, zTarget, W) # nM, B, B
    logits = logits.permute(1, 2, 0) # B, B, nM
    labels = torch.arange(logits.shape[0], dtype=torch.int64, device=zQuery.device).unsqueeze(1).expand(-1, zQuery.shape[1])
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none') # B, nM
    return loss


def soft_update_params(net, target_net, tau):
    """
    from https://github.com/MishaLaskin/curl/blob/master/utils.py#L28
    MIT License
    Copyright (c) 2020 CURL (Contrastive Unsupervised Representations for Reinforcement Learning) Authors (https://arxiv.org/abs/2004.04136)
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )
        

def randomCropNoBatch(img, HC, WC):
    """
    non batched version of random crop
    :param img: C, H, W
    :param HC: size of cropped image
    :param WC: size of cropped image
    no check is made if HC or WC is larger than H or W
    :return: C, HC, WC
    """
    _, H, W = img.shape
    hInd = torch.randint(0, H - HC + 1, size=(1,)).item()
    wInd = torch.randint(0, W - WC + 1, size=(1,)).item()
    return img[:, hInd:hInd+HC, wInd:wInd + WC]

def randomCrop(img, HC, WC):
    """
    :param img: B, C, H, W
    :param HC: size of cropped image
    :param WC: size of cropped image
    no check is made if HC or WC is larger than H or W
    :return: B, C, HC, WC
    """
    I = []
    for imgB in img:
        I.append(randomCropNoBatch(imgB, HC, WC))

    return torch.stack(I, dim=0)
    
    
def centerCrop(img, HC, WC):
    """
    :param img: B, C, H, W
    :param HC: size of cropped image
    :param WC: size of cropped image
    no check is made if HC or WC is larger than H or W
    :return: B, C, HC, WC
    """
    B, C, H, W = img.shape
    hInd = (H - HC) // 2
    wInd = (W - WC) // 2
    return img[:, :, hInd:hInd+HC, wInd:wInd+WC]
