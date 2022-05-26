import torch



class NeRF_2(torch.nn.Module):
    def __init__(self, **kwargs):
        super(NeRF_2, self).__init__()
        self.xLayer = torch.nn.Linear(3, 64)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(128, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 4)
        )

        if kwargs['sigmaActivation'] == 'softplus':
            self.sigmaActivation = torch.nn.Softplus()
        elif kwargs['sigmaActivation'] == 'ReLU':
            self.sigmaActivation = torch.nn.ReLU()

    def forward(self, x, z):
        """
        :param x: N, 3
        :param z: N, nM, k
        :return: N, nM, 1;   N, nM, 3
        """
        N = x.shape[0]
        _, nM, k = z.shape
        xEncoding = torch.nn.functional.relu(self.xLayer(x)) # N, 64
        xEncoding = xEncoding.unsqueeze(1).expand(-1, nM, -1).reshape(N*nM, -1) # N*nM, 64
        z = z.reshape(N*nM, k) # N*nM, k
        out = self.network(torch.cat([xEncoding, z], dim=1)) # N*nM, 4
        sigma = self.sigmaActivation(out[:,0:1]).view(N, nM, 1)
        c = torch.sigmoid(out[:,1:]).view(N, nM, 3)
        return sigma, c




class NeRF_k(torch.nn.Module):
    def __init__(self, **kwargs):
        super(NeRF_k, self).__init__()
        self.xLayer = torch.nn.Linear(3, 64)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(64+kwargs['k'], 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 4)
        )

        if kwargs['sigmaActivation'] == 'softplus':
            self.sigmaActivation = torch.nn.Softplus()
        elif kwargs['sigmaActivation'] == 'ReLU':
            self.sigmaActivation = torch.nn.ReLU()

    def forward(self, x, z):
        """
        :param x: N, 3
        :param z: N, nM, k
        :return: N, nM, 1;   N, nM, 3
        """
        N = x.shape[0]
        _, nM, k = z.shape
        xEncoding = torch.nn.functional.relu(self.xLayer(x)) # N, 64
        xEncoding = xEncoding.unsqueeze(1).expand(-1, nM, -1).reshape(N*nM, -1) # N*nM, 64
        z = z.reshape(N*nM, k) # N*nM, k
        out = self.network(torch.cat([xEncoding, z], dim=1)) # N*nM, 4
        sigma = self.sigmaActivation(out[:,0:1]).view(N, nM, 1)
        c = torch.sigmoid(out[:,1:]).view(N, nM, 3)
        return sigma, c



