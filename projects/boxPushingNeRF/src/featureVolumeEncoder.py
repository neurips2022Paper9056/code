import torch

class FeatureVolumeEncoder_1(torch.nn.Module):
    def __init__(self, F, k):
        super(FeatureVolumeEncoder_1, self).__init__()
        self.conv1Layer = torch.nn.Conv3d(in_channels=F,
                                          out_channels=2*F,
                                          kernel_size=[3, 3, 3],
                                          )

        self.conv2Layer = torch.nn.Conv3d(in_channels=2*F,
                                          out_channels=2*F,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          )

        self.conv3Layer = torch.nn.Conv3d(in_channels=2*F,
                                          out_channels=2*F,
                                          kernel_size=[3, 3, 3],
                                          stride=[2, 2, 2],
                                          )
        self.convNetwork = torch.nn.Sequential(
            self.conv1Layer,
            torch.nn.ReLU(),
            self.conv2Layer,
            torch.nn.ReLU(),
            self.conv3Layer,
            torch.nn.ReLU(),
        )
        self.convToDense = torch.nn.Linear(in_features=8192, out_features=300)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, k),
        )

    def forward(self, Y):
        B = Y.shape[0]
        Y = self.convNetwork(Y)
        Y = Y.reshape(B, -1)
        Y = self.convToDense(Y)
        return self.dense(Y)
