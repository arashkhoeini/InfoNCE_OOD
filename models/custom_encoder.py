import torch.nn as nn
import torch

class CustomEncoder(nn.Module):

    def __init__(self, base_encoder, dim=128):
        """
        dim: feature dimension (default: 128)
        T: softmax temperature (default: 0.07)
        """
        super(CustomEncoder, self).__init__()

        self.encoder = base_encoder
        # dim_mlp = self.encoder.classifier[1].weight.shape[1]
        # print(dim_mlp)
        # self.encoder.fc = nn.Sequential(
        #     nn.Linear(dim_mlp, dim)
        # )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, dim)
        )
        

    def forward(self, X):
        """
        Input:
            X: a batch of query images
        Output:
            representations: the representations of the input images
        """
        representations = self.fc(self.encoder(X).pooler_output.squeeze((-1,-2)))
        return representations