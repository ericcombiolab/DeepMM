import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, in_channels=6):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(num_classes=out_dim),
            "resnet50": models.resnet50(num_classes=out_dim)
        }

        self.backbone = self._get_basemodel(base_model)
        
        # Modify the first conv layer to accept different number of input channels
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

class LinearEvaluation(nn.Module):
    def __init__(self, model, classes = 2):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(128, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedding = self.model(x)
        pred = self.sigmoid(self.linear(embedding))
        return embedding, pred
    
class FinetuneModel(nn.Module):
    def __init__(self, model, classes = 2):
        super().__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            if "linear" not in name:
                param.requires_grad = False

    def forward(self, x):
        embeddings, pre = self.model(x)
        return embeddings, pre



