import torch.nn as nn
import torchvision.models as models


def infer_arch_from_state_dict(state_dict):
    """Infer the ResNet variant ('resnet18' or 'resnet50') from checkpoint keys.

    Bottleneck blocks (resnet50) carry a third 1x1 ``conv3``; BasicBlock blocks
    (resnet18) only have ``conv1``/``conv2``.  Only key *suffixes* are inspected,
    so this works for a plain ``LinearEvaluation`` state_dict
    (``model.backbone.*``) and for a ``FinetuneModel``-wrapped one
    (``model.model.backbone.*``).
    """
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    keys = tuple(state_dict.keys())
    if any(k.endswith('layer1.0.conv3.weight') for k in keys):
        return 'resnet50'
    if any(k.endswith('layer1.0.conv2.weight') for k in keys):
        return 'resnet18'
    raise ValueError(
        'Cannot infer backbone architecture from checkpoint keys; expected '
        'layer1.0.conv2.weight (resnet18) or layer1.0.conv3.weight (resnet50)'
    )


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

    def forward(self, x):
        embedding = self.model(x)
        # CrossEntropyLoss expects raw, unbounded logits.  Applying Sigmoid
        # here compressed the two class scores before the loss and weakened
        # gradients.  Convert logits to probabilities only during inference.
        logits = self.linear(embedding)
        return embedding, logits
    
class FinetuneModel(nn.Module):
    def __init__(self, model, classes=2, unfreeze="linear"):
        super().__init__()
        self.model = model
        valid = {"linear", "layer4", "layer3_layer4", "all"}
        if unfreeze not in valid:
            raise ValueError(
                f"Unknown unfreeze mode {unfreeze!r}; choose from {sorted(valid)}"
            )

        for name, param in self.model.named_parameters():
            trainable = "linear" in name
            if unfreeze in {"layer4", "layer3_layer4", "all"}:
                trainable = trainable or "backbone.layer4" in name
            if unfreeze in {"layer3_layer4", "all"}:
                trainable = trainable or "backbone.layer3" in name
            if unfreeze == "all":
                trainable = True
            param.requires_grad = trainable
        self.unfreeze = unfreeze

    def forward(self, x):
        embeddings, pre = self.model(x)
        return embeddings, pre



