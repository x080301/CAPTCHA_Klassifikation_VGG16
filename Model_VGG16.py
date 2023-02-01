import torch
import torch.nn as nn
from torchvision import models


class VGGnet(nn.Module):
    def __init__(self, fine_tuning=True, num_classes=12):
        super(VGGnet, self).__init__()

        # import VGG16 model
        model = models.vgg19_bn(pretrained=True)

        # import feature extracter
        self.features = model.features

        # fix parameters in feature extracter
        set_parameter_requires_no_grad(self.features, fine_tuning)

        # import avgpool
        self.avgpool = model.avgpool

        # refactor classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),  # nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, num_classes)  # nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #x = x.view(x.size(0), 512 * 7 * 7)
        out = self.classifier(x)
        return out


def set_parameter_requires_no_grad(model, fine_tuning):
    if fine_tuning:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    net = models.vgg16()
    print("VGG16")
    print(net)

    net = VGGnet()
    print("refactored VGG16")
    print(net)
