#-*-coding:utf-8-*-


from torchvision.models.mobilenet import MobileNetV2
import torch


class MobileNet(MobileNetV2):

    def __init__(self):
        super(MobileNet, self).__init__()
        self.freeze()

    def freeze(self):
        params = self.classifier.parameters()
        for param in params:
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    model = MobileNet()
    x = torch.rand(18, 3, 416, 416)
    print(model(x).shape)
