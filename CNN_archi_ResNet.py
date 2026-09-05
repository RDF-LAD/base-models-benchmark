import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        # On fixe le même nombre de canaux en entrée et en sortie pour éviter les projections complexes
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # La skip connection directe (pas de shortcut nécessaire car les dimensions sont identiques)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Addition directe
        out = F.relu(out)
        return out

class BasicBlockModified(nn.Module):
    def __init__(self, channels):
        super(BasicBlockModified, self).__init__()
        # On fixe le même nombre de canaux en entrée et en sortie pour éviter les projections complexes
        self.conv1 = nn.Conv2d(channels, 2*channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(2*channels)
        self.conv2 = nn.Conv2d(2*channels, 2*channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2*channels)
        self.shortcut = nn.Conv2d(channels, 2*channels, kernel_size=1,stride=2,bias=False)
        self.shortcut_bn = nn.BatchNorm2d(2*channels)
    def forward(self, x):
        identity = self.shortcut_bn(self.shortcut(x))  # La skip connection directe (pas de shortcut nécessaire car les dimensions sont identiques)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Addition directe
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=120):
        super(ResNet, self).__init__()
        
        # Convolution initiale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=1,stride=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Set of blocks 1
        self.layer1 = BasicBlock(64)
        self.layer2 = BasicBlock(64)
        self.layer3 = BasicBlock(64)

        # link blocks 1
        self.link1 = BasicBlockModified(64)

        # Set of blocks 2
        self.layer4 = BasicBlock(128)
        self.layer5 = BasicBlock(128)
        self.layer6 = BasicBlock(128)
        self.layer6bis = BasicBlock(128)

        # link blocks 2
        self.link2 = BasicBlockModified(128)

        # Set of blocks 3
        self.layer7 = BasicBlock(256)
        self.layer8 = BasicBlock(256)
        self.layer9 = BasicBlock(256)
        self.layer9bis = BasicBlock(256)
        self.layer9ter = BasicBlock(256)
        self.layer9quater = BasicBlock(256)

        # link blocks 3
        self.link3 = BasicBlockModified(256)

        # Set of blocks 4
        self.layer10 = BasicBlock(512)
        self.layer11 = BasicBlock(512)
        self.layer12 = BasicBlock(512)

        # Layer final
        self.pool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.link1(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x=self.layer6bis(x)

        x = self.link2(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer9bis(x)
        x = self.layer9ter(x)
        x = self.layer9quater(x)

        x = self.link3(x)

        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# example_model = SimpleResNet(num_classes=10)
# summary(example_model,input_size=(1, 3, 224, 224))