from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import torchvision

class AlexNet_Plus(nn.Module):
    def __init__(self, num_classes=1000, bits=128):
        super(AlexNet_Plus, self).__init__()
        self.num_classes = num_classes
        self.bits = bits

        self.features_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.features_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.features_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv_trans = nn.Conv2d(384, 256, kernel_size=1)

        self.features_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv_trans1 = nn.Conv2d(256, 256, kernel_size=1)

        self.features_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.atten_conv1 = nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1)
        self.atten_batchnorm1 = nn.BatchNorm2d(self.num_classes)

        self.atten_conv2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1)
        self.atten_batchnorm2 = nn.BatchNorm2d(self.num_classes)

        self.atten_conv4 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.atten_conv3 = nn.Conv2d(self.num_classes, 1, kernel_size=1)
        self.atten_batchnorm3 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier_1 = nn.Linear(256 * 6 * 6, 4096)
        self.classifier_4 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, self.bits)

        # self.W = torch.tensor(np.random.normal(0, 0.01, (self.bits, self.num_classes)), dtype=torch.float).cuda()
        # self.W.requires_grad_(True)
        # self.b = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        # self.b.requires_grad_(True)

        self.softmax_layer = nn.Linear(self.bits, self.num_classes, bias=True)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        x = self.maxpool1(F.relu(self.features_0(x)))
        x = self.maxpool2(F.relu(self.features_3(x)))
        x = F.relu(self.features_6(x))
        level3 = self.conv_trans(x)
        x = F.relu(self.features_8(x))
        level4 = self.conv_trans1(x)
        x = F.relu(self.features_10(x))
        
        Fa = F.relu(x + level3 + level4)

        atten_x = self.atten_conv1(Fa)
        atten_x = self.atten_batchnorm1(atten_x)

        atten_x = self.atten_conv2(atten_x)
        atten_x = self.atten_batchnorm2(atten_x)
        Fc = F.relu(atten_x)
        Ym = self.gap(self.atten_conv4(Fc))
        Ym = torch.flatten(Ym, 1)

        atten_x = self.atten_conv3(Fc)
        atten_x = self.atten_batchnorm3(atten_x)
        M = self.sigmoid(atten_x)
        M = torch.repeat_interleave(M, repeats=256, dim=1)
        
        Fm = torch.mul(Fa, M) + Fa
        x = self.maxpool4(Fm)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_4(x)
        Fi = self.sigmoid(self.linear3(x))

        Ysem = self.softmax_layer(Fi)
        # Ysem = torch.mm(Fi, self.W) + self.b

        return Ym, Fi, Ysem

class Simple_discriminator(nn.Module):
    def __init__(self, bits=128):
        super(Simple_discriminator, self).__init__()
        self.bits = bits
        self.fc1 = nn.Linear(self.bits, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 1000)
        self.relu3 = nn.ReLU(inplace=True)
        self.output = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.output(x)
        output = self.sigmoid(x)

        return output

class FAH(nn.Module):
    def __init__(self, device, pth,  num_classes=1000, bits=128, init = True):
        super(FAH, self).__init__()
        self.num_classes = num_classes
        self.bits = bits
        self.pth = pth
        self.backbone = AlexNet_Plus(self.num_classes, self.bits).to(device)
        if init:
            self.init_weight()

    def init_weight(self):
        print("INIT BACKBONE WEIGHT")
        model_dict = self.backbone.state_dict()
        pretrained_dict = torch.load(self.pth)
        state_dict = {}
        for key, value in model_dict.items():
            key_map1 = key[:8] + '.' + key[9:]
            key_map2 = key[:10] + '.' + key[11:]

            if key_map1 in pretrained_dict and value.size() == pretrained_dict[key_map1].size():
                state_dict[key] = pretrained_dict[key_map1]
                print("loading weights {}, size {}".format(key, pretrained_dict[key_map1].size()))
            elif key_map2 in pretrained_dict and value.size() == pretrained_dict[key_map2].size():
                state_dict[key] = pretrained_dict[key_map2]
                print("loading weights {}, size {}".format(key, pretrained_dict[key_map2].size()))
            else:
                state_dict[key] = model_dict[key]
        self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        Ym, Fi, Y_sem = self.backbone(x)
        # print(Ym.shape, Fi.shape, Y_sem.shape)
        return Ym, Fi, Y_sem


if __name__ == "__main__":
    x = torch.rand(size=(512, 3, 224, 224)).cuda()


    device = torch.device('cuda')
    pth = "/home/jackzhou/PycharmProjects/CBRSIR_hash/Models/alexnet.pth"
    model = FAH(device, pth, 21, 128)
    Ym, Fi, Ysem= model(x)
    print(Ym.shape, Fi.shape, Ysem.shape)
    # model = torchvision.models.alexnet(False)
    # print("INIT BACKBONE W")
    # model_dict = model.state_dict()
    # for key, value in model_dict.items():
    #     print(key, value.shape)
