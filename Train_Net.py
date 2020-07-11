from Models.NET import FAH, Simple_discriminator
from Load_data import get_train_valid_loader, get_test_loader, Create_data

import torch
from torch import nn, optim
import numpy as np
import torchvision.datasets as dset
from torch.autograd import Variable

class L_loss(nn.Module):
    def __init__(self, batch_size, m, lamta, rou, epsiluo, num_bits, CELoss, BCELoss):
        super(L_loss, self).__init__()
        self.batch_size = batch_size
        self.m = m
        self.lamta = lamta
        self.rou = rou
        self.epsiluo = epsiluo
        self.num_bits = num_bits
        self.CELoss = CELoss
        self.BCEloss = BCELoss

    def forward(self, Ym, Fi, Yi, y):
        l_pair = torch.zeros(1).cuda()
        l_pair.requires_grad_(True)

        target = Fi.detach()
        l_sem = self.lamta * self.CELoss(Yi, y)
        l_att = self.rou * self.CELoss(Ym, y)
        self.batch_size = Ym.shape[0]

        for i in range(0, self.batch_size):
            for j in range(i + 1, self.batch_size):
                if y[i] == y[j]:
                    l_pair = torch.norm(Fi[i] - Fi[j]) ** 2 + l_pair
                else:
                    if self.m - torch.norm(Fi[i] - Fi[j]) ** 2 > 0:
                        l_pair = (self.m - torch.norm(Fi[i] - Fi[j]) ** 2) + l_pair
        l_pair = l_pair/(2. * self.batch_size * (self.batch_size - 1))

        l_qua = self.BCEloss(Fi, target) * self.epsiluo
        print(l_pair.item(), l_sem.item(), l_att.item(), l_qua.item())
        return l_pair + l_sem + l_att + l_qua

class LG_loss(nn.Module):
    def __init__(self, batchsize, BCELoss):
        super(LG_loss, self).__init__()
        self.criteon = BCELoss

    def forward(self, logits):
        self.label = torch.ones(size=(logits.shape[0], 1)).float().cuda()
        return self.criteon(logits, self.label)

class LD_loss(nn.Module):
    def __init__(self, batchsize, BCELoss):
        super(LD_loss, self).__init__()
        self.batch_size = batchsize

        self.criteon = BCELoss

    def forward(self, DGx, Dz):
        self.real_labels = torch.ones(Dz.shape[0]).float().cuda()
        self.fake_labels = torch.zeros(DGx.shape[0]).float().cuda()
        return self.criteon(DGx, self.fake_labels) + self.criteon(Dz, self.real_labels)


def Create_Z(batchsize, num_bits):
    tmp1 = np.ones(shape=(batchsize, int(num_bits/2)))
    tmp2 = np.zeros(shape=(batchsize, int(num_bits/2)))
    Z = np.hstack((tmp1, tmp2))
    for i in range(batchsize):
        np.random.shuffle(Z[i])
    Z = torch.from_numpy(Z).float().cuda()
    return Z


def train():
    lr = 0.0001

    m = 1
    lamta = 1.5
    rou = 1.5
    epsiluo = 0.01


    batch_size = 128
    ratio = 0.8
    num_workers = 4
    pin_memory = True
    train_dataset = dset.ImageFolder(root='/home/jackzhou/PycharmProjects/CBRSIR_hash/Dataset/train')
    num_classes = len(train_dataset.class_to_idx)
    num_bits = 128
    full_dataset = Create_data(train_dataset, augment=False)
    train_loader, valid_loader = get_train_valid_loader(full_dataset, batch_size, ratio, num_workers, pin_memory)
    test_dataset = dset.ImageFolder(root='/home/jackzhou/PycharmProjects/CBRSIR_hash/Dataset/test')
    test_dataset = Create_data(test_dataset, augment=False)
    test_losder = get_test_loader(test_dataset, batch_size)

    device = torch.device('cuda')
    if False:
        pth = "/home/jackzhou/PycharmProjects/CBRSIR_hash/Models/alexnet.pth"
        Model = FAH(device, pth, num_classes, num_bits)
        Discriminator = Simple_discriminator(num_bits).to(device)
    else:
        model_path = "/home/jackzhou/PycharmProjects/CBRSIR_hash/Model_save/Model_epoch400.pkl"
        Model = FAH(device, model_path, num_classes, num_bits, False).to(device)
        Model.load_state_dict(torch.load(model_path)['net'])
        Discriminator = Simple_discriminator(num_bits).to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    criteon2 = nn.BCELoss().to(device)

    L = L_loss(batch_size, m, lamta, rou, epsiluo, num_bits, criteon, criteon2)
    LG = LG_loss(batch_size, criteon2)
    LD = LD_loss(batch_size, criteon2)

    print("===========================================================================")
    if True:
        Init_Epoch = 1
        Freeze_Epoch = 3
        Model.train()
        optimizer = optim.Adam(Discriminator.parameters(), betas=(0.5, 0.999), lr=0.001)

        for param in Model.backbone.parameters():
            param.requires_grad = False
        for param in Discriminator.parameters():
            param.requires_grad = True

        for epoch in range(Init_Epoch, Freeze_Epoch):
            for i, (x, y) in enumerate(train_loader):
                if torch.cuda.is_available():
                    x, y = x.float().cuda(), y.long().cuda()

                optimizer.zero_grad()
                Ym, Fi, Yi = Model(x)
                Fi.detach()

                DGx = Discriminator(Fi)
                Z = Create_Z(batch_size, num_bits)
                Dz = Discriminator(Z)
                loss = LD(DGx, Dz)
                loss.backward()
                optimizer.step()
            print(loss.item())

    if True:
        Init_Epoch = 3
        Freeze_Epoch = 401
        Model.train()
        optimizer = optim.Adam(Model.parameters(), lr, weight_decay=0.0005)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        ld_save = list()
        lg_save = list()
        l_save = list()

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in Model.backbone.parameters():
            param.requires_grad = True
        for param in Discriminator.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            print("EPOCH: ", epoch)
            for i, (x, y) in enumerate(train_loader):
                if torch.cuda.is_available():
                    x, y = x.float().cuda(), y.long().cuda()
                y = y.reshape(y.shape[0])
                if i % 5 == 0 and i != 0:
                    optimizer.zero_grad()
                    Ym, Fi, Yi = Model(x)

                    DGx = Discriminator(Fi)
                    loss = LG(DGx)
                    print("LG loss:", loss.item())
                    lg_save.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    Z = Create_Z(batch_size, num_bits)
                    Dz = Discriminator(Z)
                    ld_loss = LD(DGx, Dz)
                    ld_save.append(ld_loss)
                else:
                    optimizer.zero_grad()
                    Ym, Fi, Yi = Model(x)
                    loss = L(Ym, Fi, Yi, y)
                    l_save.append(loss.item())
                    print("L  loss:", loss.item())

                    loss.backward()
                    optimizer.step()

            # lr_scheduler.step()
            if epoch % 20 == 0:
                state = {'net': Model.state_dict()}
                torch.save(state, r'/home/jackzhou/PycharmProjects/CBRSIR_hash/Model_save/Model_epoch{}.pkl'.format(epoch))

    return l_save, lg_save, ld_save



if __name__ == "__main__":
    # np.random.seed(1)
    torch.manual_seed(1)
    l_save, lg_save, ld_save = train()
    # Create_Z(10, 10)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(range(len(l_save)), l_save, 'black', label='L_loss')
    plt.subplot(212)
    plt.plot(range(len(lg_save)), lg_save, 'blue', label='LG_loss')
    plt.plot(range(len(ld_save)), ld_save, 'red', label='LD_loss')
    plt.legend()
    plt.show()
