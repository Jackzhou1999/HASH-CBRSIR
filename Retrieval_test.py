from Models.NET import FAH
import numpy as np
import torch
import torchvision.datasets as dset
from Load_data import Create_data, get_test_loader
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Coder:
    def __init__(self, num_bits, model_pth, data_pth):
        self.device = torch.device('cuda')

        self.test_dataset = dset.ImageFolder(root=data_pth)
        self.num_classes = len(self.test_dataset.class_to_idx)
        self.test_dataset = Create_data(self.test_dataset, augment=False)
        self.test_losder = get_test_loader(self.test_dataset, 10)

        self.Model = FAH(self.device, model_pth, self.num_classes, num_bits, False).to(self.device)
        self.Model.eval()
        self.Model.load_state_dict(torch.load(model_pth)['net'])
        self.threshold = 0.5

    def encode(self):
        real_code = list()
        label = list()
        for i, (x, y) in enumerate(self.test_losder):
            if torch.cuda.is_available():
                x = x.float().cuda()
            _, fi, _ = self.Model(x)
            real_code.append(fi.cpu().detach().numpy())
            label.append(y)

        real_code = np.vstack(real_code)
        label = np.vstack(label)
        hash_code = np.zeros_like(real_code)
        indX, indY = np.where(real_code > self.threshold)
        for i, j in zip(indX, indY):
            hash_code[i][j] = 1

        return hash_code, real_code, label

def MAP(Q, top, hash_code, groundtruth):
    num_type = len(np.unique(groundtruth))
    total_sample = hash_code.shape[0]
    idx = np.random.permutation(range(total_sample))
    hash_code = hash_code[idx]
    groundtruth = groundtruth[idx]

    query_sample = hash_code[:Q, :]
    query_sample_label = groundtruth[:Q, :]
    images = hash_code[Q:, :]
    images_label = groundtruth[Q:, :]

    classes_num = []
    for i in range(num_type):
        classes_num.append(np.sum(images_label == i))
    Map = []
    for i in range(Q):
        image = query_sample[i].reshape(1, -1)
        distance = cdist(image, images, metric='hamming')
        top_label = images_label[np.argsort(distance[0])[:top]]
        AP = 0.
        count = 0.
        for j in range(top):
            if top_label[j] == query_sample_label[i]:
                count += 1
                AP += (count/(j+1))
        AP /= classes_num[int(query_sample_label[i][0])]
        Map.append(AP)
    Map = np.mean(Map)
    # print(Map)
    return Map

def P_R(Q, hash_code, groundtruth):
    num_type = len(np.unique(groundtruth))
    total_sample = hash_code.shape[0]
    idx = np.random.permutation(range(total_sample))
    hash_code = hash_code[idx]
    groundtruth = groundtruth[idx]

    query_sample = hash_code[:Q, :]
    query_sample_label = groundtruth[:Q, :]
    images = hash_code[Q:, :]
    images_label = groundtruth[Q:, :]

    classes_num = []
    for i in range(num_type):
        classes_num.append(np.sum(images_label == i))

    for i in range(Q):
        P_list = []
        R_list = []
        image = query_sample[i].reshape(1, -1)
        distance = cdist(image, images, metric='hamming')
        index = np.argsort(distance[0])
        for j in np.arange(1, 31):
            cut_index = index[:j]
            label = images_label[cut_index]
            TP = np.sum(label == query_sample_label[i][0])
            FP = j -TP
            P = TP/(TP + FP)
            P_list.append(P)

            all_label_i_num = classes_num[int(query_sample_label[i][0])]
            FN = all_label_i_num - TP
            R = TP/(TP+FN)
            R_list.append(R)

        plt.figure()
        plt.grid(True)
        plt.plot(R_list, P_list, 'black')
        plt.show()



if __name__ == "__main__":
    # np.random.seed(1)
    for idx in range(20, 420, 20):
        model_path = "/home/jackzhou/PycharmProjects/CBRSIR_hash/Model_save/Model_epoch{}.pkl".format(idx)
        data_path = '/home/jackzhou/PycharmProjects/CBRSIR_hash/Dataset/test'
        retrieval_model = Coder(128, model_path, data_path)
        hash_code, real_code, label = retrieval_model.encode()
        # P_R(10, hash_code, label)
        save = []
        for i in range(500):
            tmp = MAP(20, 60, hash_code, label)
            save.append(tmp)
        print(idx, np.mean(save), np.max(save), np.min(save))


