import numpy as np
import argparse
import torch, gc
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io

import resnet
from torchattacks.attacks.ssfgsm import SS_FGSM


def loadData():
    DataPath = '900(1000)_PaviaU01/paviaU.mat'
    TRPath = '900(1000)_PaviaU01/TRLabel.mat'
    TSPath = '900(1000)_PaviaU01/TSLabel.mat'
    cluster_idxPath = '900(1000)_PaviaU01/kmeans/km20_idx.mat'
    DatasupPath = '900(1000)_PaviaU01/slic/slic_4.mat'

    # load data
    Data = io.loadmat(DataPath)
    TrLabel = io.loadmat(TRPath)
    TsLabel = io.loadmat(TSPath)
    cluster_idx = io.loadmat(cluster_idxPath)
    Data_sup = io.loadmat(DatasupPath)

    Data = Data['paviaU']
    Data = Data.astype(np.float32)
    TrLabel = TrLabel['TRLabel']
    TsLabel = TsLabel['TSLabel']
    cluster_idx = cluster_idx['km_idx']
    Data_sup = Data_sup['slic']

    return Data, TrLabel, TsLabel, cluster_idx, Data_sup


def createPatches(X, y, windowSize):
    [m, n, l] = np.shape(X)
    temp = X[:, :, 0]
    pad_width = np.floor(windowSize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = X[:, :, i]
        pad_width = np.floor(windowSize / 2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    [ind1, ind2] = np.where(y != 0)
    TrainNum = len(ind1)
    patchesData = np.empty((TrainNum, l, windowSize, windowSize), dtype='float32')
    patchesLabels = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch = np.reshape(patch, (windowSize * windowSize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, windowSize, windowSize))
        patchesData[i, :, :, :] = patch
        patchlabel = y[ind1[i], ind2[i]]
        patchesLabels[i] = patchlabel

    return patchesData, patchesLabels


def Normalize(dataset):
    [m, n, b] = np.shape(dataset)
    # change to [0,1]
    for i in range(b):
        _range = np.max(dataset[:, :, i]) - np.min(dataset[:, :, i])
        dataset[:, :, i] = (dataset[:, :, i] - np.min(dataset[:, :, i])) / _range

    return dataset


def TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize):
    Data_TR, TR_gt_M = createPatches(Data, TR_gt, windowSize)
    Data_TS, TS_gt_M = createPatches(Data, TS_gt, windowSize)

    # change to the input type of PyTorch
    Data_TR = torch.from_numpy(Data_TR)
    Data_TS = torch.from_numpy(Data_TS)

    TrainLabel = torch.from_numpy(TR_gt_M) - 1
    TrainLabel = TrainLabel.long()
    TestLabel = torch.from_numpy(TS_gt_M) - 1
    TestLabel = TestLabel.long()

    return Data_TR, Data_TS, TrainLabel, TestLabel


def str2bool(v):
    return v.lower() in ('true', '1')


def main():
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='# of images in each batch of data')
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Whether to train or test the model')
    parser.add_argument('--epochs', type=int, default=300,
                        help='# of epochs to train for')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Initial learning rate value')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='value of weight dacay for regularization')
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help="Whether to run on the GPU")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_path', default='./checkpoint/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epsilon', type=float, default=0.03, metavar='LR',
                        help='adversarial rate (default: 0.1)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print("CUDA Available: ", torch.cuda.is_available())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    Data, TR_gt, TS_gt, cluster_idx, Data_sup = loadData()
    [m, n, b] = np.shape(Data)
    Classes = len(np.unique(TR_gt)) - 1
    print(Classes)

    Data = Normalize(Data)
    # ==================================================================================================================
    windowSize = 32

    sqr = np.sqrt(windowSize * windowSize * b)

    pretrained_model = "900(1000)_PaviaU01/Train/900(1000)net_resnet18.pkl"
    pre_model = resnet.ResNet18()
    pre_model = pre_model.to(device)
    pre_model.load_state_dict(torch.load(pretrained_model))
    pre_model.eval()

    print('Training window size:', windowSize)

    [Data_tr, Data_ts, TrainLabel, TestLabel] = TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize)

    # ==================================================================================================================
    # Adversarial training setup
    # adversary = FGSMAttack(epsilon=0.03)
    for epoch in range(1):
        # testFull(args, device, targeted_model, Data, windowSize)
        part = args.test_batch_size
        number = len(TestLabel) // part

        pred_y_adv = np.empty((len(TestLabel)), dtype='float32')

        # Test_advSample
        Datats_sup, _= createPatches(Data_sup, TS_gt, windowSize)
        Datats_sup = torch.Tensor(Datats_sup)
        num_correct_adv = 0
        for i in range(number):
            tempdata = Data_ts[i * part:(i + 1) * part, :, :].to(device)
            tempsup = Datats_sup[i * part:(i + 1) * part, :, :].to(device)
            TestLabel_1 = TestLabel[i * part:(i + 1) * part].to(device)

            SS_FGSM_attack = SS_FGSM(pre_model, eps=args.epsilon)
            x_adv = SS_FGSM_attack(cluster_idx, tempsup, tempdata, TestLabel_1)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv += torch.sum(out_C == TestLabel_1, 0)

            temp_C = pre_model(x_adv)
            temp_CC = torch.max(temp_C, 1)[1].squeeze()
            pred_y_adv[i * part:(i + 1) * part] = temp_CC.cpu()

            del tempdata, TestLabel_1, x_adv, tempsup, out_C, temp_C, temp_CC

        if (i + 1) * part < len(TestLabel):
            tempdata = Data_ts[(i + 1) * part:len(TestLabel), :, :].to(device)
            tempsup = Datats_sup[(i + 1) * part:len(TestLabel), :, :].to(device)
            TestLabel_1 = TestLabel[(i + 1) * part:len(TestLabel)].to(device)

            SS_FGSM_attack = SS_FGSM(pre_model, eps=args.epsilon)
            x_adv = SS_FGSM_attack(cluster_idx, tempsup, tempdata, TestLabel_1)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv += torch.sum(out_C == TestLabel_1, 0)

            temp_C = pre_model(x_adv)
            temp_CC = torch.max(temp_C, 1)[1].squeeze()
            pred_y_adv[(i + 1) * part:len(TestLabel)] = temp_CC.cpu()

            del tempdata, TestLabel_1, x_adv, tempsup, out_C, temp_C, temp_CC

        print('num_correct_adv: ', num_correct_adv)
        print('accuracy of adv test set: %f\n' % (num_correct_adv.item() / len(TestLabel)))


        # Test_adv
        Classes = np.unique(TestLabel)
        EachAcc_adv = np.empty(len(Classes))

        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0

            for j in range(len(TestLabel)):
                if TestLabel[j] == cla:
                    sum += 1
                if TestLabel[j] == cla and pred_y_adv[j] == cla:
                    right += 1

            EachAcc_adv[i] = right.__float__() / sum.__float__()
        print(EachAcc_adv)

if __name__=='__main__':
    main()
