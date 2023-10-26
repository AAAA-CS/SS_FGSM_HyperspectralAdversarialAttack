import os
import numpy as np
import random
import argparse
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io
#from Targeted_Models.models import ResNet18
from Targeted_Models.models import VGG
import time
from PIL import Image   #图片处理
from sklearn.decomposition import PCA #进行PCA降维
# from heatmap import *
#import pyheatmap.heatmap as heatmap
from Targeted_Models.models import inception_v3
from Targeted_Models.models import inception_resnet_v2

def loadData():

    DataPath = '900(1000)_PaviaU01/paviaU.mat'
    TRPath = '900(1000)_PaviaU01/TRLabel.mat'
    TSPath = '900(1000)_PaviaU01/TSLabel.mat'

    # load data
    Data = io.loadmat(DataPath)
    TrLabel = io.loadmat(TRPath)
    TsLabel = io.loadmat(TSPath)
    
    Data = Data['paviaU']
    Data = Data.astype(np.float32)
    TrLabel = TrLabel['TRLabel']
    TsLabel = TsLabel['TSLabel']

    return Data, TrLabel, TsLabel

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createPatches(X, y, windowSize, windowsizemax):

    [m, n, l] = np.shape(X)
    temp = X[:,:,0]
    pad_width = np.floor(windowSize/2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2,n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')
    for i in range(l):
        temp = X[:,:,i]
        pad_width = np.floor(windowSize/2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        #print("temp2:",temp2.shape)
        x2[:, :, i] = temp2


    [ind1, ind2] = np.where(y != 0)
    TrainNum = len(ind1)
    patchesData = np.empty((TrainNum, l, windowsizemax, windowsizemax), dtype='float32')
    patchesLabels = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    pad_width_enlarge = np.floor((windowsizemax-windowSize)/2)
    pad_width_enlarge = np.int_(pad_width_enlarge)
    for i in range(len(ind1)):
        # patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patchh = np.empty((windowsizemax, windowsizemax, l), dtype='float32')
        for j in range(l):
            temp11 = patch[:,:,j]
            temp11 = np.pad(temp11, pad_width_enlarge, 'mean')
            patchh[:,:,j] = temp11
        patchh = np.reshape(patchh, (windowsizemax * windowsizemax, l))
        patchh = np.transpose(patchh)
        patchh = np.reshape(patchh, (l, windowsizemax, windowsizemax))
        patchesData[i, :, :, :] = patchh
        patchlabel = y[ind1[i], ind2[i]]
        patchesLabels[i] = patchlabel
    
    return patchesData, patchesLabels


def Normalize(dataset):

    [m, n, b] = np.shape(dataset)
    #change to [0,1]
    for i in range(b):
        _range = np.max(dataset[:, :, i])-np.min(dataset[:, :, i])
        dataset[:, :, i] = (dataset[:, :, i] - np.min(dataset[:, :, i]))/_range
    return dataset


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}    

def testFull(args, device, model, Data, windowSize):
    # show the whole image
    # The whole data is too big to test in one time; So dividing it into several parts
    part = args.test_batch_size
    [m, n, b] = np.shape(Data)
    x = Data
    temp = x[:,:,0]
    pad_width = np.floor(windowSize/2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2,n2] = temp2.shape
    x2 = np.empty((m2, n2, b), dtype='float32')
    
    for i in range(b):
        temp = x[:,:,i]
        pad_width = np.floor(windowSize/2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:,:,i] = temp2
        
    pred_all = np.empty((m*n, 1), dtype='float32')   
    number = m*n//part
    for i in range(number):
        D = np.empty((part, b, windowSize, windowSize), dtype='float32')
        count = 0
        for j in range(i*part, (i+1)*part):
            row = j//n
            col = j - row*n
            row2 = row + pad_width
            col2 = col + pad_width
            patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch = np.reshape(patch, (windowSize * windowSize, b))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (b, windowSize, windowSize))
            D[count, :, :, :] = patch
            count += 1
    
        temp = torch.from_numpy(D)
        #temp = temp.cuda()
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[i*part:(i+1)*part, 0] = temp3.cpu()
        del temp, temp2, temp3, D
    
    if (i+1)*part < m*n:
        D = np.empty((m*n-(i+1)*part, b, windowSize, windowSize), dtype='float32')
        print(D.shape)
        count = 0
        for j in range((i+1)*part, m*n):
            row = j // n
            col = j - row * n
            row2 = row + pad_width
            col2 = col + pad_width
            patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch = np.reshape(patch, (windowSize * windowSize, b))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (b, windowSize, windowSize))
            D[count, :, :, :] = patch
            count += 1
    
        temp = torch.from_numpy(D)
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[(i + 1) * part:m*n, 0] = temp3.cpu()
        del temp, temp2, temp3, D

    pred_all = np.reshape(pred_all, (m, n)) + 1
    io.savemat(args.save_path, {'PredAll': pred_all})  

 # construct the network
OutChannel = 32
class CNN_1(nn.Module):
    def __init__(self, input_feature,Classes):
        super(CNN_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_feature,
                out_channels = OutChannel,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(OutChannel, OutChannel*2, 3, 1, 1),
            nn.BatchNorm2d(OutChannel*2),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(0.5),

        )


        self.out = nn.Linear(OutChannel*2, Classes)  # fully connected layer, output 16 classes

    def forward(self, x):
  
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self, input_feature, Classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_feature,
                out_channels = OutChannel,
                kernel_size = 3,
                stride = 1,
                padding = 1,
            ),
            nn.BatchNorm2d(OutChannel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(OutChannel, OutChannel*2, 3, 1, 1),
            nn.BatchNorm2d(OutChannel*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(OutChannel*2, OutChannel*4, 3, 1, 1),
            nn.BatchNorm2d(OutChannel*4),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(0.5),

        )

        self.out = nn.Linear(OutChannel*4, Classes)  # fully connected layer, output 16 classes

    def forward(self, x):
  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.out(x)
        return x


def TrainFuc(args, device, model, Data, TR_gt, TS_gt, windowSize, windowsizemax):
        Data_TR, TR_gt_M = createPatches(Data, TR_gt, windowSize,windowsizemax)
        Data_TS, TS_gt_M = createPatches(Data, TS_gt,windowSize,windowsizemax)
        
        # change to the input type of PyTorch
        Data_TR = torch.from_numpy(Data_TR)
        Data_TS = torch.from_numpy(Data_TS)
        
        TrainLabel = torch.from_numpy(TR_gt_M)-1
        TrainLabel = TrainLabel.long()
        TestLabel = torch.from_numpy(TS_gt_M)-1
        TestLabel = TestLabel.long()

        return Data_TR, Data_TS, TrainLabel, TestLabel

    
def main():
    # torch.cuda.empty_cache()
    # Training settings
    # default：不指定参数时的默认值；
    # type：命令行参数应该被转换成的类型；
    # help：参数的帮助信息；
    # metavar在usage说明中的参数名称，对于必选参数默认是参数名称，对于可选参数默认是全大写的参数名称.
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait bfore logging training status')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--save_path', default='900(1000)_PaviaU01/Train/pred_All.mat', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--saveNet_path', default='900(1000)_PaviaU01/Train/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)#确保每次运行文件，生成的随机数都是固定的，使得实验结果一致

    print("CUDA Available: ", torch.cuda.is_available())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    Data, TR_gt, TS_gt = loadData()
    [m, n, b] = np.shape(Data)

    Classes = len(np.unique(TR_gt)) - 1 #除掉0

    Data = Normalize(Data)
    #===============================================================================================
    model = ResNet18()
    model = model.to(device)
    get_parameter_number(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()  # the target label is not one-hotted

    windowSize_max = 32
    print('Training window size:', windowSize_max)
    
    [Data_TR0, Data_TS0, TrainLabel, TestLabel] = TrainFuc(args, device, model, Data,TR_gt, TS_gt, windowSize_max, windowSize_max)

    print('Training data size:', Data_TR0.size(0))
    print('Testing window size:', Data_TS0.size(0))
    
    datasetTr = dataf.TensorDataset(Data_TR0,  TrainLabel) 
    train_loader = dataf.DataLoader(datasetTr, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    for epoch in range(args.epochs):
        #print(epoch)
        for step, (b_x0,b_y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            # move train data to GPU
            b_x0 = b_x0.to(device)
            b_y = b_y.to(device)

            output = model(b_x0)  # cnn output

            loss = criterion(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # scheduler.step()
            if step % 1000 == 0:
                model.eval()

                pred_y = np.empty((len(TestLabel)), dtype='float32')
                number = len(TestLabel) // args.test_batch_size
                for i in range(number):
                    temp0 = Data_TS0[i * args.test_batch_size:(i + 1) * args.test_batch_size, :, :, :]
                    temp0 = temp0.to(device)
                    temp22 = model(temp0)
                    temp33 = torch.max(temp22, 1)[1].squeeze()
                    pred_y[i * args.test_batch_size:(i + 1) * args.test_batch_size] = temp33.cpu()
                    del temp0, temp22, temp33

                if (i + 1) * args.test_batch_size < len(TestLabel):
                    temp0 = Data_TS0[(i + 1) * args.test_batch_size:len(TestLabel), :, :, :]
                    temp0 = temp0.to(device)
                    temp22 = model(temp0)
                    temp33 = torch.max(temp22, 1)[1].squeeze()
                    pred_y[(i + 1) * args.test_batch_size:len(TestLabel)] = temp33.cpu()
                    del temp0, temp22, temp33

                pred_y = torch.from_numpy(pred_y).long()
                accuracy = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)
                print('Epoch: ', epoch, '| train loss: %.5f' % loss.data.cpu().numpy(), '| test accuracy: %.5f' % accuracy)

            torch.save(model.state_dict(), args.saveNet_path + '900(1000)net_resnet.pkl')
            model.train()

    model.load_state_dict(torch.load(args.saveNet_path + '900(1000)net_resnet.pkl'))
    model.eval()
    testFull(args, device, model, Data, windowSize_max)
    

    part = args.test_batch_size
    pred_y = np.empty((len(TestLabel)), dtype='float32')
    number = len(TestLabel)//part
    for i in range(number):
        temp = Data_TS0[i*part:(i+1)*part, :, :]
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i*part:(i+1)*part] = temp3.cpu()
        del temp, temp2, temp3

    if (i+1)*part < len(TestLabel):
        temp = Data_TS0[(i+1)*part:len(TestLabel), :, :]
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i+1)*part:len(TestLabel)] = temp3.cpu()
        del temp, temp2, temp3

    pred_y = torch.from_numpy(pred_y).long()
    OA = torch.sum(pred_y == TestLabel).type(torch.FloatTensor) / TestLabel.size(0)

    Classes = np.unique(TestLabel)
    EachAcc = np.empty(len(Classes))

    for i in range(len(Classes)):
        cla = Classes[i]
        right = 0
        sum = 0

        for j in range(len(TestLabel)):
            if TestLabel[j] == cla:
                sum += 1
            if TestLabel[j] == cla and pred_y[j] == cla:
                right += 1

        EachAcc[i] = right.__float__()/sum.__float__()

    print("OA:",OA)
    print("EachAcc:",EachAcc)


if __name__ == '__main__':
    main()



