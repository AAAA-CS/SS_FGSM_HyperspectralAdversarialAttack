import math
import random
import os

import numpy as np
import torch
import scipy.io as sio

def get_dis(data, cen_band, k):
    [_, _, b] = np.shape(data)
    dis_list = np.zeros((b, k))
    for c in range(k):
        tempdata_c = data[:, :, cen_band[c]].flatten()
        for x in range(b):
            tempdata_x = data[:, :, x].flatten()
            i = np.cov(tempdata_c)
            j = np.cov(tempdata_x)
            dis_list[x, c] = math.fabs(i - j)
    return dis_list


def classify(data, cen_band, k):
    dis_list = get_dis(data, cen_band, k)
    [m, n, b] = data.shape
    minDis = np.argmin(dis_list, axis=1)
    tem_band = np.empty_like(minDis)
    for i in range(len(minDis)):
        sum = 0
        t = 0
        for j in range(len(minDis)):
            if(minDis[i] == minDis[j]):
                sum = sum + j
                t = t + 1
        tem_band[i] = int(sum / t)
    tem_band = np.unique(tem_band)
    if len(tem_band) != k:
        while True:
            tband = random.randint(0, (b - 1))
            if tband not in tem_band:
                tem_band = np.append(tem_band, tband)
            if len(tem_band) == k:
                break
    changed = tem_band - cen_band
    return changed, tem_band


def kmeans(data, k):
    [m, n, b] = data.shape
    cen_band = []
    while True:
        tband = random.randint(0, (b-1))
        if tband not in cen_band:
            cen_band.append(tband)
        if len(cen_band) == k:
            break
    print(cen_band)

    changed, newcen_band = classify(data, cen_band, k)
    print(newcen_band)
    for i in range(10):
        changed, newcen_band = classify(data, newcen_band, k)
        print(newcen_band)

    cluster_data = np.empty_like(data)
    dis_list = get_dis(data, cen_band, k)
    minDis = np.argmin(dis_list, axis=1)
    for nband in range(b):
        for i in range(m):
            for j in range(n):
                cluster_data[i, j, nband] = data[i, j, cen_band[minDis[nband]]]

    cluster_idx = np.zeros(b)
    for i in range(b):
        cluster_idx[i] = cen_band[minDis[i]]
    return cluster_data, cen_band, cluster_idx



def kmeans_noise(noise, cluster_idx):
    [_, ln] = np.shape(cluster_idx)

    # cluster_idx = np.reshape(cluster_idx, (ln))
    # cluster_idx = torch.from_numpy(cluster_idx).to(0)
    # index = torch.unique(cluster_idx).to(0)
    # for i in range(len(index)):
    #    [indxx] = torch.where(cluster_idx == index[i])
    #    date_sum = torch.sum(noise[:, indxx, :, :], 0) / indxx.size(0)
    #    noise[:, indxx, :, :] = date_sum

    cluster_idx = np.reshape(cluster_idx, (ln, 1))
    for i in range(1, len(cluster_idx)):
        if cluster_idx[i] == cluster_idx[i - 1]:
            noise[:, i, :, :] = noise[:, (i - 1), :, :]
    return noise

def main():
    DataPath = '900(1000)_PaviaU01/paviaU.mat'

    # load data
    Data = sio.loadmat(DataPath)
    Data = Data['paviaU']

    k = 20

    Data_cluster, _, cluster_idx = kmeans(Data, k)

    save_Path = '900(1000)_PaviaU01/kmeans/'
    if not os.path.exists(save_Path):
        os.makedirs(save_Path)
    save_idx = os.path.join(save_Path, 'km' + str(int(k)) + '_idx' + '.mat')
    sio.savemat(save_idx, {'km_idx': cluster_idx})


if __name__ == '__main__':
    main()