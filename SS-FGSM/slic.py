import math
import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
import os
import torch, gc


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b


    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):

    def make_cluster(self, h, w):
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, data, K, M):
        self.K = K
        self.M = M
        self.data = data

        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(int(h), int(w)))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_N_times(self):
        self.init_clusters()
        self.move_clusters()
        for i in range(1):
            self.assignment()
            self.update_cluster()
            # name = 'lenna_M{m}_K{k}_loop{loop}.png'.format(loop=i, m=self.M, k=self.K)
            # self.save_current_image(name)

    
def get_slic(K, M, data):
    [m, n, b] = np.shape(data)
    # ===PCA====
    n_components = 3
    x = np.reshape(data, (m * n, b))
    pca = PCA(n_components, copy=True, whiten=False)
    x = pca.fit_transform(x)
    _, b = x.shape
    data = np.reshape(x, (m, n, b))

    p = SLICProcessor(data, K, M)
    p.iterate_N_times()
    c_map = np.zeros((m, n, 1))
    for idx, cluster in enumerate(p.clusters):
        for pix in cluster.pixels:
            c_map[pix[0]][pix[1]][0] = idx
    c_map = c_map.astype(int)
    c_map_idx = np.reshape(c_map, (m * n))
    c_map_idx = np.unique(c_map_idx)
    return c_map, c_map_idx


def sup_noise(noise, tempsup):
    with torch.no_grad():
        tempsup_idx = tempsup.reshape(tempsup.size(0)*tempsup.size(1)*tempsup.size(2)*tempsup.size(3))
        tempsup_idx = torch.unique(tempsup_idx)
        temp_noise = noise
        for i in range(tempsup_idx.size(0)):
            [idx1, _, idx3, idx4] = torch.where(tempsup == tempsup_idx[i])
            date_sum = torch.sum(noise[idx1, :, idx3, idx4], 0) / idx1.size(0)
            temp_noise[idx1, :, idx3, idx4] = date_sum
    return temp_noise


def main():
    DataPath = '900(1000)_PaviaU01/paviaU.mat'
    # load data
    Data = sio.loadmat(DataPath)
    Data = Data['paviaU']

    K = 52000
    M = 4

    c_map, c_map_idx = get_slic(K, M, Data)
    print(c_map_idx.shape)

    save_Path = '900(1000)_PaviaU01/slic/'
    if not os.path.exists(save_Path):
        os.makedirs(save_Path)
    save_idx = os.path.join(save_Path, 'slic_' + str(int(M)) + '.mat')
    sio.savemat(save_idx, {'slic': c_map_idx})


if __name__ == '__main__':
    main()
