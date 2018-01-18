import numpy as np
import os, random, inspect

import source.utility as util

from tensorflow.contrib.learn.python.learn.datasets import base

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

class DataSet(object):

    def __init__(self, who_am_i, class_len, data_len, height, width, channel):

        self._who_am_i = who_am_i
        self._class_len = class_len
        self._data_len = data_len
        self._height = height
        self._width = width
        self._channel = channel
        self._valid_idx = 0
        self._amount = len(util.get_filelist(directory=PACK_PATH+"/dataset/"+str(self._who_am_i), extensions=["npy"]))

    @property
    def amount(self):
        return self._amount

    @property
    def class_num(self):
        return self._class_len

    @property
    def data_size(self):
        return self._data_len, self._height, self._width, self._channel

    def next_batch(self, batch_size=10, validation=False):

        find_path = PACK_PATH+"/dataset/"+str(self._who_am_i)
        dirlist = util.get_dirlist(path=find_path, dataset_dir="dataset")

        batch_label = []
        batch_data = []

        tmp_label = 0
        for di in dirlist:
            fi_list = util.get_filelist(directory=find_path+"/"+di, extensions=["npy"])

            for fi in fi_list:
                batch_label.append(tmp_label)

                tmp_data = np.load(file=fi)
                batch_data.append(tmp_data)

            tmp_label += 1

        if(batch_size == self._amount):
            indices = range(self._amount)
        elif(validation):
            indices = np.asarray(int(self._valid_idx)).reshape((1, 1))
            self._valid_idx += 1
            if(self._valid_idx >= self._amount):
                self._valid_idx += 0
        else:
            indices = np.random.rand(batch_size) * self._amount

        data = np.zeros((0, self._data_len), float)
        label = np.zeros((0, self._class_len), int)

        for idx in indices:
            idx = int(idx)

            tmp_label = batch_label[idx]

            tmp_data = batch_data[idx]
            tmp_data = np.asarray(tmp_data).reshape((1, len(tmp_data)))

            label = np.append(label, np.eye(self._class_len)[int(np.asfarray(tmp_label))].reshape(1, self._class_len), axis=0)
            data = np.append(data, tmp_data, axis=0)

        return data, label

def dataset_constructor():

    f = open(PACK_PATH+"/dataset/format.txt", 'r')
    class_len = int(f.readline())
    data_len = int(f.readline())
    height = int(f.readline())
    width = int(f.readline())
    channel = int(f.readline())
    f.close()

    train = DataSet(who_am_i="train", class_len=class_len, data_len=data_len, height=height, width=width, channel=channel)
    test = DataSet(who_am_i="test", class_len=class_len, data_len=data_len, height=height, width=width, channel=channel)
    valid = DataSet(who_am_i="valid", class_len=class_len, data_len=data_len, height=height, width=width, channel=channel)

    return base.Datasets(train=train, test=test, validation=valid)
