import os, glob, shutil, psutil, inspect, random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def check_path(path):
    if(os.path.exists(path)):
        return True
    else:
        return False

def check_file(filename):
    if(os.path.isfile(filename)):
        return True
    else:
        return False

def check_memory():
    pid = os.getpid()
    proc = psutil.Process(pid)
    used_mem = proc.memory_info()[0]

    print("Memory Used: %.2f GB\t( %.2f MB )" %(used_mem/(2**30), used_mem/(2**20)))

    return used_mem

def make_path(path):
    os.mkdir(path)

def refresh_directory(path):
    if(check_path(path)):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def get_dir_and_file_name(path=None):

    tmp_name = path.split("/")
    tmp_sub = tmp_name[len(tmp_name)-2]
    # tmp_file = tmp_name[len(tmp_name)-2]+"_"+tmp_name[len(tmp_name)-1].split(".")[0]
    tmp_file = tmp_name[len(tmp_name)-1].split(".")[0]

    return tmp_sub, tmp_file

def get_dirlist(path=None, save=True): # make directory list from path

    directories = []
    for dirname in os.listdir(path):
        directories.append(dirname)

    directories.sort()

    if(save):
        f = open(PACK_PATH+"/dataset/labels.txt", "w")
        for di in directories:
            f.write(str(di))
            f.write("\n")
        f.close()

    return directories

def get_filelist(directory=None, extensions=None): # make directory list from directory with path

    file_list = []
    for dirName, subdirList, fileList in os.walk(directory):
        for filename in fileList:
            for ext in extensions:
                if ext in filename.lower():  # check whether the file's DICOM
                    filename = os.path.join(dirName,filename)
                    file_list.append(filename)

    print(str(len(file_list))+" files in "+directory)

    return file_list

def copy_file(origin, copy):

    cnt = 0
    for ori in origin:

        tmp_sub, tmp_file = get_dir_and_file_name(path=ori)
        shutil.copy(ori, copy+"/"+str(tmp_file)+str(cnt)+".png")

        cnt += 1

def shuffle_csv(filename=None):
    f = open(filename+".csv", "r")
    lines = f.readlines()
    f.close()

    random.shuffle(lines)

    f = open(filename+".csv", "w")
    f.writelines(lines)
    f.close()

def save_dataset_to_csv(save_as="sample", label=None, data=None, mode='w'):

    f = open(PACK_PATH+"/dataset/"+save_as+".csv", mode)

    f.write(str(label))
    f.write(",")

    for da in data:
        f.write(str(da))
        f.write(",")

    f.write("\n")

    f.close()

def save_graph_as_image(train_list, test_list, ylabel=""):

    print(" Save "+ylabel+" graph in ./graph")

    x = np.arange(len(train_list))
    plt.clf()
    plt.rcParams['lines.linewidth'] = 1
    plt.plot(x, train_list, label="train "+ylabel)
    plt.plot(x, test_list, label="test "+ylabel)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.ylim(-0.1, max([1, max(train_list), max(test_list)])*1.1)
    if(ylabel == "accuracy"):
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    #plt.show()

    if(not(os.path.exists("./graph"))):
        os.mkdir("./graph")
    else:
        pass
    now = datetime.now()

    plt.savefig("./graph/"+now.strftime('%Y%m%d_%H%M%S%f')+"_"+ylabel+".png")
