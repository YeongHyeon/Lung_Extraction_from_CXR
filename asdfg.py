import os, inspect
import cv2

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import source.cv_functions as cvf
import source.utility as util

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def save_amp_and_frequency(sub="sample", save_as="sample", data=None, refresh=False):

    plt.clf()
    plt.figure(1)
    plt.plot(data)
    plt.xlabel("bin")
    plt.ylabel("Freq")

    plt.savefig(PACK_PATH+"/images/"+save_as+".png")

def extract_segments(filename):

    tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

    origin = cvf.load_image(path=filename)
    gray = cvf.rgb2gray(rgb=origin)
    resized = cvf.resizing(image=gray, width=500)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+".png", image=resized)

    movavg = cvf.moving_avg_filter(binary_img=resized, k_size=10)

    print(np.average(movavg)*0.3)
    mulmul = movavg.copy()
    for i in range(20):
        ret,thresh = cv2.threshold(mulmul, np.average(mulmul)*0.3, 255, cv2.THRESH_BINARY)
        cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_thresh.png", image=thresh)

        mulmul = cvf.normalizing(binary_img=resized*(thresh / 255))
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_mul.png", image=mulmul)

def main():

    extensions = ["BMP", "bmp", "PNG", "png", "JPG", "jpg", "JPEG", "jpeg"]

    util.refresh_directory(PACK_PATH+"/images")

    print("Enter the path")
    # usr_path = input(">> ")
    usr_path = "/home/yeonghyeon/Desktop/images/lung_image_20"

    if(util.check_path(usr_path)):
        files = util.get_filelist(directory=usr_path, extensions=extensions)
        for fi in files:
            print(fi)
            extract_segments(filename=fi)

    else:
        print("Invalid path :"+usr_path)

if __name__ == '__main__':

    main()
