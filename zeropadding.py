import os, inspect
import cv2

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import source.cv_functions as cvf
import source.utility as util

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def main():

    extensions = ["BMP", "bmp", "PNG", "png", "JPG", "jpg", "JPEG", "jpeg"]

    util.refresh_directory(PACK_PATH+"/images")

    print("Enter the path")
    # usr_path = input(">> ")
    usr_path = "/media/yeonghyeon/Toshiba/lung/datasets/20171204"

    if(util.check_path(usr_path)):
        files = util.get_filelist(directory=usr_path, extensions=extensions)
        for fi in files:
            print(fi)

            tmp_sub, tmp_file = util.get_dir_and_file_name(path=fi)

            if(not(util.check_path(path=PACK_PATH+"/images/"+str(tmp_file)+"/"))):
                util.make_path(path=PACK_PATH+"/images/"+str(tmp_file)+"/")

            image = cvf.load_image(path=fi)

            if(image.shape[0] > image.shape[1]): # height > width
                resized = cvf.resizing(image=image, width=int(500*(image.shape[1]/image.shape[0])), height=500)
            else:
                resized = cvf.resizing(image=image, width=500, height=int(500*(image.shape[0]/image.shape[1])))
            zeropad = cvf.zero_padding(image=resized, height=500, width=500)
            print(image.shape)
            print(resized.shape)
            print(zeropad.shape)
            cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+".png", image=zeropad)
    else:
        print("Invalid path :"+usr_path)

if __name__ == '__main__':

    main()
