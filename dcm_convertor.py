import os, inspect
import cv2

import numpy as np

import source.cv_functions as cvf
import source.utility as util

extensions = ["DCM", "dcm"]
import dicom


print("\nEnter the path")
usr_path = input(">> ")
if(util.check_path(usr_path)):
    files = util.get_filelist(directory=usr_path, extensions=extensions)
    for fi in files:
        print("Convert: "+str(fi))

        tmp_name = fi.split("/")
        tmp_file = tmp_name[len(tmp_name)-1].split(".")[0]

        dir_sp = fi.split("/")
        main_dir = ""
        for ds in dir_sp[:len(dir_sp)-1]:
            main_dir += ds
            main_dir += "/"

        dicom_data = dicom.read_file(fi)
        try:
            dicom_numpy = dicom_data.pixel_array
        except:
            print("TypeError: No pixel data found in this dataset.")
        else:
            sumx = np.sum(dicom_numpy) / (dicom_numpy.shape[0]*dicom_numpy.shape[1])
            dicom_normal = (dicom_numpy / sumx) * 127

            # area1 = np.mean(dicom_normal[:int(dicom_numpy.shape[0]/4), :int(dicom_numpy.shape[1]/4)])
            # area2 = np.mean(dicom_normal[int(dicom_numpy.shape[0]/4*3):, :int(dicom_numpy.shape[1]/4)])
            # area3 = np.mean(dicom_normal[:int(dicom_numpy.shape[0]/4), int(dicom_numpy.shape[1]/4*3):])
            # area4 = np.mean(dicom_normal[int(dicom_numpy.shape[0]/4*3):, int(dicom_numpy.shape[1]/4*3):])
            #
            # threshold = np.mean([area1, area2, area3, area4])
            # if(threshold > 127):
            #     dicom_normal = 255 - dicom_normal
            # print(main_dir)
            cvf.save_image(path=main_dir, filename=tmp_file+".bmp", image=dicom_normal)
