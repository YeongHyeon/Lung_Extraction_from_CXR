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
        tmp_sub = tmp_name[len(tmp_name)-2]
        tmp_file = tmp_name[len(tmp_name)-2]+"_"+tmp_name[len(tmp_name)-1].split(".")[0]

        dir_sp = usr_path.split("/")
        main_dir = ""
        for ds in dir_sp[:len(dir_sp)-1]:
            main_dir += ds
            main_dir += "/"
        main_dir += dir_sp[len(dir_sp)-1]+"_bmp"

        if(not(util.check_path(path=main_dir))):
            util.make_path(path=main_dir)
        if(not(util.check_path(path=main_dir+"/"+tmp_sub))):
            util.make_path(path=main_dir+"/"+tmp_sub)

        dicom_data = dicom.read_file(fi)
        dicom_numpy = dicom_data.pixel_array

        sumx = sum(sum(dicom_numpy))/(dicom_numpy.shape[0]*dicom_numpy.shape[1])

        cvf.save_image(path=main_dir+"/"+tmp_sub+"/", filename=tmp_file+".bmp", image=dicom_numpy/sumx)
