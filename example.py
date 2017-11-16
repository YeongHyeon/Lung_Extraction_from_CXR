import os, inspect
import cv2

import numpy as np

import source.cv_functions as cvf
import source.utility as util

extensions = ["BMP", "bmp", "PNG", "png", "JPG", "jpg", "JPEG", "jpeg", "DCM", "dcm"]

print("\nEnter the path")
usr_path = input(">> ")

# mode = "w"
# if(util.check_path(usr_path)):
#     list_dir = util.get_dirlist(path=usr_path, save=False)
#     print(list_dir)
#
#     for li_d in list_dir:
#         list_file = util.get_filelist(directory=usr_path+"/"+li_d, extensions=extensions)
#
#         for li_f in list_file:
#             f = open("files.txt", mode)
#             li_f = li_f.replace(" ", "\ ")
#
#             tmp_name = li_f.split("/")
#             tmp_sub = tmp_name[len(tmp_name)-2]
#             tmp_file = tmp_name[len(tmp_name)-1].split(".")[0]
#
#             if(not(util.check_path(path="/home/visionlab/Desktop/images/CXR_dataSet2/"))):
#                 util.make_path(path="/home/visionlab/Desktop/images/CXR_dataSet2/")
#             if(not(util.check_path(path="/home/visionlab/Desktop/images/CXR_dataSet2/"+str(tmp_sub)))):
#                 util.make_path(path="/home/visionlab/Desktop/images/CXR_dataSet2/"+str(tmp_sub))
#
#             os.system("med2image -i "+str(li_f)+" -o /home/visionlab/Desktop/images/CXR_dataSet2/"+str(tmp_sub)+"/"+str(tmp_file)+".png")
#             f.write(str(li_f))
#             f.write("\n")
#             f.close()
#         mode = "a"

# lstFilesDCM = []  # create an empty list
# for dirName, subdirList, fileList in os.walk(usr_path):
#     for filename in fileList:
#         if ".bmp" in filename.lower():  # check whether the file's DICOM
#             lstFilesDCM.append(os.path.join(dirName,filename))
#             print(os.path.join(dirName,filename))

import dicom
ds = dicom.read_file("/home/yeonghyeon/Desktop/FILE00001.dcm")
xx = ds.pixel_array
print(Xx)
