import os, inspect
import cv2

import numpy as np

import source.cv_functions as cvf
import source.utility as util

extensions = ["DCM", "dcm"]
import dicom
import tifffile as tiff

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

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
            dicom_normal = (dicom_numpy / sumx) * (2**7-1)

            # area1 = np.mean(dicom_normal[:int(dicom_numpy.shape[0]/4), :int(dicom_numpy.shape[1]/4)])
            # area2 = np.mean(dicom_normal[int(dicom_numpy.shape[0]/4*3):, :int(dicom_numpy.shape[1]/4)])
            # area3 = np.mean(dicom_normal[:int(dicom_numpy.shape[0]/4), int(dicom_numpy.shape[1]/4*3):])
            # area4 = np.mean(dicom_normal[int(dicom_numpy.shape[0]/4*3):, int(dicom_numpy.shape[1]/4*3):])
            #
            # threshold = np.mean([area1, area2, area3, area4])
            # if(threshold > 127):
            #     dicom_normal = 255 - dicom_normal
            cvf.save_image(path=main_dir, filename=tmp_file+".bmp", image=dicom_normal)

            dist = (2**16-1) / np.max(dicom_numpy)
            dicom_normal = dicom_numpy * dist

            dicom_normal, _ = image_histogram_equalization(image=dicom_numpy, number_bins=int(2**(16/1)))
            print(np.max(dicom_normal))

            tiff.imsave(main_dir+tmp_file+".tiff", (dicom_normal*(2**0)).astype(np.uint16))

            tmp = tiff.imread(main_dir+tmp_file+".tiff")
            print(np.max(tmp))
