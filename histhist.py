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
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_avg.png", image=movavg)

    gray_updown = np.zeros((0, movavg.shape[1]), float)
    hist_sum = np.zeros((movavg.shape[1]), float)
    cnt = 0
    for rt in movavg:
        hist_sum = np.sum([hist_sum, rt], axis=0)
        if(cnt % 10 == 0):
            histo = hist_sum / movavg.shape[0]
            histo_a = np.asarray(histo).reshape((1, len(histo)))
            for i in range(10):
                if(gray_updown.shape[0] == movavg.shape[0]):
                    break
                gray_updown = np.append(gray_updown, histo_a, axis=0)
            hist_sum = np.zeros((movavg.shape[1]), float)
        cnt += 1

    gray_updown = cvf.normalizing(binary_img=gray_updown)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_updown.png", image=gray_updown)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_updown2.png", image=255-gray_updown)

    resized_tr = np.transpose(movavg)
    gray_sideside = np.zeros((0, resized_tr.shape[1]), float)
    hist_sum = np.zeros((resized_tr.shape[1]), float)
    cnt = 0
    for rt in resized_tr:
        hist_sum = np.sum([hist_sum, rt], axis=0)
        if(cnt % 10 == 0):
            histo = hist_sum / resized_tr.shape[0]
            histo_a = np.asarray(histo).reshape((1, len(histo)))
            for i in range(10):
                if(gray_sideside.shape[0] == movavg.shape[1]):
                    break
                gray_sideside = np.append(gray_sideside, histo_a, axis=0)
            hist_sum = np.zeros((resized_tr.shape[1]), float)
        cnt += 1

    gray_sideside = cvf.normalizing(binary_img=np.transpose(gray_sideside))
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_sideside.png", image=gray_sideside)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_sideside2.png", image=255-gray_sideside)

    norm1 = cvf.normalizing(binary_img=gray_updown*gray_sideside)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_mul1.png", image=norm1)

    norm2 = cvf.normalizing(binary_img=(255 - norm1)**2)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_mul2.png", image=norm2)

    norm3 = cvf.normalizing(binary_img=resized + norm2)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_summul.png", image=norm3)

    ret,thresh = cv2.threshold(norm2, 254, 255, cv2.THRESH_BINARY)
    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_trhe.png", image=thresh)

    # thresh[:10] = 255
    # thresh[thresh.shape[0]-10:] = 255
    # thresh[:, :10] = 255
    # thresh[:, thresh.shape[1]-10:] = 255
    # cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_trhe2.png", image=thresh)

    thresh = cvf.normalizing(binary_img=thresh)
    contours = cvf.contouring(binary_img=thresh)
    boxes = cvf.contour2box(contours=contours, padding=5)

    for b in boxes:
        x, y, w, h = b

        if((x > 0) and (y > 0)):
            if((x+w < thresh.shape[1]) and (y+h < thresh.shape[0])):
                thresh[y:y+h, x:x+w] = 0

    cvf.save_image(path=PACK_PATH+"/images/", filename=str(tmp_file)+"_trhe3.png", image=thresh)

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
