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

    plt.savefig(PACK_PATH+"/images/"+str(tmp_file)+"/"+save_as+".png")


def extract_segments(filename):

    tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

    if(not(util.check_path(path=PACK_PATH+"/images/"+str(tmp_file)+"/"))):
        util.make_path(path=PACK_PATH+"/images/"+str(tmp_file)+"/")

    origin = cvf.load_image(path=filename)
    gray = cvf.rgb2gray(rgb=origin)
    resized = cvf.resizing(image=gray, width=500)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+".png", image=resized)

    mulmul = resized.copy()
    for i in range(20):
        ret,thresh = cv2.threshold(mulmul, np.average(mulmul)*0.3, 255, cv2.THRESH_BINARY)
        cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_thresh1.png", image=thresh)

        mulmul = cvf.normalizing(binary_img=resized*(thresh / 255))

    movavg = cvf.moving_avg_filter(binary_img=mulmul, k_size=10)
    adap = cvf.adaptiveThresholding(binary_img=movavg, neighbor=111, blur=False, blur_size=3)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_adap.png", image=255-adap)

    result = resized*((255-adap)/255)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_result1.png", image=result)

    movavg = cvf.moving_avg_filter(binary_img=result, k_size=10)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_result2.png", image=movavg)

    ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.5, 255, cv2.THRESH_BINARY_INV)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_thresh2.png", image=thresh)

    contours = cvf.contouring(binary_img=thresh)
    boxes = cvf.contour2box(contours=contours, padding=20)

    resized = cvf.resizing(image=gray, width=500)

    cnt = 0
    for box in boxes:
        x, y, w, h = box

        if((x > 0) and (y > 0)):
            if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):

                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_0_"+str(cnt)+".png", image=thresh[y:y+h, x:x+w])
                pad = cvf.zero_padding(image=thresh[y:y+h, x:x+w], height=500, width=500)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_1_"+str(cnt)+".png", image=pad)
                pad2 = cvf.remain_only_biggest(binary_img=pad)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_2_"+str(cnt)+".png", image=pad2)
                pad_res = cvf.zero_padding(image=resized[y:y+h, x:x+w], height=500, width=500)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_3_"+str(cnt)+".png", image=pad_res*(pad2/255))
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_4_"+str(cnt)+".png", image=resized[y:y+h, x:x+w])
                cnt += 1

    for b in boxes:
        x, y, w, h = b

        if((x > 0) and (y > 0)):
            if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):
                cv2.rectangle(resized,(x,y),(x+w,y+h),(255, 255, 255),2)
                cv2.rectangle(thresh,(x,y),(x+w,y+h),(255, 255, 255),2)

    # cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="opened.png", image=dilated)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="contour.png", image=thresh)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="resized.png", image=resized)

    cvf.save_image(path=PACK_PATH+"/images/", filename="resized"+str(tmp_file)+".png", image=resized)

def main():

    extensions = ["BMP", "bmp", "PNG", "png", "JPG", "jpg", "JPEG", "jpeg"]

    util.refresh_directory(PACK_PATH+"/images")

    print("Enter the path")
    # usr_path = input(">> ")
    usr_path = "/home/yeonghyeon/Desktop/images/post_processing_1113"

    if(util.check_path(usr_path)):
        files = util.get_filelist(directory=usr_path, extensions=extensions)
        for fi in files:
            print(fi)
            extract_segments(filename=fi)

    else:
        print("Invalid path :"+usr_path)

if __name__ == '__main__':

    main()
