import os, inspect
import cv2

import numpy as np

import source.cv_functions as cvf
import source.utility as util

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    # ret,thresh1 = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    # ret,thresh2 = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)
    # ret,thresh3 = cv2.threshold(res, 127, 255, cv2.THRESH_TRUNC)
    # ret,thresh4 = cv2.threshold(res, 127, 255, cv2.THRESH_TOZERO)
    # ret,thresh5 = cv2.threshold(res, 127, 255, cv2.THRESH_TOZERO_INV)

def tmp_main():

    util.refresh_directory(PACK_PATH+"/images")

    img = cvf.load_image(path="/home/yeonghyeon/Desktop/total/pul edema_post.bmp")
    print(img.shape)

    gray = cvf.rgb2gray(rgb=img)
    print(gray.shape)

    res = cvf.resizing(image=gray, width=500)
    print(res.shape)
    print("AVG: "+str(np.average(res)))

    print(np.average(res), np.average(res*2))
    cvf.save_image(path=PACK_PATH+"/images/", filename="resx2.png", image=res*res+res)

    feed = cvf.feeding_outside_filter(binary_img=res, thresh=100)
    cvf.save_image(path=PACK_PATH+"/images/", filename="feed.png", image=feed)

    movavg = cvf.moving_avg_filter(binary_img=feed, k_size=10)
    cvf.save_image(path=PACK_PATH+"/images/", filename="aveage.png", image=movavg)

    ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.8, 255, cv2.THRESH_BINARY_INV)
    cvf.save_image(path=PACK_PATH+"/images/", filename="thresh.png", image=thresh)
    # movavg = cvf.moving_avg_filter(binary_img=thresh, k_size=3)
    # cvf.save_image(path=PACK_PATH+"/images/", filename="aveage2.png", image=movavg)

    contours = cvf.contouring(binary_img=thresh)

    boxes = cvf.contour2box(contours=contours, padding=15)

    res = cvf.resizing(image=gray, width=500)
    cnt = 0
    for b in boxes:
        x, y, w, h = b

        if((x > 0) and (y > 0)):
            if((x+w < res.shape[1]) and (y+h < res.shape[0])):
                cvf.save_image(path=PACK_PATH+"/images/", filename="box_0_"+str(cnt)+".png", image=res[y:y+h, x:x+w])
                cnt += 1

    cnt = 0
    for box1 in boxes:
        x1, y1, w1, h1 = box1

        for box2 in boxes:
            x2, y2, w2, h2 = box2

            x_crop = min(x1,x2)
            y_crop = min(y1,y2)
            w_crop = max(x1+w1,x2+w2)
            h_crop = max(y1+h1,y2+h2)

            if((x_crop > 0) and (y_crop > 0)):
                if((x_crop+w_crop < res.shape[1]) and (y_crop+h_crop < res.shape[0])):
                    cvf.save_image(path=PACK_PATH+"/images/", filename="box_1_"+str(cnt)+".png", image=res[y_crop:h_crop, x_crop:w_crop])
                    cnt += 1

    for box1 in boxes:
        x1, y1, w1, h1 = box1

        for box2 in boxes:
            x2, y2, w2, h2 = box2

            x_crop = min(x1,x2)
            y_crop = min(y1,y2)
            w_crop = max(x1+w1,x2+w2)
            h_crop = max(y1+h1,y2+h2)

            if((x_crop > 0) and (y_crop > 0)):
                if((x_crop+w_crop < res.shape[1]) and (y_crop+h_crop < res.shape[0])):
                    cv2.rectangle(res,(x_crop,y_crop),(x_crop+w_crop,y_crop+h_crop),(255, 255, 255),2)
                    cv2.rectangle(thresh,(x_crop,y_crop),(x_crop+w_crop,y_crop+h_crop),(255, 255, 255),2)
    # for b in boxes:
    #     x, y, w, h = b
    #
    #     if((x > 0) and (y > 0)):
    #         if((x+w < res.shape[1]) and (y+h < res.shape[0])):
    #             cv2.rectangle(res,(x,y),(x+w,y+h),(255, 255, 255),2)
    #             cv2.rectangle(thresh,(x,y),(x+w,y+h),(255, 255, 255),2)


    cvf.save_image(path=PACK_PATH+"/images/", filename="withbox.png", image=res)
    cvf.save_image(path=PACK_PATH+"/images/", filename="withbox_thre.png", image=thresh)

    # cv2.imshow('image',res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def extract_segments(filename):

    tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

    if(not(util.check_path(path=PACK_PATH+"/images/"+str(tmp_file)))):
        util.make_path(path=PACK_PATH+"/images/"+str(tmp_file))

    origin = cvf.load_image(path=filename)
    gray = cvf.rgb2gray(rgb=origin)
    resized = cvf.resizing(image=gray, width=500)
    avg = np.average(resized)

    # feed = cvf.feeding_outside_filter(binary_img=resized, thresh=100)
    # cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="feed.png", image=feed)
    movavg = cvf.moving_avg_filter(binary_img=resized, k_size=10)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="movavg.png", image=movavg)

    ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.5, 255, cv2.THRESH_BINARY_INV)

    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="origin.png", image=origin)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="thresh.png", image=thresh)

    contours = cvf.contouring(binary_img=thresh)
    boxes = cvf.contour2box(contours=contours, padding=50)

    resized = cvf.resizing(image=gray, width=500)

    cnt = 0
    for box in boxes:
        x, y, w, h = box

        if((x > 0) and (y > 0)):
            if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):

                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_0_"+str(cnt)+".png", image=thresh[y:y+h, x:x+w])
                pad = cvf.zero_padding(image=thresh[y:y+h, x:x+w], height=500, width=500)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_1_"+str(cnt)+".png", image=pad)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_2_"+str(cnt)+".png", image=resized[y:y+h, x:x+w])
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
    usr_path = input(">> ")
    # usr_path = "/home/visionlab/Desktop/total"

    if(util.check_path(usr_path)):
        files = util.get_filelist(directory=usr_path, extensions=extensions)
        for fi in files:
            print(fi)
            extract_segments(filename=fi)

    else:
        print("Invalid path :"+usr_path)

if __name__ == '__main__':

    # tmp_main()
    main()
