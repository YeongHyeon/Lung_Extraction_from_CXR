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

    img = cvf.load_image(path="/home/visionlab/Desktop/total/pul edema_post.bmp")
    print(img.shape)

    gray = cvf.rgb2gray(rgb=img)
    print(gray.shape)

    res = cvf.resizing(image=gray, width=500)
    print(res.shape)
    print("AVG: "+str(np.average(res)))

    feed = cvf.feeding_outside_filter(binary_img=res, thresh=100)
    cvf.save_image(path=PACK_PATH+"/images/", filename="feed.png", image=feed)

    movavg = cvf.moving_avg_filter(binary_img=feed, k_size=10)
    cvf.save_image(path=PACK_PATH+"/images/", filename="aveage.png", image=movavg)

    ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.7, 255, cv2.THRESH_BINARY_INV)
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

    origin = cvf.load_image(path=filename)
    gray = cvf.rgb2gray(rgb=origin)
    resized = cvf.resizing(image=gray, width=500)
    avg = np.average(resized)

    feed = cvf.feeding_outside_filter(binary_img=resized, thresh=100)
    movavg = cvf.moving_avg_filter(binary_img=feed, k_size=int(resized.shape[0]/50))

    ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.7, 255, cv2.THRESH_BINARY_INV)

    contours = cvf.contouring(binary_img=thresh)
    boxes = cvf.contour2box(contours=contours, padding=50)

    cnt = 0
    loop = len(boxes)
    for idx in range(loop):
        x, y, w, h = boxes[idx]

        if((x > 0) and (y > 0)):
            if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):

                for idx2 in range(loop):
                    if(idx != idx2):
                        x2, y2, w2, h2 = boxes[idx2]

                        if((x2 > 0) and (y2 > 0)):
                            if((x2+w2 < resized.shape[1]) and (y2+h2 < resized.shape[0])):
                                if((w*0.5 <= x2+w2-x) and (w >= x2+w2-x)):
                                    if(((y <= y2) and (y+h <= y2)) or ((y <= y2+h2) and (y+h <= y2+h2))):
                                        pass
                                        # boxes.append([min(x, x2), min(y, y2), max(x+w, x2+w2)-min(x, x2), max(y+h, y2+h2)-min(y, y2)])

    resized = cvf.resizing(image=gray, width=500)
    # for box in boxes:
    #     x, y, w, h = box
    #
    #     if((x > 0) and (y > 0)):
    #         if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):
    #             if(not(util.check_path(path=PACK_PATH+"/images/"+str(tmp_file)))):
    #                 util.make_path(path=PACK_PATH+"/images/"+str(tmp_file))
    #
    #             cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_0_"+str(cnt)+".png", image=thresh[y:y+h, x:x+w])
    #             pad = cvf.zero_padding(image=thresh[y:y+h, x:x+w], height=500, width=500)
    #             cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_1_"+str(cnt)+".png", image=pad)
    #             cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_2_"+str(cnt)+".png", image=resized[y:y+h, x:x+w])
    #             cnt += 1

    box_comb = []
    for box1 in boxes:
        x1, y1, w1, h1 = box1

        for box2 in boxes:
            x2, y2, w2, h2 = box2
            if((box1 == box2) or ((x1 <= x2) and (y1 <= y2) and (x1+w1 >= x1+w1) and (y1+h1 >= y1+h1))):
                continue

            x_start = min(x1,x2)
            y_start = min(y1,y2)
            x_end = max(x1+w1,x2+w2)
            y_end = max(y1+h1,y2+h2)

            if((x_start > 0) and (y_start > 0)):
                if((x_end < resized.shape[1]) and (y_end < resized.shape[0])):
                    box_comb.append([x_start, y_start, x_end-x_start, y_end-y_start])

    cnt = 0
    for box in box_comb:
        x, y, w, h = box

        if((x > 0) and (y > 0)):
            if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):
                if(not(util.check_path(path=PACK_PATH+"/images/"+str(tmp_file)))):
                    util.make_path(path=PACK_PATH+"/images/"+str(tmp_file))

                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_0_"+str(cnt)+".png", image=thresh[y:y+h, x:x+w])
                pad = cvf.zero_padding(image=thresh[y:y+h, x:x+w], height=500, width=500)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_1_"+str(cnt)+".png", image=pad)
                cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename=str(tmp_file)+"_2_"+str(cnt)+".png", image=resized[y:y+h, x:x+w])
                cnt += 1

    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="origin.png", image=origin)
    cvf.save_image(path=PACK_PATH+"/images/"+str(tmp_file)+"/", filename="thresh.png", image=thresh)

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
    usr_path = "/home/visionlab/Desktop/total"

    if(util.check_path(usr_path)):
        list_dir = util.get_dirlist(path=usr_path, save=False)
        print(list_dir)

        for li_d in list_dir:
            list_file = util.get_filelist(directory=usr_path+"/"+li_d, extensions=extensions)

            for li_f in list_file:
                print(li_f)
                extract_segments(filename=li_f)

        list_file = util.get_filelist(directory=usr_path, extensions=extensions)

        for li_f in list_file:
            print(li_f)
            extract_segments(filename=li_f)

    else:
        print("Invalid path :"+usr_path)

if __name__ == '__main__':

    # tmp_main()
    main()
