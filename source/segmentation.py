import os, sys, inspect
import cv2

import tensorflow as tf
import numpy as np

import source.utility as util
import source.cv_functions as cvf

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def save_crops(image=None, boxes=None, ratio=1, file_name=None): # save the segments

    cnt = 0
    for box in boxes:
        x, y, w, h, result, acc = box
        rx, ry, rw, rh = x * ratio, y * ratio, w * ratio, h * ratio

        if((rx > 0) and (ry > 0)):
            if((rx+rw < image.shape[1]) and (ry+rh < image.shape[0])):
                if((result == "lung_left") or (result == "lung_right")):
                    cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_"+str(result)+"_"+str(cnt)+"_"+str(int(acc*100))+".png", image=image[ry:ry+rh, rx:rx+rw])
                    cnt += 1

def draw_boxes(image=None, boxes=None, ratio=1, file_name=None):

    for box in boxes:
        x, y, w, h, result, acc = box
        rx, ry, rw, rh = x * ratio, y * ratio, w * ratio, h * ratio

        if((rx > 0) and (ry > 0)):
            if((rx+rw < image.shape[1]) and (ry+rh < image.shape[0])):
                if((result == "lung_left") or (result == "lung_right")):
                    cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 5)
                    cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 3)
                    cv2.putText(image, result+" "+str(int(acc*100))+"%", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
                    cv2.putText(image, result+" "+str(int(acc*100))+"%", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


    cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_origin.png", image=image)

    return image

def concatenate(image=None, boxes=None, ratio=1, file_name=None):

    box_left = []
    box_right = []
    for box in boxes:
        x, y, w, h, result, acc = box
        rx, ry, rw, rh = x * ratio, y * ratio, w * ratio, h * ratio

        if((rx > 0) and (ry > 0)):
            if((rx+rw < image.shape[1]) and (ry+rh < image.shape[0])):
                if(result == "lung_left"):
                    box_left.append([rx, ry, rw, rh, result, acc])
                elif(result == "lung_right"):
                    box_right.append([rx, ry, rw, rh, result, acc])

    cnt = 0
    box_concat = []
    for box_r in box_right:
        x_r, y_r, w_r, h_r, result_r, acc_r = box_r

        for box_l in box_left:
            x_l, y_l, w_l, h_l, result_l, acc_l = box_l

            x_start = min(x_r,x_l)
            y_start = min(y_r,y_l)
            x_end = max(x_r+w_r,x_l+w_l)
            y_end = max(y_r+h_r,y_l+h_l)

            if((x_start > 0) and (y_start > 0)):
                if((x_end < image.shape[1]) and (y_end < image.shape[0])):
                    box_concat.append([rx, ry, rw, rh, result, acc])
                    cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_concat_"+str(cnt)+"_"+str(int((acc_r+acc_r)/2*100))+".png", image=image[y_start:y_end, x_start:x_end])
                    cnt += 1

def convert_image(image=None, height=None, width=None, channel=None):

    resized_image = cv2.resize(image, (width, height))

    return np.asarray(resized_image).reshape((1, height*width*channel))

def extract_segments(filename,
                     height=None, width=None, channel=None,
                     sess=None, x_holder=None, training=None,
                     prediction=None, saver=None):

    if(util.check_file(filename=filename)):
        tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

        if(not(util.check_path(path=PACK_PATH+"/results/"))):
            util.make_path(path=PACK_PATH+"/results/")
        if(not(util.check_path(path=PACK_PATH+"/results/"+str(tmp_file)+"/"))):
            util.make_path(path=PACK_PATH+"/results/"+str(tmp_file)+"/")

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

        if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
            saver.restore(sess, PACK_PATH+"/checkpoint/checker")
            f = open(PACK_PATH+"/dataset/labels.txt", 'r')
            content = f.readlines()
            f.close()
            for idx in range(len(content)):
                content[idx] = content[idx][:len(content[idx])-1] # rid \n

            boxes_pred = []
            cnt = 0
            for b in boxes:
                x, y, w, h = b
                if((x > 0) and (y > 0)):
                    if((x+w < resized.shape[1]) and (y+h < resized.shape[0])):

                        pad = cvf.zero_padding(image=thresh[y:y+h, x:x+w], height=500, width=500)
                        pad2 = cvf.remain_only_center(binary_img=pad)
                        pad_res = cvf.zero_padding(image=resized[y:y+h, x:x+w], height=500, width=500)

                        xdata = pad_res*(pad2/255)

                        prob = sess.run(prediction, feed_dict={x_holder:convert_image(image=xdata, height=height, width=width, channel=channel), training:False})
                        result = str(content[int(np.argmax(prob))])
                        acc = np.max(prob)

                        boxes_pred.append([x, y, w, h, result, acc])

                        cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_"+str(result)+"_"+str(int(acc*100))+"_"+str(cnt)+".png", image=xdata)

            boxes_pred = sorted(boxes_pred, key=lambda l:l[4], reverse=True) # sort by result
            boxes_pred = sorted(boxes_pred, key=lambda l:l[5], reverse=True) # sort by acc

            ratio = round(origin.shape[0] / resized.shape[0])

            # save_crops(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)
            save_crops(image=resized, boxes=boxes_pred, ratio=1, file_name=tmp_file)
            # concatenate(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)
            concatenate(image=resized, boxes=boxes_pred, ratio=1, file_name=tmp_file)

            # origin_clone = draw_boxes(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)
            # cvf.save_image(path=PACK_PATH+"/results/", filename=str(tmp_file)+"_origin.png", image=origin_clone)

            origin_res = cvf.resizing(image=origin, width=500)
            resized_clone = draw_boxes(image=origin_res, boxes=boxes_pred, ratio=1, file_name=tmp_file)
            cvf.save_image(path=PACK_PATH+"/results/", filename=str(tmp_file)+"_origin.png", image=resized_clone)

            # while(True):
            #     cv2.imshow('Image', origin_clone)
            #
            #     key = cv2.waitKey(1) & 0xFF
            #     if(key == ord("q")):
            #         print("window is closed.")
            #         break
            #
            # cv2.destroyAllWindows()
        else:
            print("You must training first!")
    else:
        print("Invalid File: "+str(filename))
