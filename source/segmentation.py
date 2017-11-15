import os, sys, inspect
import cv2

import tensorflow as tf
import numpy as np

import source.utility as util
import source.cv_functions as cvf

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

# def combine_boxs(image=None, boxes=None):
#
#     box_comb = []
#     for box1 in boxes:
#         x1, y1, w1, h1 = box1
#
#         box_comb.append([x1, y1, w1, h1])
#         for box2 in boxes:
#             x2, y2, w2, h2 = box2
#
#             if((x1 <= x2) and (y1 <= y2) and (x1+w1 >= x1+w1) and (y1+h1 >= y1+h1)): # rid duplicated iamge
#                 continue
#
#             x_start = min(x1,x2)
#             y_start = min(y1,y2)
#             x_end = max(x1+w1,x2+w2)
#             y_end = max(y1+h1,y2+h2)
#
#             if((x_start > 0) and (y_start > 0)):
#                 if((x_end < image.shape[1]) and (y_end < image.shape[0])):
#                     box_comb.append([x_start, y_start, x_end-x_start, y_end-y_start])
#
#     return box_comb

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
                    box_right.append([rx, ry, rw, rh, result, acc])
                    cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_concat_"+str(cnt)+"_"+str(int((acc_r+acc_r)/2*100))+".png", image=image[y_start:y_end, x_start:x_end])
                    cnt += 1

    boxes_pred = sorted(boxes_pred, key=lambda l:l[5], reverse=True) # sort by acc

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
        origin_clone = origin
        gray = cvf.rgb2gray(rgb=origin)
        resized = cvf.resizing(image=gray, width = 500)

        avg = np.average(resized)

        # ret,thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
        #
        # erosed = cvf.erosion(binary_img=thresh, k_size=3, iterations=7)
        # cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_gray.png", image=gray)
        # cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_thresh.png", image=thresh)
        # cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_erose.png", image=erosed)
        #
        # contours = cvf.contouring(binary_img=erosed)
        # boxes = cvf.contour2box(contours=contours, padding=50)

        feed = cvf.feeding_outside_filter(binary_img=resized, thresh=100)
        movavg = cvf.moving_avg_filter(binary_img=feed, k_size=int(resized.shape[0]/50))

        ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.7, 255, cv2.THRESH_BINARY_INV)
        cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_thresh.png", image=thresh)

        contours = cvf.contouring(binary_img=thresh)
        boxes = cvf.contour2box(contours=contours, padding=50)
        # boxes = combine_boxs(image=thresh, boxes=boxes_tmp)

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
                        prob = sess.run(prediction, feed_dict={x_holder:convert_image(image=pad, height=height, width=width, channel=channel), training:False})
                        result = str(content[int(np.argmax(prob))])
                        acc = np.max(prob)

                        boxes_pred.append([x, y, w, h, result, acc])

                        cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_"+str(result)+"_"+str(int(acc*100))+"_"+str(cnt)+".png", image=pad)

            boxes_pred = sorted(boxes_pred, key=lambda l:l[4], reverse=True) # sort by result
            boxes_pred = sorted(boxes_pred, key=lambda l:l[5], reverse=True) # sort by acc

            ratio = round(origin.shape[0] / resized.shape[0])

            save_crops(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)
            concatenate(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)

            origin_clone = draw_boxes(image=origin_clone, boxes=boxes_pred, ratio=ratio, file_name=tmp_file)
            cvf.save_image(path=PACK_PATH+"/results/", filename=str(tmp_file)+"_origin.png", image=origin_clone)

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
