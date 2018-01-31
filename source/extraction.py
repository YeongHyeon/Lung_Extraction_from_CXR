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
                    # cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_"+str(result)+"_"+str(cnt)+"_crop"+".png", image=image[ry:ry+rh, rx:rx+rw])
                    cnt += 1

def draw_boxes(image=None, boxes=None, ratio=1, file_name=None):

    for box in boxes:
        x, y, w, h, result, acc = box
        rx, ry, rw, rh = x * ratio, y * ratio, w * ratio, h * ratio

        if((rx > 0) and (ry > 0)):
            if((rx+rw < image.shape[1]) and (ry+rh < image.shape[0])):
                if((result == "lung_left") or (result == "lung_right")):
                    # cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 5)
                    cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                    # cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif(result == "lung"):
                    # cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 5)
                    cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
                    # cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif(result == "others"):
                    pass
                else:
                    # cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (255, 255, 255), 5)
                    cv2.rectangle(image, (rx, ry), (rx+rw, ry+rh), (0, 0, 255), 2)
                    # cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(image, result, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    pass

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
    tmp_boxes = []
    for box_r in box_right:
        x_r, y_r, w_r, h_r, result_r, acc_r = box_r

        for box_l in box_left:
            x_l, y_l, w_l, h_l, result_l, acc_l = box_l

            center_rx = x_r + (w_r / 2)
            center_ry = y_r + (h_r / 2)
            center_lx = x_l + (w_l / 2)
            center_ly = y_l + (h_l / 2)

            dist_limit = max(h_r, h_l) / 2

            if(abs(center_ry - center_ly) > dist_limit): # concat by y relation.
                continue
            else:
                x_start = min(x_r,x_l)
                y_start = min(y_r,y_l)
                x_end = max(x_r+w_r,x_l+w_l)
                y_end = max(y_r+h_r,y_l+h_l)

                if((x_start > 0) and (y_start > 0)):
                    if((x_end < image.shape[1]) and (y_end < image.shape[0])):
                        tmp_boxes.append([x_start, y_start, x_end-x_start, y_end-y_start, "lung", (acc_r+acc_l)/2])

    box_concat = []
    try:
        max_idx = 0
        tmp_size = 0
        for idx in range(len(tmp_boxes)):
            x, y, w, h, result, acc = tmp_boxes[idx]

            if((w * h) > tmp_size):
                tmp_size = w * h
                max_idx = idx

        x, y, w, h, result, acc = tmp_boxes[max_idx]
        box_concat.append([x, y, w, h, result, acc])
        cvf.save_image(path=PACK_PATH+"/results/"+str(file_name)+"/", filename=str(file_name)+"_concat_"+str(cnt)+"_"+str(int((acc_r+acc_l)/2*100))+".png", image=image[y:y+h, x:x+w])
        cnt += 1
    except:
        pass
    # try:
    #     max_idx = 0
    #     tmp_acc = 0
    #     for idx in range(len(tmp_boxes)):
    #         x, y, w, h, result, acc = tmp_boxes[idx]
    #
    #         if(acc > tmp_acc):
    #             tmp_acc = acc
    #             max_idx = idx
    #
    #     x, y, w, h, result, acc = tmp_boxes[max_idx]
    #     box_concat.append([x, y, w, h, result, acc])
    # except:
    #     pass

    return box_concat # return only one box

def intersection_over_union(filename="", boxes=None, ratio=None):

    #boxes is only one box.(concatenated lung)

    f = open(PACK_PATH+"/BBox_List_2017.csv")
    lines = f.readlines()
    f.close

    iou = 0
    bbox = []
    for idx in range(len(lines)):
        if(idx == 0):
            continue
        imgname, diagnosis, box_x, box_y, box_w, box_h  = lines[idx].split(",")
        box_x, box_y, box_w, box_h = float(box_x)/ratio, float(box_y)/ratio, float(box_w)/ratio, float(box_h)/ratio

        area = abs((box_x+box_w)-box_x) * abs((box_y+box_h)-box_y)
        if imgname in filename:
            bi_x, bi_y, bi_w, bi_h = int(box_x), int(box_y), int(box_w), int(box_h)
            bbox.append([bi_x, bi_y, bi_w, bi_h, str(diagnosis), int(100)])
            print(imgname)

            for lung in boxes:
                lung_x, lung_y, lung_w, lung_h, label, score = lung

                # four points from diagnosis bbox.
                dotlist = []
                dotlist.append([box_x, box_y])
                dotlist.append([box_x+box_w, box_y])
                dotlist.append([box_x, box_y+box_h])
                dotlist.append([box_x+box_w, box_y+box_h])

                indot = []
                outdot = []
                for dot in range(len(dotlist)):
                    dot_x, dot_y = dotlist[dot]
                    if((lung_x <= dot_x) and (lung_y <= dot_y) and (lung_x+lung_w >= dot_x) and (lung_y+lung_h >= dot_y)):
                        indot.append(dot)
                    else:
                        outdot.append(dot)

                if(len(indot) >= 3): # four or three points in lung box.
                    iou = 1
                elif(len(indot) == 2): # two points in lung box
                    inter = 0
                    if((indot[0] == 0) and (indot[1] == 1)):
                        inter = abs((box_x+box_w)-box_x) * abs((lung_y+lung_h)-box_y)
                    elif((indot[0] == 0) and (indot[1] == 2)):
                        inter = abs((box_y+box_h)-box_y) * abs((lung_x+lung_w)-box_x)
                    elif((indot[0] == 1) and (indot[1] == 3)):
                        inter = abs((box_y+box_h)-box_y) * abs((box_x+box_w)-lung_x)
                    elif((indot[0] == 2) and (indot[1] == 3)):
                        inter = abs((box_x+box_w)-box_x) * abs((box_y+box_h)-lung_y)
                    if(inter != 0):
                        iou = inter / area
                    else:
                        iou = 0
                elif(len(indot) == 1): # one points in lung box
                    inter = 0
                    dot_x, dot_y = dotlist[indot[0]]
                    if(indot[0] == 0):
                        inter = abs((lung_x+lung_w)-dot_x) * abs((lung_y+lung_h)-dot_y)
                    elif(indot[0] == 1):
                        inter = abs((lung_x)-dot_x) * abs((lung_y+lung_h)-dot_y)
                    elif(indot[0] == 2):
                        inter = abs((lung_x+lung_w)-dot_x) * abs((lung_y)-dot_y)
                    elif(indot[0] == 3):
                        inter = abs((lung_x)-dot_x) * abs((lung_y)-dot_y)
                    if(inter != 0):
                        iou = inter / area
                    else:
                        iou = 0
                else: # zero points in lung box
                    iou = 0
            break

    return iou, bbox

def convert_image(image=None, height=None, width=None, channel=None):

    resized_image = cv2.resize(image, (width, height))

    return np.asarray(resized_image).reshape((1, height*width*channel))

def extract_lung(usr_path, extensions=None,
                     height=None, width=None, channel=None,
                     sess=None, x_holder=None, training=None,
                     prediction=None, saver=None):

    if(not(util.check_path(path=PACK_PATH+"/results/"))):
        util.make_path(path=PACK_PATH+"/results/")

    summf = open(PACK_PATH+"/results/summary.csv", "w")
    summf.write("FILENAME")
    summf.write(",")
    summf.write("DETECT")
    summf.write(",")
    summf.write("IOU")
    summf.write("\n")

    files = util.get_filelist(directory=usr_path, extensions=extensions)
    files.sort()
    for filename in files:
        print(filename)

        if(util.check_file(filename=filename)):
            tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

            if(not(util.check_path(path=PACK_PATH+"/results/"+str(tmp_file)+"/"))):
                util.make_path(path=PACK_PATH+"/results/"+str(tmp_file)+"/")

            origin = cvf.load_image(path=filename)
            try:
                gray = cvf.rgb2gray(rgb=origin)
            except: # if origin image is grayscale
                gray = origin
            resized = cvf.resizing(image=gray, width=500)
            cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre1_origin.png", image=resized)

            mulmul = resized.copy()
            for i in range(20):
                ret,thresh = cv2.threshold(mulmul, np.average(mulmul)*0.3, 255, cv2.THRESH_BINARY)
                cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre2_thresh.png", image=thresh)

                mulmul = cvf.normalizing(binary_img=resized*(thresh / 255))
                cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre3_normalize.png", image=mulmul)

            movavg = cvf.moving_avg_filter(binary_img=mulmul, k_size=10)
            adap = cvf.adaptiveThresholding(binary_img=movavg, neighbor=111, blur=False, blur_size=3)
            cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre4_adaptrhesh.png", image=255-adap)

            masking = resized*((255-adap)/255)
            cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre5_mask1.png", image=masking)

            movavg = cvf.moving_avg_filter(binary_img=masking, k_size=5)
            cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre6_mask2.png", image=movavg)

            ret,thresh = cv2.threshold(movavg, np.average(movavg)*0.5, 255, cv2.THRESH_BINARY_INV)
            cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_pre7_thresh.png", image=thresh)

            contours = cvf.contouring(binary_img=thresh)
            boxes_tmp = cvf.contour2box(contours=contours, padding=20)
            boxes = cvf.rid_repetition(boxes=boxes_tmp, binary_img=thresh)

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
                            pad2 = cvf.remain_only_biggest(binary_img=pad)
                            pad_res = cvf.zero_padding(image=resized[y:y+h, x:x+w], height=500, width=500)

                            xdata = pad_res*(pad2/255)

                            prob = sess.run(prediction, feed_dict={x_holder:convert_image(image=xdata, height=height, width=width, channel=channel), training:False})
                            result = str(content[int(np.argmax(prob))])
                            acc = np.max(prob)

                            boxes_pred.append([x, y, w, h, result, acc])

                            # cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_"+str(result)+"_"+str(int(round(acc, 2)*100))+"_"+str(cnt)+".png", image=xdata)

                            cnt += 1

                boxes_pred = sorted(boxes_pred, key=lambda l:l[4], reverse=True) # sort by result
                boxes_pred = sorted(boxes_pred, key=lambda l:l[5], reverse=True) # sort by acc

                ratio = origin.shape[0] / resized.shape[0]

                save_crops(image=resized, boxes=boxes_pred, ratio=1, file_name=tmp_file)
                concats = concatenate(image=resized, boxes=boxes_pred, ratio=1, file_name=tmp_file)

                iou, bbox = intersection_over_union(filename=filename, boxes=concats, ratio=ratio)
                summf.write(str(filename))
                summf.write(",")
                summf.write(str(len(concats)))
                summf.write(",")
                summf.write(str(iou))
                summf.write("\n")

                origin_res1 = cvf.resizing(image=origin, width=500)
                origin_res2 = origin_res1.copy()
                origin_res3 = origin_res1.copy()

                origin_res_lr = draw_boxes(image=origin_res1, boxes=boxes_pred, ratio=1, file_name=tmp_file)
                cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_origin_lr.png", image=origin_res_lr)
                origin_res_concat1 = draw_boxes(image=origin_res1, boxes=concats, ratio=1, file_name=tmp_file)
                cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_origin_lr_and_concat.png", image=origin_res_concat1)
                origin_res_concat2 = draw_boxes(image=origin_res2, boxes=concats, ratio=1, file_name=tmp_file)
                cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_origin_concat.png", image=origin_res_concat2)
                if(len(bbox) > 0):
                    origin_res_bbox = draw_boxes(image=origin_res3, boxes=bbox, ratio=1, file_name=tmp_file)
                    cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_origin_bbox.png", image=origin_res_bbox)
                    origin_res_concat3 = draw_boxes(image=origin_res3, boxes=concats, ratio=1, file_name=tmp_file)
                    cvf.save_image(path=PACK_PATH+"/results/"+str(tmp_file)+"/", filename=str(tmp_file)+"_origin_concat_bbox.png", image=origin_res_concat3)


            else:
                print("You must training first!")
        else:
            print("Invalid File: "+str(filename))
    summf.close()
