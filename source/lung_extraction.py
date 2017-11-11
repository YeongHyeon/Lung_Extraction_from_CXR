import os, sys, inspect
import cv2

import tensorflow as tf
import numpy as np

import source.cv_functions as cvf

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def extract_segments(filename,
                     sess=None, x=None, training=None,
                     prediction=None, saver=None):

    tmp_sub, tmp_file = util.get_dir_and_file_name(path=filename)

    origin = cvf.load_image(path=filename)
    result = origin
    gray = cvf.rgb2gray(rgb=origin)
    resized = cvf.resizing(image=gray, width = 500)

    avg = np.average(resized)

    ret,thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)

    erosed = cvf.erosion(binary_img=thresh, k_size=3, iterations=7)

    _, contours, _ = cvf.contouring(binary_img=erosed)
    boxes = cvf.contour2box(contours=contours, padding=50)

    if(os.path.exists(PACK_PATH+"/checkpoint/checker.index")):
        saver.restore(sess, PACK_PATH+"/checkpoint/checker")

        f = open(PACK_PATH+"/dataset/labels.txt", 'r')
        content = f.readlines()
        f.close()
        for idx in range(len(content)):
            content[idx] = content[idx][:len(content[idx])-1] # rid \n

        boxes_pred = []
        for b in boxes:
            x, y, w, h = b

            if((x > 0) and (y > 0)):
                if((x < resized.shape[1]) and (y < resized.shape[0])):
                    prob = sess.run(prediction, feed_dict={x:img_predict(image=resized[y:y+h, x:x+w]), training:False})
                    result = str(content[int(np.argmax(prob))])
                    acc = np.max(prob)

                    if(acc > 0.85):
                        boxes_pred.append([x, y, w, h, result, acc])

        boxes_pred = sorted(boxes, key=lambda l:l[4], reverse=True) # sort by result
        boxes_pred = sorted(boxes, key=lambda l:l[5], reverse=True) # sort by acc

        ratio = int(origin.shape[0] / resized.shape[0])

        for bp in boxes_pred:
            x, y, w, h = bp
            x, y, w, h = x * ratio, y * ratio, w * ratio, h * ratio

            if((x > 0) and (y > 0)):
                if((x < resized.shape[1]) and (y < resized.shape[0])):
                    cv2.rectangle(result, (x,y), (x+w,y+h), (0, 255, 0), 2)
                    cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
                    cv2.putText(frame, result+" "+str(int(acc*100))+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        while(True):
            cv2.imshow('Image',result)

            if(key == ord("q")):
                print("\n\nQUIT")
                break

        cv2.destroyAllWindows()
    else:
        print("You must training first!")
