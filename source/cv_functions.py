import numpy as np
import cv2

def load_image(path=""):

    return cv2.imread(path)

def save_image(path="", filename="", image=None):

    cv2.imwrite(path + filename, image)

def rgb2gray(rgb=None):

    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

def resizing(image=None, width=0, height=0):

    # set ratio
    if(width == 0):
        width = int(image.shape[0] / image.shape[1] * height)
    elif(height == 0):
        height = int(image.shape[1] / image.shape[0] * width)
    else:
        pass

    return cv2.resize(image, (width, height))

def bluring(gray=None, k_size=11):

    return cv2.GaussianBlur(gray, (k_size, k_size), 0)

def adaptiveThresholding(gray=None, neighbor=5, blur=False, k_size=3):

    if(blur):
        gray = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, neighbor, 1)

def erosion(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)

    return cv2.erode(binary_img, kernel, iterations=iterations)

def dilation(binary_img=None, k_size=5, iterations=1):

    kernel = np.ones((k_size, k_size),np.uint8)

    return cv2.dilate(binary_img, kernel, iterations=iterations)

def custom_opeing(binary_img=None, ero_size=5, dil_size=5, iterations=1):

    ero_kernel = np.ones((ero_size, ero_size),np.uint8)
    dil_kernel = np.ones((dil_size, dil_size),np.uint8)

    tmp_ero = cv2.erode(binary_img, ero_kernel, iterations=iterations)

    return cv2.dilate(tmp_ero, dil_kernel, iterations=iterations)

def custom_closing(binary_img=None, ero_size=5, dil_size=5, iterations=1):

    ero_kernel = np.ones((ero_size, ero_size),np.uint8)
    dil_kernel = np.ones((dil_size, dil_size),np.uint8)

    tmp_dil = cv2.dilate(binary_img, dil_kernel, iterations=iterations)

    return cv2.erode(tmp_dil, ero_kernel, iterations=iterations)

def opening(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations) # iteration = loop

def closing(binary_img=None, k_size=2, iterations=1):

    kernel = np.ones((k_size, k_size), np.uint8)

    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations) # iteration = loop

def contouring(binary_img=None):

    # return two values: contours, hierarchy
    # cv2.RETR_EXTERNAL

    return cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

def contour2box(contours=None, padding=10):

    boxes =[]
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if(area < 30):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = x-int(padding/2), y-int(padding/2), w+int(padding), h+int(padding)
        boxes.append([x, y, w, h])

    return boxes
