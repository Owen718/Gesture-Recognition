import cv2
import numpy as np

def img_process(roi):   #图像去噪
    blur = cv2.blur(roi,(3,3))
    blur = cv2.GaussianBlur(roi,(3,3),0)
    blur = cv2.medianBlur(roi,5)
    blur = cv2.bilateralFilter(roi,9,75,75)
    return roi

def skin_detection_YCrCb(roi):  #肤色检测，从图像中分割出皮肤区域
    YCrCb = cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)
    (y,cr,cb) = cv2.split(YCrCb)
    cr = cv2.GaussianBlur(cr,(5,5),0)
    ret,skin_mask  = cv2.threshold(cr,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = cv2.bitwise_and(roi,roi,mask = skin_mask)
    return res


def skin_detection_YCrCb_filtered(roi):#基于YCrCb颜色空间Cr,Cb范围筛选法

    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    skin = np.zeros(cr.shape, dtype = np.uint8)
    res,cr = cv2.threshold(cr,133,173,cv2.THRESH_BINARY)  #筛出130-175的值
    res,cb = cv2.threshold(cb,77,127,cv2.THRESH_BINARY)   #筛出77-127的值
    skin = cv2.bitwise_and(cr,cb,dst=None,mask = None)
    roi = cv2.bitwise_and(roi,roi, mask = skin)
    
    return roi

def morpy_porcess(roi):  #形态学处理，进行开运算
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(roi,kernel,iterations=1)
    dilation = erosion
    #dilation = cv2.dilate(erosion,kernel,iterations=2)
    #kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #dilation = cv2.morphologyEx(roi,cv2.MORPH_OPEN,kernel,iterations=2)
    return dilation

def hands_contours(roi):
    canny = cv2.Canny(roi,50,200)
    h = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = h[0]
    ret = np.ones(roi.shape,np.uint8)
    cv2.drawContours(ret,contours,-1,(255,255,255),1)
    return ret

