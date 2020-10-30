import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
w0, h0 = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
circle_xy=[]  #存放掌心位置数据的列表
n=1
cap = cv2.VideoCapture(0) #开摄像头
 

def skin_detection_YCrCb_filtered(roi):#基于改进后的HCrCb肤色检测模型
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    HSV = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)  #转换至HSV空间

    (h,s,v) = cv2.split(HSV)   #拆分出H S V通道 
    (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    skin = np.zeros(cr.shape, dtype = np.uint8)
 

    h[h>23]=0
    h[h<1]=0

    h_skin = cv2.bitwise_and(h,h,dst=None,mask = None)
    #cv2.cvtColor(hsv_roi,cv2.COLOR_HSV2BGR)
    res,cr = cv2.threshold(cr,135,170,cv2.THRESH_BINARY)  #筛出130-175的值
    res,cb = cv2.threshold(cb,94,125,cv2.THRESH_BINARY)   #筛出77-127的值
    skin = cv2.bitwise_and(cr,cb,dst=None,mask = None)  #与运算
   # skin = cv2.bitwise_and(skin,h,dst=None,mask = None)
    roi = cv2.bitwise_and(roi,roi,mask = skin)

    return roi

def img_process(roi):   #图像去噪,最大连通域算法
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(15,15),None)
    ret,binary = cv2.threshold(roi,55,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    binary = cv2.dilate(binary,kernel)

    return binary




def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img = cv2.imread(r'C:\Users\Owen\Pictures\face6.jpg')

roi = skin_detection_YCrCb_filtered(img)
roi = img_process(roi)

cv_show('img',roi)
    