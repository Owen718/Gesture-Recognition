import cv2
import numpy as np

def img_process(roi):   #图像去噪
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY,)
    blur = cv2.GaussianBlur(roi,(15,15),None)
    ret,binary = cv2.threshold(blur,55,255,cv2.THRESH_BINARY)
    return binary

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
    #roi_show = np.zeros(cr.shape,dtype = np.uint8)
    #roi_show = 255
    res,cr = cv2.threshold(cr,133,173,cv2.THRESH_BINARY)  #筛出130-175的值
    res,cb = cv2.threshold(cb,77,127,cv2.THRESH_BINARY)   #筛出77-127的值
    skin = cv2.bitwise_and(cr,cb,dst=None,mask = None)  #与运算
    #ret,skin = cv2.threshold(skin,10,250,cv2.THRESH_BINARY)
    roi = cv2.bitwise_and(roi,roi, mask = skin)  #与运算
    #roi_show = cv2.bitwise_and(roi_show,skin,mask= None)

    return roi

def morpy_porcess(roi):  #形态学处理，进行开运算
    kernel = np.ones((3,3),np.uint8)
    #erosion = cv2.erode(roi,kernel,iterations=1)
    #dilation = erosion
    #dilation = cv2.dilate(erosion,kernel,iterations=2)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,kernel,iterations=1)
    return dilation

def hands_contours(roi,roi_original):
    canny = cv2.Canny(roi,50,200)  #边缘检测
    h,hierarchy= cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  #寻找轮廓
    #contours = h  #列表中的第一个
    ret = np.zeros(shape=[roi.shape[0],roi.shape[1],3],dtype = np.uint8)  #创建一个三通道的空白图像
  
    cv2.drawContours(ret,h,-1,(0,0,255),2)  #绘制近似前的轮廓（红）

    for con in h:
        if 70 < cv2.contourArea(con):
            hull = cv2.convexHull(con,hull=None,clockwise=None,returnPoints=True)
            #hull = cv2.convexHull(contour,hull=None,clockwise=None,returnPoints=True)
            for hull_point in hull:
                hull_point = (hull_point[0][0],hull_point[0][1])
                cv2.circle(roi_original,hull_point,5,(0,255,0),-1)
        

    return ret,roi_original

def distance_transform(roi):
    dist_img = cv2.distanceTransform(roi,cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
    return dist_img
# distance_type 计算距离的公式
#     参看 cv2.DIST_* ，常用cv2.DIST_L1
# mask_size 
#     参看 cv2.DIST_MASK_*
