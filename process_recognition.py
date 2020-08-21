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
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.morphologyEx(roi,cv2.MORPH_CLOSE,kernel,iterations=1)
    return dilation

def quasi_Euclidean_distance(point1,point2):  #准欧式距离 point1=(i,j) point2=(h,k)  已对运算速度优化
    if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]) :   #
        #return abs(point1[0]-point2[0]) + (2**0.5 - 1) * abs(point1[1]-point2[1])
        return int(abs(point1[0]-point2[0]) + 0.4142* abs(point1[1]-point2[1]))
    else:
        return int(0.4142 * abs(point1[0]-point2[0]) + abs(point1[1]-point2[1]))        



def hands_contours(roi,roi_original,x,y,radius):  #轮廓绘制与关键点筛选
    canny = cv2.Canny(roi,50,200)  #边缘检测
    h,hierarchy= cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  #寻找轮廓
    #ret = np.zeros(shape=[roi.shape[0],roi.shape[1],3],dtype = np.uint8)  #创建一个三通道的空白图像
    hands_keypoints = []
    cv2.drawContours(roi_original,h,-1,(0,0,255),2)  #绘制近似前的轮廓（红）
    cv2.circle(roi_original,(x,y),4,(255,0,0),3)  #标出手心(蓝色)
    for con in h:
        if 50 < cv2.contourArea(con):
            hull = cv2.convexHull(con,hull=None,clockwise=None,returnPoints=True)  #求凸包
            #defects = cv2.convexityDefects(con,hull)
            for i,hull_point in enumerate(hull):
                hull_point = (hull_point[0][0],hull_point[0][1])
                if quasi_Euclidean_distance(hull_point,(x,y)) > radius-5 and hull_point[1]<roi_original.shape[0]-10: #筛除一部分点
                    hands_keypoints.append(hull_point)

    for i,pointi in enumerate(hands_keypoints):  #根据区域去重
        for j,pointj in enumerate(hands_keypoints):
            if quasi_Euclidean_distance(pointi,pointj) < 40 and i != j:
                del(hands_keypoints[j])
                    #del(hands_keypoints[i+1])

                    #cv2.circle(roi_original,(x,y),radius=quasi_Euclidean_distance(hull_point,(x,y)),color=(255,255,255),thickness=1)
    return roi,roi_original,hands_keypoints

def distance_transform(roi):  #距离变换求手心坐标
    distance = cv2.distanceTransform(roi,cv2.DIST_L2, cv2.CV_32F)
    maxdist = np.max(distance)   #获取距离变换矩阵中的最大值作为内接圆半径
    circle_xy = np.where(distance == maxdist)  #获取内接圆圆心坐标
    if circle_xy[0].shape == 1 and circle_xy[1].shape ==1 :
        y = np.int(circle_xy[0])
        x = np.int(circle_xy[1])
    else:
        y = np.int(circle_xy[0][0])
        x = np.int(circle_xy[1][0])
    #cv2.circle(roi,(x,y),int(maxdist),(255,0,0),1,2,0)
    return x,y,maxdist

def gesture_estimate(roi,circle_x,circle_y,radius,hands_keypoints):  #姿态估计，2D建模
    estimate_roi = np.ones(roi.shape,dtype=np.uint8) #创建一幅白底图像
    estimate_roi = 255 * estimate_roi

    radius = int(radius)
    max_right_handspoint_x = 0  #最右端关键点坐标
    max_right_handspoint_y = 0
    max_left_handspoint_x = 9999  #最左端关键点坐标
    max_left_handspoint_y = 9999
    handpoints_average_length=0
    handpoints_all_length=0
    for i,point in enumerate(hands_keypoints):
        handpoints_all_length += quasi_Euclidean_distance(point,(circle_x,circle_y))
        if max_right_handspoint_x < point[0]:   #求最右端的手势关键点坐标
            max_right_handspoint_x = hands_keypoints[i][0]
            max_right_handspoint_y = hands_keypoints[i][1]
        if max_left_handspoint_x > point[0]:
            max_left_handspoint_x = hands_keypoints[i][0]
            max_left_handspoint_y = hands_keypoints[i][1]
    
 
        #cv2.line(roi,(0,0),(max_right_handspoint_x,0),color=(0,255,255),thickness=5)
        #handpoints_all_length
    for i,point in enumerate(hands_keypoints):  
        if point[1] > max_left_handspoint_y and point[0] < circle_x:
            del hands_keypoints[i]
  

    if max_right_handspoint_x:
        cv2.putText(roi,'right',(max_right_handspoint_x,max_right_handspoint_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0))
    if max_left_handspoint_x:
        cv2.putText(roi,'left',(max_left_handspoint_x,max_left_handspoint_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0))


    for i,point in enumerate(hands_keypoints):  #遍历指尖、划线
        cv2.circle(roi,point,3,(255,0,0),-1)
        if i < len(hands_keypoints)-1:
            cv2.line(roi,hands_keypoints[i],(circle_x,circle_y),(0,255,0),thickness=2)
            cv2.circle(estimate_roi,hands_keypoints[i],7,(0,0,255),thickness=-1)
            cv2.line(estimate_roi,hands_keypoints[i],(circle_x,circle_y),(0,0,0),5)
            cv2.putText(roi,str(i),hands_keypoints[i],cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0))


    return roi,estimate_roi




    '''
    for i,point in enumerate(hands_keypoints):  #去重
        if i < len(hands_keypoints)-1:
            if quasi_Euclidean_distance(hands_keypoints[i],hands_keypoints[i+1]) < 50 :
                del(hands_keypoints[i+1])
        #cv2.circle(roi,point,5,(0,255,0),-1)
    for point in hands_keypoints:
        cv2.circle(roi,point,5,(0,255,0),-1)
    print(hands_keypoints)
    '''
    
    return roi




