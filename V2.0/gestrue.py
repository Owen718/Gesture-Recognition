import cv2
import numpy as np
class gestrue_recognition:    
    def quasi_Euclidean_distance(self,point1,point2):  #准欧式距离 point1=(i,j) point2=(h,k)  已对运算速度优化
        if abs(point1[0]-point2[0]) > abs(point1[1]-point2[1]) :   #
            #return abs(point1[0]-point2[0]) + (2**0.5 - 1) * abs(point1[1]-point2[1])
            return int(abs(point1[0]-point2[0]) + 0.4142* abs(point1[1]-point2[1]))
        else:
            return int(0.4142 * abs(point1[0]-point2[0]) + abs(point1[1]-point2[1]))        

    def skin_detection_YCrCb_filtered(self,roi):#基于HCrCb混合肤色模型H,Cr,Cb范围筛选法
        YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
        hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)  #转换至HSV色彩空间
        (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
        (h,s,v) = cv2.split(hsv)
        skin = np.zeros(cr.shape, dtype = np.uint8)
        h[h>31]=0  #筛出h值为1-31的区域
        res,cr = cv2.threshold(cr,133,173,cv2.THRESH_BINARY)  #筛出130-175的值
        res,cb = cv2.threshold(cb,77,127,cv2.THRESH_BINARY)   #筛出77-127的值
        skin = cv2.bitwise_and(cr,cb,dst=None,mask = None)  #cr cb运算
        #skin = cv2.bitwise_and(skin,h,dst = None,mask = None)  #和h掩膜与运算

        roi = cv2.bitwise_and(roi,roi, mask = skin)  #与运算
        
        return roi

    def img_process(self,roi):  #图像预处理
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(roi,(3,3),None)
        ret,binary = cv2.threshold(roi,55,255,cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        binary = cv2.dilate(binary,kernel)

        return binary

    def distance_transform(self,roi):  #距离变换求手心坐标
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
    


    def area_filter(self,roi,roi_draw):  #利用面积去消除错误的分割区域
        h,hierarchy = cv2.findContours(roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #寻找轮廓
        hands_keypoints=[]
        #cv2.drawContours(roi_draw,h,-1,(0,0,0),2)  #绘制近似前的轮廓

        for con in h:  #计算各个轮廓的面积，并在轮廓中心标出
            if len(h)!=1:
                M = cv2.moments(con)  # 计算第一条轮廓的各阶矩,字典形式
                if M["m00"]!=0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                if(cv2.contourArea(con)>3000):  #轮廓大小如果大于3000
                    cv2.drawContours(roi_draw,np.array([con]),-1,(0,0,0),2)
                    #cv2.putText(roi_draw,str(cv2.contourArea(con)),(center_x,center_y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
                else:
                    cv2.drawContours(roi,np.array([con]),-1,(0),-1)

        x,y,r = self.distance_transform(roi)
        cv2.circle(roi_draw,(int(x),int(y)),radius = 2,color = (0,0,0),thickness = 2)
        return roi_draw,roi,int(x),int(y),int(r)


    def find_fingers(self,roi,roi_draw,x,y,r): #roi图像，手心坐标x，手心坐标y，手部半径r
        finger = [] #定义一个列表用来存放手指的中点，手指的宽度和坐标  finger.append(np.array[finger_center,finger_width,finger_y])
        
        for i,line in enumerate(roi):
            #print(num)
            #print(line)
            blank = np.where(line == 255)  #寻找值为255的点，存入blank
            blank = np.array(blank)      #将blank转化为数组
            if(blank.size):  #判断空格
                for j,line_x in enumerate(blank.reshape(blank.shape[1],1)):
                    #cv2.circle(hand_palm,(line_x,i),2,(0,0,0),-1)
                    if(line[line_x-1]==0 and line[line_x-2]==0 and i != 0 ):  
                        # 门限： and (line_x < hand_palm_xy[1]-int(hand_palm_R) or line_x > hand_palm_xy[1]+int(hand_palm_R))
                        #cv2.circle(hand_palm,(line_x,i),1,(255,0,0),-1) #画手指边
                        finger_x = line_x
                        finger_len = 0
                        while(line[line_x]==255):  #遍历求手指宽度
                            finger_len+=1
                            line_x+=1
                    
                        if(finger_len > 5):
                            finger.append([int(line_x)-int(finger_len/2),finger_len,i])  #存入手指中点坐标，手指的宽度,y 坐标


        for i in range(len(finger)-1,-1,-1): #倒序遍历骨架点,去除手掌内的点
            if(self.quasi_Euclidean_distance([finger[i][0],finger[i][2]],[x,y]) < 1.2*r):     
                finger.remove(finger[i])

        for finger_key in finger: 
            cv2.circle(roi_draw,(finger_key[0],finger_key[2]),2,(0,0,0),-1)
        
        return finger,roi_draw

                            

            