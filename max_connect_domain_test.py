import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
w0, h0 = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
n=1
cap = cv2.VideoCapture(0) #开摄像头
 

def skin_detection_YCrCb_filtered(roi):#基于改进后的HCrCb肤色检测模型
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    HSV = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)  #转换至HSV空间

    (h,s,v) = cv2.split(HSV)   #拆分出H S V通道 
    (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    skin = np.zeros(cr.shape, dtype = np.uint8)
 
    #h=cv2.equalizeHist(h)
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





def img_process(roi):   #图像去噪
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(roi,(3,3),None)
    #ret,binary = cv2.threshold(roi,55,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    openning = cv2.morphologyEx(roi,cv2.MORPH_OPEN,kernel)
    ret,openning = cv2.threshold(openning,55,255,cv2.THRESH_BINARY)
    return openning

def connective_domain(roi):  #腐蚀-膨胀 求轮廓
    kernel = np.ones((3,3),np.uint8)

    #spin_roi = spin_roi * 255
 
    erode = cv2.erode(roi,kernel,iterations=3)
    dialet = cv2.dilate(roi,kernel,iterations=3)
    spin_roi = erode - dialet  
    spin_roi[spin_roi==0]=254
    spin_roi[spin_roi==255]=0


    return  spin_roi 

if __name__ == "__main__":
    while(1):
        ret, frame = cap.read() #读取摄像头的内容
        frame = cv2.flip(frame, 2)
        start = time.time()
        roi = frame[y0:y0+h0,x0:x0+w0] #取手势所在框图并进行处理
        roi_original = roi.copy()
        roi = skin_detection_YCrCb_filtered(roi)  #皮肤检测与识别
        roi = img_process(roi)  #去噪、二值化
        spin_roi = connective_domain(roi)

        cv2.imshow('binary',roi)
        #circle_x,circle_y,radius = process_recognition.distance_transform(roi) #使用距离变换求手掌心位置
      
      
        key = cv2.waitKey(1) & 0xFF#按键判断并进行一定的调整
        #按'a''d''w''s'分别将选框左移，右移
        # ，上移，下移
        #按'q'键退出录像
        if key == ord('s') and y0 < frame.shape[0]-10:
            y0 += 10
        elif key == ord('w') and y0 > 10:
            y0 -= 10

        elif key == ord('d') and x0 < frame.shape[0]-10:
            x0 += 10
        elif key == ord('a') and x0 > 10:
            x0 -= 10
        if key == ord('q'):
            break

        end = time.time()
        seconds = end-start
        #print( "Time taken : {0} seconds".format(seconds))
        fps = 1 / seconds
        #print( "Estimated frames per second : {0}".format(fps))
        cv2.putText(frame, "FPS: {0}".format(float('%.1f'%fps)),(int(frame.shape[0]/10),int(frame.shape[1]/10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    1)
        cv2.imshow('frame', frame)#播放摄像头的内容
        #cv2.imshow('roi_draw',roi_draw)
        cv2.imshow('roi_original',roi_original)
        cv2.imshow('spin_roi',spin_roi)


    cap.release()
    cv2.destroyAllWindows() #关闭所有窗口