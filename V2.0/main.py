import cv2
import time
from gestrue import gestrue_recognition
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
w0, h0 = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
circle_xy=[]  #存放掌心位置数据的列表
n=1
cap = cv2.VideoCapture(0) #开摄像头

gestrue = gestrue_recognition() 

if __name__ == "__main__":
    while(1):
        start = time.time()
        ret, frame = cap.read() #读取摄像头的内容
        
        frame = cv2.flip(frame, 2)
        roi = frame[y0:y0+h0,x0:x0+w0] #取手势所在框图并进行处理
        roi_original = roi.copy()
        
        roi_processed = gestrue.skin_detection_YCrCb_filtered(roi_original)  #利用YCrCb颜色模型进行图像分割
        roi_processed = gestrue.img_process(roi_processed)   #滤波处理
        
        zeros = np.ones((roi_processed.shape[0],roi_processed.shape[1]),dtype = np.uint8)
        zeros *=255
        roi_draw = cv2.merge([zeros,zeros,zeros])  #创建空白图像

        roi_draw,roi_processed,hand_x,hand_y,hand_r= gestrue.area_filter(roi_processed,roi_draw)  #面积滤波
        finger_list,roi_draw = gestrue.find_fingers(roi_processed,roi_draw,hand_x,hand_y,hand_r)



        
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
        elif key == ord('v'):  #保存当前帧
            cv2.imwrite(str(time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime()))+".jpg",roi_processed)
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

        cv2.imshow('roi_origin',roi_original)
        cv2.imshow('roi_processed',roi_processed)
        cv2.imshow('roi_draw',roi_draw)


    cap.release()
    cv2.destroyAllWindows() #关闭所有窗口