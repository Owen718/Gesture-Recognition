import cv2
import process_recognition
import time

font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
w0, h0 = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
circle_xy=[]  #存放掌心位置数据的列表
n=1
cap = cv2.VideoCapture(0) #开摄像头
 
if __name__ == "__main__":
    while(1):
        ret, frame = cap.read() #读取摄像头的内容
        frame = cv2.flip(frame, 2)
        start = time.time()
        roi = frame[y0:y0+h0,x0:x0+w0] #取手势所在框图并进行处理
        roi_original = roi.copy()
        roi = process_recognition.skin_detection_YCrCb_filtered(roi)  #皮肤检测与识别
        roi = process_recognition.img_process(roi)  #去噪、二值化
        #roi = process_recognition.morpy_porcess(roi)   
        circle_x,circle_y,radius = process_recognition.distance_transform(roi) #使用距离变换求手掌心位置
        circle_xy.append((circle_x,circle_y))
        if circle_y !=0 and abs(circle_y-circle_xy[n-1][1])/circle_y > 0.1:  #若y值数据变化幅度大于10%
            circle_y = (circle_y + circle_xy[n-1]) / 2  #取均值
        n+=1

        roi,roi_draw,keypoints = process_recognition.hands_contours(roi,roi_original,circle_x,circle_y,radius) #绘制轮廓并对特征点进行筛选
        #print(keypoints)
        roi_draw,estimate_roi= process_recognition.gesture_estimate(roi_draw,circle_x,circle_y,radius,keypoints)  #姿态估计
  
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
        cv2.imshow('estimate_roi',estimate_roi)
        cv2.imshow('roi_draw',roi_draw)


    cap.release()
    cv2.destroyAllWindows() #关闭所有窗口