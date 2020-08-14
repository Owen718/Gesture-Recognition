import cv2
import process_recognition


font = cv2.FONT_HERSHEY_SIMPLEX #设置字体
size = 0.5 #设置大小
 
w0, h0 = 300, 300 #设置拍摄窗口大小
x0,y0 = 300, 100 #设置选取位置
 
cap = cv2.VideoCapture(0) #开摄像头
 
if __name__ == "__main__":
    while(1):
        ret, frame = cap.read() #读取摄像头的内容
        frame = cv2.flip(frame, 2)
        roi = frame[y0:y0+h0,x0:x0+w0] #取手势所在框图并进行处理
        roi = process_recognition.img_process(roi)
        roi = process_recognition.skin_detection_YCrCb_filtered(roi)
        roi = process_recognition.morpy_porcess(roi)
        roi = process_recognition.hands_contours(roi)

        key = cv2.waitKey(1) & 0xFF#按键判断并进行一定的调整
        #按'a''d''w''s'分别将选框左移，右移，上移，下移
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
        cv2.imshow('frame', frame)#播放摄像头的内容
        cv2.imshow('roi',roi)


    cap.release()
    cv2.destroyAllWindows() #关闭所有窗口