import cv2
import numpy as np 

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img = cv2.imread(r'C:\Users\Owen\Pictures\skin_test.jpg')
cv_show('img',img)
YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
skin = np.zeros(cr.shape, dtype = np.uint8)


res,cr = cv2.threshold(cr,130,255,cv2.THRESH_BINARY)
res,cb = cv2.threshold(cb,77,127,cv2.THRESH_BINARY)

skin = cv2.bitwise_and(cr,cb,dst=None,mask = None)
roi = cv2.bitwise_and(img,img, mask = skin)

cv_show('roi',roi)