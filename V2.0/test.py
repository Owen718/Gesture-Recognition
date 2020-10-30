import cv2
import numpy as np

i = 1
j = 1
num = 1

def np_index_value(np_array,x,y):  #越界返回0
    try:
        return np_array[x][y]
    except IndexError:
        return 0



def Seed_Filling(img,result): #种子填充算法 递归实现
    global i,j,num
    row,col = img.shape #row为行，col为列,j为行，i为列
    if(i==col): #一行结束,
        i=1
        j+=1

    if(j==row-1):  #临界
        return result

    if(img[j][i]==255):  #四邻域
        result[j][i]=num
        if(np_index_value(img,j-1,i)==255 and np_index_value(result,j-1,i)==0):
            result[j-1][i]=num
        if(np_index_value(img,j+1,i)==255 and np_index_value(result,j+1,i)==0):
            result[j+1][i]=num
        if(np_index_value(img,j,i+1)==255 and np_index_value(result,j,i+1)==0):
            result[j][i+1]=num
        if(np_index_value(img,j,i-1)==255 and np_index_value(result,j,i-1)==0):
            result[j][i-1]=num  
        num+=1
    i+=1
    
    return Seed_Filling(img,result)







img = cv2.imread(r'2020-10-29-21_45_04.jpg')

zeros = np.zeros((img.shape[0],img.shape[1]),dtype = np.uint8)

img_draw = cv2.merge([zeros,zeros,zeros])

ones = np.array([
    [0,255,255,255,0],
    [255,255,255,255,0],
    [255,255,255,255,0],
    [255,0,255,255,0]
])

result = np.zeros((ones.shape[0],ones.shape[1]),dtype = np.uint8)

result  = Seed_Filling(ones,result)

print(result)
#cv_show('1',img)
#cv_show('draw',img_draw)

