
import cv2
import numpy as np
import sys
import time

color_list=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255]]

# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]



def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points



def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img

# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img



def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=100):
    global result 
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    result[seed_row][seed_col] = color_list[num%len(color_list)] #标红
    
    
    for offset in offsets:
        neighbor_row = min(max(0, seed_row+offset[0]), rows-1)
        neighbor_col = min(max(0, seed_col+offset[1]), cols-1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img

# max_num 表示连通域最多存在的个数
def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img



if __name__ == "__main__":
    sys.setrecursionlimit(1000000000)


    img = cv2.imread(r'2020-10-29-21_45_39.jpg',0)
    img = cv2.resize(img,dsize=None,fx=0.2,fy=0.2)
    

    zeros = np.zeros((img.shape[0],img.shape[1]),dtype = np.uint8)
    result = cv2.merge([zeros,zeros,zeros])
    
    time1 = time.time()

    # print("原始二值图像")
    # print(binary_img)

    # print("Two_Pass")
    # binary_img = Two_Pass(binary_img, NEIGHBOR_HOODS_8)
    # binary_img, points = reorganize(binary_img)
    # print(binary_img, points)

    print("Seed_Filling")
    binary_img = Seed_Filling(img, NEIGHBOR_HOODS_8)
    time2 = time.time()
    print(time2 - time1)
    


    result = cv2.resize(result,dsize = None,fx=5,fy=5)
    cv2.imshow("result",result)

    cv2.waitKey(0)
    
    #print(binary_img, points)
    #print(binary_img)
    #
    # print(points)