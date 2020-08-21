# 将char类型的01的图转为int的二位list
def intarray(binstring):
    '''Change a 2D matrix of 01 chars into a list of lists of ints'''
    return [[1 if ch == '1' else 0 for ch in line]
            for line in binstring.strip().split()]


def toTxt(intmatrix):
    '''Change a 2d list of lists of 1/0 ints into lines of '#' and '.' chars'''
    return '\n'.join(''.join(('#' if p else '.') for p in row) for row in intmatrix)


# 定义像素周围的8领域
#   P9 P2 P3
#   P8 P1 P4
#   P7 P6 P5
def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x + 1, y - 1, x - 1, y + 1
    # print ((x,y))
    return [i[y1][x], i[y1][x1], i[y][x1], i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

# 计算领域中像素从0-1变化的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1  循环所有前景像素点，符合条件的像素点标记为删除
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P4 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P6 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing1.append((x, y))
        # 步骤一结束后删除changing1中所有标记的点
        for x, y in changing1: image[y][x] = 0
        # Step 2 重复遍历图片，标记所有需要删除的点
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and  # (Condition 0)
                        P2 * P6 * P8 == 0 and  # Condition 4
                        P2 * P4 * P8 == 0 and  # Condition 3
                        transitions(n) == 1 and  # Condition 2
                        2 <= sum(n) <= 6):  # Condition 1
                    changing2.append((x, y))
        # 步骤二结束后删除changing2中所有标记的点
        for x, y in changing2: image[y][x] = 0
        # 不断重复当changing1，changing2都为空的时候返回图像
        print("c1: ", changing1)
        print("c2", changing2)
    return image


rc01 = '''\
00000000000000000000000000000000000000000000000000000000000
01111111111111111100000000000000000001111111111111000000000
01111111111111111110000000000000001111111111111111000000000
01111111111111111111000000000000111111111111111111000000000
01111111100000111111100000000001111111111111111111000000000
00011111100000111111100000000011111110000000111111000000000
00011111100000111111100000000111111100000000000000000000000
00011111111111111111000000000111111100000000000000000000000
00011111111111111110000000000111111100000000000000000000000
00011111111111111111000000000111111100000000000000000000000
00011111100000111111100000000111111100000000000000000000000
00011111100000111111100000000111111100000000000000000000000
00011111100000111111100000000011111110000000111111000000000
01111111100000111111100000000001111111111111111111000000000
01111111100000111111101111110000111111111111111111011111100
01111111100000111111101111110000001111111111111111011111100
01111111100000111111101111110000000001111111111111011111100
00000000000000000000000000000000000000000000000000000000000\
'''



if __name__ == '__main__':
    print(rc01)
    image = intarray(rc01)
    print(image)
    print('\nFrom:\n%s' % toTxt(image))
    after = zhangSuen(image)
    print('\nTo thinned:\n%s' % toTxt(after))



