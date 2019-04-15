import cv2
import numpy as np

# Harris角点
def HarrisDetect(img):
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 图像转换为float32
    gray = np.float32(gray)

    # harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.02)

    # 图像膨胀
    # dst = cv2.dilate(dst, None)

    # 图像腐蚀
    # dst = cv2.erode(dst, None)

    # 阈值设定
    img[dst > 0.0001*dst.max()] = [225, 0, 0]
    cv2.imshow('Harris', img)

    return img

# 求顶点
def pointDetect(Harris_img, img):
    # 求图像大小
    shape = img.shape
    height = shape[0]
    width = shape[1]

    upLeftX = 0
    upLeftY = 0
    downLeftX = 0
    downLeftY = 0
    upRightX = 0
    upRightY = 0
    downRightX = 0
    downRightY = 0

    # 求左上顶点
    for i in range(0, round(width/2)):
        for j in range(0, round(height/2)):
            if upLeftX == 0 and upLeftY == 0 and Harris_img[j][i][0] == 225:
                upLeftX = i
                upLeftY = j
                break
        if upLeftX or upLeftY:
            break

    # 求右上顶点
    for i in range(width-1, (round(width * (3/4))), -1):
        for j in range(0, round(height/6)):
            if upRightX == 0 and upRightY == 0 and Harris_img[j][i][0] == 225:
                upRightX = i
                upRightY = j
                break
        if upRightX or upRightY:
            break

    # 求左下顶点
    for j in range(height-1, round(height/2), -1):
        for i in range(0, round(width/2)):
            if downLeftX == 0 and downLeftY == 0 and Harris_img[j][i][0] == 225:
                downLeftX = i
                downLeftY = j
                break
        if downLeftX or downLeftY:
            break

    # 求右下顶点
    for i in range(width-1, round(width/2), -1):
        for j in range(round(height/2), height-1,):
            if downRightX == 0 and downRightY == 0 and Harris_img[j][i][0] == 225:
                downRightX = i
                downRightY = j
                break
        if downRightY or downRightY:
            break

    img[upLeftY][upLeftX][0] = 0
    img[upLeftY][upLeftX][1] = 255
    img[upLeftY][upLeftX][2] = 0

    print("左上坐标：", upLeftY, upLeftX)

    img[upRightY][upRightX][0] = 0
    img[upRightY][upRightX][1] = 255
    img[upRightY][upRightX][2] = 0

    print("右上坐标：", upRightY, upRightX)

    img[downRightY][downRightX][0] = 0
    img[downRightY][downRightX][1] = 255
    img[downRightY][downRightX][2] = 0

    print("右下坐标：", downRightY, downRightX)

    img[downLeftY][downLeftX][0] = 0
    img[downLeftY][downLeftX][1] = 255
    img[downLeftY][downLeftX][2] = 0

    print("左下坐标：", downLeftY, downLeftX)

    # 图像膨胀
    img = cv2.dilate(img, None)

    # 描边
    cv2.line(img, (upLeftX, upLeftY), (upRightX, upRightY), (255, 0, 0), 1)
    cv2.line(img, (upRightX, upRightY), (downRightX, downRightY), (255, 0, 0), 1)
    cv2.line(img, (downRightX, downRightY), (downLeftX, downLeftY), (255, 0, 0), 1)
    cv2.line(img, (downLeftX, downLeftY), (upLeftX, upLeftY), (255, 0, 0), 1)

    cv2.imshow('result', img)

# 利用绝对中位差排除异常Theta
def getMAD(s):
    median = np.median(s)
    # 这里的b为波动范围
    b = 1.4826
    mad = b * np.median(np.abs(s-median))

    # 确定一个值，用来排除异常值范围
    lower_limit = median - (3*mad)
    upper_limit = median + (3*mad)

    # print(mad, lower_limit, upper_limit)
    return lower_limit, upper_limit

# 通过霍夫变换计算角度
def CalcDegree(srcImage):
    midImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    dstImage = cv2.Canny(midImage, 50, 200, 3)
    lineimage = srcImage.copy()

    # 通过霍夫变换检测直线
    # 第4个参数为阈值，阈值越大，检测精度越高
    lines = cv2.HoughLines(dstImage, 1, np.pi / 180, 200)
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    sum = 0

    # 绝对中位差排除异常值
    thetaList = []
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            thetaList.append(theta)
    #print(thetaList)

    lower_limit, upper_limit = getMAD(thetaList)

    # 判断是否需要旋转操作
    thetaavg_List = []
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            if lower_limit <= theta <= upper_limit:
                thetaavg_List.append(theta)
    thetaAvg = np.mean(thetaavg_List)
    #print(thetaAvg)

    deviation = 0.01
    if (np.pi/2-deviation <= thetaAvg <= np.pi/2+deviation) or (0 <= thetaAvg <= deviation) or (np.pi-deviation <= thetaAvg <= 180):
        angle = 0
    else:
        # 依次画出每条线段
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                if lower_limit <= theta <= upper_limit:
                    #print("theta:", theta, " rho:", rho)
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(round(x0 + 1000 * (-b)))
                    y1 = int(round(y0 + 1000 * a))
                    x2 = int(round(x0 - 1000 * (-b)))
                    y2 = int(round(y0 - 1000 * a))
                    # 只选角度最小的作为旋转角度
                    sum += theta
                    cv2.line(lineimage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Imagelines", lineimage)



        # 对所有角度求平均，这样做旋转效果会更好
        average = sum / len(lines)
        res = DegreeTrans(average)
        if res > 45:
            angle = 90 + res
        elif res < 45:
            print(2)
            angle = -90 + res

    return angle

# 度数转换
def DegreeTrans(theta):
    res = theta / np.pi * 180
    print(res)
    return res

# 逆时针旋转图像degree角度（原尺寸）
def rotateImage(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # degree = 180 - degree
    if 65 < abs(degree) < 90:
        print(11)
        RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree * (4/5), 1)
    elif 0 < abs(degree) < 65:
        print(abs(degree))
        RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree/3, 1)

    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(50, 46, 65))
    return rotate

# 利用掩模排除多余的点
def getMask(img, Harris_img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)

    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel)

    mask_inv = cv2.bitwise_not(mask)


    img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    img2_fg = cv2.bitwise_and(Harris_img, Harris_img, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)

    cv2.imshow('mask_result', dst)

    return dst

# 导入图像
filename = r'.\data\test_images\5.jpeg'
# filename = 'ex0302.jpg'
# filename = 'ex0303.bmp'
img = cv2.imread(filename)
imgCopy = img.copy()

# 调整图像角度
# degree = CalcDegree(img)        # 求矩形主要方向与x轴的夹角degree
# if degree != 0:                 # 若夹角不为0，则图像需要旋转
#     rotate = rotateImage(img, degree)
#     imgCopy = rotate.copy()
# else:                           # 夹角很小时，可以不旋转
#     rotate = img.copy()
rotate = img.copy()
# Harris角点检测
Harris_img = HarrisDetect(rotate)

# 求掩模，提出距离票据太远的点
mask = getMask(imgCopy, Harris_img)

# 求四个角点，标出
pointDetect(mask, imgCopy)


cv2.waitKey(0)
cv2.destroyAllWindows()
