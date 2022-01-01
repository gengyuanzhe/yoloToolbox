# coding:utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot

from glob import glob
import os


def mat_flower(file_name):
    src = cv.imread(file_name, 1)
    src = cv.resize(src, (150, 150))
    tmp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    tmp = cv.medianBlur(tmp, 1)
    b, g, r = cv.split(src)
    height, width = tmp.shape  # 获取图片宽高
    _, alpha = cv.threshold(tmp, 60, 255, cv.THRESH_BINARY)
    for i in range(width // 2 - 15, width // 2 + 15):
        for j in range(height // 2 - 10, height // 2 + 10):
            alpha = cv.floodFill(alpha, mask=None, seedPoint=(i, j), newVal=(120, 120, 120),
                                 loDiff=(10, 10, 10), upDiff=(10, 10, 10))[1]

    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if alpha[i][j] == 120:
                alpha[i][j] = 255
            else:
                alpha[i][j] = 0

    # 根据轮廓去洞
    contour, hier = cv.findContours(alpha, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv.drawContours(alpha, [cnt], 0, 255, -1)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # alpha = cv.morphologyEx(alpha, cv.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv.morphologyEx(alpha, cv.MORPH_OPEN, kernel, iterations=1)

    # 这里不用alpha通道做merge，而是直接设置rgb，因为yolo训练和测试都是使用3通道的图片
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if alpha[i][j] == 0:
                b[i][j] = g[i][j] = r[i][j] = 0

    # plot.imshow(alpha, 'gray')
    # plot.show()

    resultImg = cv.merge([b, g, r], 3)
    resultImg = cutImage(resultImg, alpha)
    return resultImg


def cutImage(img, alpha):
    contours, _ = cv.findContours(alpha, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    x1, y1, x2, y2 = alpha.shape[0], alpha.shape[1], 0, 0
    for cnt in contours:
        # print(cnt)
        x1 = min(x1, np.min(cnt[:, :, 1]))
        y1 = min(y1, np.min(cnt[:, :, 0]))
        x2 = max(x2, np.max(cnt[:, :, 1]))
        y2 = max(y2, np.max(cnt[:, :, 0]))
        # print(x1, x2, y1, y2)
    return img[x1:x2, y1:y2, :]


if __name__ == '__main__':
    if not os.path.exists("snap"):
        os.mkdir("snap")

    for filename in glob("picture/*jpg"):
        print(f"process {filename}")
        dst = mat_flower(filename)
        tar_file = "./snap/" + os.path.splitext(os.path.basename(filename))[0] + '.png'
        cv.imwrite(tar_file, dst)

        # plot.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
        # plot.show()
