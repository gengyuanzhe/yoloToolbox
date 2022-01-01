# coding:utf-8
from typing import List
import cv2 as cv
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plot

BG_HEIGHT = 400
BG_WIDTH = 450
MAX_SNAP_PER_IMG = 6

snaps = []
for filename in glob("snap/*png"):
    img = cv.imread(filename)
    category = os.path.basename(filename).split('_')[0]
    height = img.shape[0]
    width = img.shape[1]
    snaps.append([int(category), height, width, img])


def isRectangleOverlap(rec1: List[int], rec2: List[int]) -> bool:
    if rec1[0] == rec1[2] or rec1[1] == rec1[3] or rec2[0] == rec2[2] or rec2[1] == rec2[3]:
        return False
    return not (rec1[2] <= rec2[0] or  # left
                rec1[3] <= rec2[1] or  # bottom
                rec1[0] >= rec2[2] or  # right
                rec1[1] >= rec2[3])  # top


def isValidRect(rect, rectsUsed):
    for rectUsed in rectsUsed:
        if isRectangleOverlap(rect, rectUsed):
            return False
    return True


def getLabel(category, rect, height, width):
    x1, y1, x2, y2 = rect
    return category, (y1 + y2) / 2 / width, (x1 + x2) / 2 / height, (y2 - y1) / width, (x2 - x1) / height


def generateImg():
    bgImg = np.zeros((BG_HEIGHT, BG_WIDTH, 3), np.uint8)
    # 使用白色填充图片区域,默认为黑色
    bgImg.fill(0)

    cnt = np.random.randint(1, MAX_SNAP_PER_IMG)

    rectsUsed = []
    labels = []
    for i in range(cnt):
        ind = np.random.randint(len(snaps))
        (category, height, width, img) = snaps[ind]
        while True:
            x1 = np.random.randint(0, BG_HEIGHT - height)
            y1 = np.random.randint(0, BG_WIDTH - width)
            x2 = x1 + height
            y2 = y1 + width
            rect = (x1, y1, x2, y2)
            if isValidRect(rect, rectsUsed):
                # print(rect)
                rectsUsed.append(rect)
                labels.append(getLabel(category, rect, BG_HEIGHT, BG_WIDTH))
                bgImg[x1:x2, y1:y2, :] = img

                break
    return bgImg, labels


if __name__ == '__main__':
    classes = set()
    if not os.path.exists("images"):
        os.mkdir("images")
    for i in range(10):
        print("i=%d" % i)
        img, labels = generateImg()
        # plot.imshow(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))
        # plot.show()
        cv.imwrite("images/%06d.png" % i, img)

        with open("images/%06d.txt" % i, "w") as f:
            for label in labels:
                f.write("%d %.6f %.6f %.6f %.6f\n" % label)
                classes.add(label[0])

    with open("images/classes.txt", "w") as f:
        for id in sorted(classes):
            f.write("%d\n" % id)
