# Project Flying Fox - mainClass.py
# Object Oriented Approach to main.py

import cv2 as cv
import numpy as np

class ThermalImg:
    """ This is a class that will create an instance of a thermal img  """
    width, height = 640,512

    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.img = cv.imread(imgPath)
        self.batCount = 0
        self.allImgs = []
        self.allImgsAug = []
        self.bats = []

    def getPath(self):
        return self.imgPath
    
    def getAllImgs(self):
        if len(self.allImgs) == 0:
            print("allImgs is empty")
        else:
            return self.allImgs
    
    def getAllImgsAug(self):
        if len(self.allImgsAug) == 0:
            print("allImgsAug is empty")
        else:
            return self.allImgsAug
    
    def showImg(self):
        cv.imshow(f"{self.imgPath}", self.img)
        cv.waitKey(0)

    def chopImg(self):
        i, j, step = 0, 0, 3
        for i in range(step):
            for j in range(step):
                # crop the image into step*rows and step*columns
                imgCrop = self.img[int(0 + self.height / step * i): int(self.height / step + self.height / step * i),
                        int(0 + self.width / step * j): int(self.width / step + self.width / step * j)]
                imgResize = cv.resize(imgCrop, (self.width, self.height))  # Resize image
                self.allImgs.append((imgResize))
                j += 1
            i += 1

    def augImg(self):
        for img in self.allImgs:
            kernel = np.ones((5, 5), np.uint8)
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
            imgBlur = cv.GaussianBlur(imgGray, (5, 5), 0)  # apply gaussian blur
            imgThresh = cv.adaptiveThreshold(  # apply adaptive threshold
            imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
            imgDilation = cv.dilate(imgThresh, kernel, iterations=1)  # apply dilation to amplify threshold result
            self.allImgsAug.append(imgDilation)

    def __str__(self):
        return f"The Img Path is: '{self.imgPath}'"
    
    def findBats(self):
        batDepthMin, batDepthMax = 50, 400
        imgs = zip(self.allImgs, self.allImgsAug)
        for img in imgs:
            blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]
            croppedBats = [crop_bat(img[0], np.int0(cv.boxPoints(cv.minAreaRect(blob)))) for blob in blobs if batDepthMin < cv.contourArea(blob) < batDepthMax]
            self.bats.extend(croppedBats)

def crop_bat(img, box):
    x1, y1, x2, y2, x3, y3, x4, y4 = int(box[0][1]), int(box[0][0]), int(box[1][1]), int(box[1][0]), int(box[2][1]), int(box[2][0]), int(box[3][1]), int(box[3][0])

    # Find distance from cx to top_left_x and cy to top_left_y to determine how many pixels the border around the cropped image should be
    top_left_x, top_left_y, bot_right_x, bot_right_y = min([x1,x2,x3,x4]), min([y1,y2,y3,y4]), max([x1,x2,x3,x4]), max([y1,y2,y3,y4])

    crop_x1 = top_left_x - 10 
    if crop_x1 <= 0:
        crop_x1 = 1 
    
    crop_x2 = bot_right_x+11
    if crop_x2 > 512:
        crop_x2 = 512 

    crop_y1 = top_left_y-10
    if crop_y1 <= 0:
        crop_y1 = 1

    crop_y2 = bot_right_y+11
    if crop_y2 > 640:
        crop_y2 = 640 
    
    bat_crop = img[crop_x1: crop_x2, crop_y1: crop_y2]

    return bat_crop

class Bat:
    pass

userInput = input("Enter Img Path: ")
thermalImg = ThermalImg(userInput)
thermalImg.chopImg()
thermalImg.augImg()
thermalImg.findBats()
for i in range(len(thermalImg.bats)):
    if i < 10:
        cv.imshow("bat i", thermalImg.bats[i])
        cv.waitKey(0)
