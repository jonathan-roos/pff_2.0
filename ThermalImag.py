# Project Flying Fox - mainClass.py
# Object Oriented Approach to main.py

import cv2 as cv
import numpy as np
from fastai.vision.all import load_learner
import pathlib
import time

class ThermalImg:
    """ This is a class that will create an instance of a thermal img object. Takes the path to thermal img as a string as param1 """
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
    bat = Bat(bat_crop, img, box)
    return bat

class BatDetector:
    """Class that takes name of a trained model .pkl file as a string for param1, an array of Bat objects as param2, 
        param3 is boolean to determine if OS is on Windows or Linux, default is true for Win. and boolean to determine if cpu is being used for param4(default is true)"""

    def __init__(self, model, bats, onWin = True, cpu = True):
        if onWin:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        self.learner = load_learner(model, cpu)
        self.bats = bats
        self.results = []
        self.filteredResults = []
    
    def getCount(self):
        return len(self.bats)

    def getFilteredCount(self):
        return len(list(self.filteredResults))

    def calculateProbs(self):
        start = time.time()
        for bat in self.bats:
            with self.learner.no_bar(), self.learner.no_logging():
                _, _, probs = self.learner.predict(bat.getImg())
                self.results.append(tuple(map(lambda x: f"{x:.4f}", probs)))
        print(f"It took {time.time()-start:.4f} seconds to calculate probs")
    
    def filterBatCount(self, notBatCon, isBatCon):
        """Takes confidence thresholds in decimal format as a string.
            param1 = confidence img is not a bat. param2 = confidence img is a bat"""
        self.filteredResults = filter(lambda x: (x[0] < notBatCon and x[1] > isBatCon), self.results)

    def showBatsBefore(self, allImgs):
        imgsBefore = allImgs.copy()
        for img in imgsBefore:
            for bat in self.bats:
                if bat.getOriginalImg() == img:
                    cv.drawContours(img, bat.getBox(), 0, (0, 0, 255), 1)
        return imgsBefore
    
    def showBatsAfter(self):

 

class Bat:
    """Class that describes an instance of a bat. Takes the cropped bat image as its first parameter"""

    def __init__(self, croppedImg, originalImg, box):
        self.img = croppedImg
        self.originalImg = originalImg
        self.box = box

    def getImg(self):
        return self.img

    def getOriginalImg(self):
        return self.originalImg

    def getBox(self):
        return self.box

def displayImg(allImgs):
    img_row_1 = cv.hconcat([allImgs[0],allImgs[1],allImgs[2]])
    img_row_2 = cv.hconcat([allImgs[3],allImgs[4],allImgs[5]])
    img_row_3 = cv.hconcat([allImgs[6],allImgs[7],allImgs[8]])
    img_concat = cv.resize(cv.vconcat([img_row_1, img_row_2, img_row_3]), (960, 768))
    cv.imshow("contours", img_concat)
    cv.waitKey(0)

userInput = input("Enter Img Path: ")
print("Finding bats...")
thermalImg = ThermalImg(userInput)
thermalImg.chopImg()
thermalImg.augImg()
thermalImg.findBats()
batDetector = BatDetector('model.pkl', thermalImg.bats)
batDetector.calculateProbs()
batDetector.filterBatCount('0.5', '0.75')
contourImgs = batDetector.showBatsBefore(thermalImg.allImgs)
displayImg(contourImgs)
print(batDetector.getCount(), batDetector.getFilteredCount())

