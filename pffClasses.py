# Project Flying Fox - mainClass.py
# Object Oriented Approach to main.py

import cv2 as cv
import numpy as np
from fastai.vision.all import load_learner
import pathlib
import time

class ThermalImg:
    """ 
        This is a class that will create an instance of a thermal img object. 
        Class properties are width and height
    """
    width, height = 640,512 # Size of thermal Imgs that were provided from Ecosure. 


    def __init__(self, imgPath):
        """
            Used to initialise ThermalImg object.
            Takes the path of a thermal image as a string.
        """
        self.imgPath = imgPath
        self.img = cv.imread(imgPath) # Returns image loaded from the specified file path
        self.batCount = 0 # Initialise a bat count of zero
        self.allImgs = [] # Initialise empty list to store each 9 image segments
        self.allImgsAug = [] # Initialise empty list to store each 9 augmented image segments
        self.bats = [] # Initialise emtpty list to store a list of Bat objects for each image segment


    def getBats(self):
        """
            Will take self.bats which is a list of 9 lists, 
            containing Bat objects that correspond to each of the 9 segments from the input image.
            Returns a single list of all Bat objects.
        """
        bats = []
        for segment in self.bats:
            bats.extend(segment)
        return bats


    def getPath(self):
        """Return input image path"""
        return self.imgPath
    

    def getAllImgs(self):
        """Return list of segmented images"""
        if len(self.allImgs) == 0:
            print("allImgs is empty")
        else:
            return self.allImgs
    

    def getAllImgsAug(self):
        """Return list of all augmented images"""
        if len(self.allImgsAug) == 0:
            print("allImgsAug is empty")
        else:
            return self.allImgsAug
    

    def showImg(self):
        """
            Show original thermal image with a waitKey of 0.
            Press any key to close img window.    
        """
        cv.imshow(f"{self.imgPath}", self.img)
        cv.waitKey(0)


    def chopImg(self):
        """
            Will segment the original thermal image into 9 equal sized segments,
            and resize them to the width and height defined by the class properties.
            Segmented images are appended to self.allImgs.
        """
        i, j, step = 0, 0, 3 # step is used to determine number of segments. e.g. 3 steps will return 9 segments, 4 will return 16 segments ect.
        for i in range(step):
            for j in range(step):
                # crop the image into step*rows and step*columns
                imgCrop = self.img[int(0 + self.height / step * i): int(self.height / step + self.height / step * i),
                        int(0 + self.width / step * j): int(self.width / step + self.width / step * j)]
                imgResize = cv.resize(imgCrop, (self.width, self.height))  # Resize image 
                self.allImgs.append((imgResize))
                j += 1
            i += 1


    def augImg(self, dilate = True):
        """
            Will apply augmentation to each img segmentation found in self.allImgs.
            Dilation is set to a default of True
        """
        for img in self.allImgs:
            kernel = np.ones((5, 5), np.uint8) # create kernel used for cv.dilate
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
            imgBlur = cv.GaussianBlur(imgGray, (5, 5), 0)  # apply gaussian blur
            imgThresh = cv.adaptiveThreshold(  # apply adaptive threshold
            imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
            if dilate:
                imgDilation = cv.dilate(imgThresh, kernel, iterations=1)  # apply dilation to amplify threshold result
                self.allImgsAug.append(imgDilation)
            else:
                imgDilation = cv.dilate(imgThresh, kernel, iterations=0)  # apply dilation to amplify threshold result
                self.allImgsAug.append(imgDilation)


    def __str__(self):
        """
            String that will be print to console when printing a ThermalImg instance.
            e.g. print(ThermalImg())
        """
        return f"The Img Path is: '{self.imgPath}'\nThere are {len(self.bats)} potential bats found."
    

    def findBats(self, batDepthMin = 50, batDepthMax = 400):
        """
            Takes batDepthMin and batDepthMax as parameters.
            Default is min = 50 and max = 400
        """
        imgs = zip(self.allImgs, self.allImgsAug)
        for img in imgs:
            blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2] # Find contours in each img in allImgsAug
            # For each bat in blobs, if the contour is > batDepthMin and < batDepthMax, then crop the bat from the original img and append to self.bats
            croppedBats = [crop_bat(img[0], np.int0(cv.boxPoints(cv.minAreaRect(blob)))) for blob in blobs if batDepthMin < cv.contourArea(blob) < batDepthMax]
            # crop_bat() will return a Bat object which holds the cropped img, and the contour coordinates for the cropped bat
            self.bats.append(croppedBats)
    

    def showContours(self, both = False):
        """
            Will use self.bats to draw contours onto copy of allImgs.
            Will join segments of allImgs and show concatinated image.
            Set both = True to see the unfiltered contours
            Must be called AFTER calculateProbs() have been called
        """
        allImgs = [] # Initialise empty list to store copy of images that the contours will be drawn onto
        for i in range(len(self.allImgs)):
            img = self.allImgs[i].copy() # create a copy of each img segment in self.allImgs
            for bat in self.bats[i]:
                # bat.probs[0] = confidence img IS NOT a bat. bat.probs[1] = confidence img IS a bat
                if bat.probs[0] < '0.9' and bat.probs[1] > '0.75': 
                    # if bat probs fall within the threshold, then the contour is drawn onto the copy of the original img
                    cv.drawContours(img, [np.int0(cv.boxPoints(cv.minAreaRect(bat.box)))], 0, (0,0,255),1)
            allImgs.append(img)
        img_row_1 = cv.hconcat([allImgs[0],allImgs[1],allImgs[2]]) # concatonate img 1-3 into row 1
        img_row_2 = cv.hconcat([allImgs[3],allImgs[4],allImgs[5]]) # concatonate img 4-6 into row 2
        img_row_3 = cv.hconcat([allImgs[6],allImgs[7],allImgs[8]]) # concatonate img 7-9 into row 3
        img_concat = cv.resize(cv.vconcat([img_row_1, img_row_2, img_row_3]), (960, 768)) # concatonate row 1-3 into full img with contours drawn
        cv.imshow("contours", img_concat) # show
        cv.waitKey(0)
        if both: # repeats steps above without checking the Bat.probs
            allImgsBefore = []
            for i in range(len(self.allImgs)):
                imgBefore = self.allImgs[i].copy()
                for bat in self.bats[i]:
                    cv.drawContours(imgBefore, [np.int0(cv.boxPoints(cv.minAreaRect(bat.box)))], 0, (0,0,255),1)
                allImgsBefore.append(imgBefore)
            img_row_before_1 = cv.hconcat([allImgsBefore[0],allImgsBefore[1],allImgsBefore[2]])
            img_row_before_2 = cv.hconcat([allImgsBefore[3],allImgsBefore[4],allImgsBefore[5]])
            img_row_before_3 = cv.hconcat([allImgsBefore[6],allImgsBefore[7],allImgsBefore[8]])
            img_concat_before = cv.resize(cv.vconcat([img_row_before_1, img_row_before_2, img_row_before_3]), (960, 768))
            cv.imshow("contours before", img_concat_before)
            cv.waitKey(0)
    

def crop_bat(img, box):
    """takes original image and the boxpoints of the contour to be cropped from img"""

    # Assigns the contour box points to human readable variables
    x1, y1, x2, y2, x3, y3, x4, y4 = int(box[0][1]), int(box[0][0]), int(box[1][1]), int(box[1][0]), int(box[2][1]), int(box[2][0]), int(box[3][1]), int(box[3][0])

    # Find distance from cx to top_left_x and cy to top_left_y to determine how many pixels the border around the cropped image should be
    top_left_x, top_left_y, bot_right_x, bot_right_y = min([x1,x2,x3,x4]), min([y1,y2,y3,y4]), max([x1,x2,x3,x4]), max([y1,y2,y3,y4])

    # create even borders
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
    
    bat_crop = img[crop_x1: crop_x2, crop_y1: crop_y2] # crop the contour from the original img
    bat = Bat(bat_crop, box) # initialise a Bat object and assign it the cropped img and the contour box points
    return bat #  return initialised Bat


class BatDetector:
    """ 
        This is a class that will create an instance of a BatDetector object. 
    """
    def __init__(self, learner, bats):
        """
            Initialiser that takes name of a trained model .pkl file as a string for param1, 
            Param2 is a list of lists containing Bat objects. Each list in bats represents one segment of the original image.
        """
        self.learner = learner # path of a trained model. is a string. must be a .pkl file
        self.bats = bats
        self.results = [] # intialise a list of tuples that hold the probablities for each bat object
        self.filteredResults = [] # will a list of the filtered results
    
    def getCount(self):
        count = 0
        for segment in self.bats:
            for bat in segment:
                count += 1
        return count

    def getFilteredCount(self):
        return len(list(self.filteredResults))

    def calculateProbs(self):
        start = time.time() # create variable to time how long it takes to predict probs
        for segment in self.bats:
            for bat in segment:
                with self.learner.no_bar(), self.learner.no_logging():
                    _, _, probs = self.learner.predict(bat.img)
                result = tuple(map(lambda x: f"{x:.4f}", probs))
                self.results.append(result)
                bat.probs = result
                    
        print(f"It took {time.time()-start:.2f} seconds to calculate probs")
    
    def filterBatCount(self, notBatCon, isBatCon):
        """
            Takes confidence thresholds in decimal format as a string.
            param1 = confidence img is not a bat. param2 = confidence img is a bat
        """
        self.filteredResults = filter(lambda x: (x[0] < notBatCon and x[1] > isBatCon), self.results)
    
class Learner:
    """
        Class that is used to create instances of a learner.

        model = trained model saved as .pkl file, as a string 
        onWin = Boolean to determine if script is being run on Win or Linux. Default is onWin = True
        cpu = boolean to determine if script uses a GPU or not. Default is cpu = True
    """
    def __init__(self, model, onWin = True, cpu = True):
        if onWin:
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        self.learn = load_learner(model, cpu)

    def predictOne(self, bat):
        with self.learn.no_bar(), self.learn.no_logging():
                _, _, probs = self.learn.predict(bat)
        return probs

class Bat:
    """
        Class that describes an instance of a bat. 
        croppedImg = Takes the cropped bat image as its first parameter
        box = contour box points for img segment croppedImg was taken from 
    """

    def __init__(self, croppedImg, box):
        self.img = croppedImg
        self.box = box

    def getImg(self):
        return self.img





