import cv2 as cv
import numpy as np

def chop_img(img, step):
    allImgs = []
    width, height, i, j = 640, 512, 0, 0
    for i in range(step):
        for j in range(step):
            # crop the image into step*rows and step*columns
            imgCrop = img[int(0 + height / step * i): int(height / step + height / step * i),
                      int(0 + width / step * j): int(width / step + width / step * j)]
            imgResize = cv.resize(imgCrop, (640, 512))  # Resize image
            imgAug = augment(imgResize)
            allImgs.append((imgResize, imgAug))
            j += 1
        i += 1
    return allImgs

def augment(img):
    kernel = np.ones((5, 5), np.uint8)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 0)  # apply gaussian blur
    imgThresh = cv.adaptiveThreshold(  # apply adaptive threshold
    imgBlur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 5)
    imgDilation = cv.dilate(imgThresh, kernel, iterations=1)  # apply dilation to amplify threshold result
    return imgDilation
    
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

def find_bats(img):
    batDepthMin, batDepthMax = 50, 400
    blobs = cv.findContours(img[1], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]
    cropped_bats = [crop_bat(img[0], np.int0(cv.boxPoints(cv.minAreaRect(blob)))) for blob in blobs if batDepthMin < cv.contourArea(blob) < batDepthMax]
    return cropped_bats

