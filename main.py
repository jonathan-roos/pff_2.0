import cv2 as cv
from functions import chop_img, find_bats
from fastai.vision.all import load_learner
from fastai.basics import default_device
import pathlib
import time

userInput = input("Img Path: ")
img = cv.imread(userInput)
start = time.time()
totalBats = 0

# list of tuples that store each cropped image in its original format and threshed format (original, threshed)
allImgs = chop_img(img, 3)

cropped_bats = []
for img in allImgs:
    cropped_bats.extend(find_bats(img))

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner("model_densenet_zeros.pkl", cpu=True) # Change to cpu=False if using a GPU
print(f"Time before Detecting bats took {time.time() - start:.2f} seconds")
print("Detecting Bats...")

start1 = time.time()


def predict(bat):
    with learn.no_bar(), learn.no_logging():
        _, _, probs = learn.predict(bat)
    return (tuple(map(lambda x: f"{x:.4f}", probs)))

results = [predict(bat) for bat in cropped_bats]

filtered_results = filter(lambda x: (x[0] < '0.5' and x[1] > '0.75'), results)

print(len(list(filtered_results)))
print(f"Detecting Bats took: {time.time()-start1:.2f} seconds")

