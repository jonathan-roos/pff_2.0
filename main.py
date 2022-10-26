from pffClasses import ThermalImg, BatDetector, Bat, Learner


userInput = input("Enter Img Path: ")
print("Finding bats...")
thermalImg = ThermalImg(userInput)
thermalImg.chopImg()
thermalImg.augImg()
thermalImg.findBats(batDepthMin = 10, contours = True)
learner = Learner('model.pkl')
batDetector = BatDetector(learner.learn, thermalImg.bats)
batDetector.calculateProbs()
batDetector.filterBatCount('0.9', '0.75')
print(batDetector.getCount(), batDetector.getFilteredCount())

