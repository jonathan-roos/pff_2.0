from pffClasses import ThermalImg, BatDetector, Bat, Learner


userInput = input("Enter Img Path: ")
print("Finding bats...")
thermalImg = ThermalImg(userInput)
thermalImg.chopImg()
thermalImg.augImg()
thermalImg.findBats()
learner = Learner('model.pkl')
batDetector = BatDetector(learner.learn, thermalImg.bats)
batDetector.calculateProbs()
batDetector.filterBatCount('0.9', '0.75')
print(f"Bat Count before nn = {batDetector.getCount()}\nFiltered Bat Count = {batDetector.getFilteredCount()}")
thermalImg.showContours(both=True)
