"""
    Example of how you could use PFF classes.
"""

from pffClasses import ThermalImg, BatDetector, Bat, Learner

# Prompt user for path to thermal img
userInput = input("Enter Img Path: ") 

# Print 'Finding Bats...' to indicate to the user that the script has started 
print("Finding bats...")

# Create instance of a ThermalImg object.
thermalImg = ThermalImg(userInput) # Takes the user input path as a parameter
thermalImg.chopImg() # Use the class method chopImg(), to chop the input into 9 segments
thermalImg.augImg() # Apply pre processing to each segment using class method augImg()
thermalImg.findBats() # Use class method finBats() to find contours in each segment

# Create instance of a Learner object.
learner = Learner('model.pkl') # Takes the name of the trained model file as a string

# Create instance of a BatDetector and pass the loaded learner 
# and the Bats found from using ThermalImg.findBats()
batDetector = BatDetector(learner.learn, thermalImg.bats) 
batDetector.calculateProbs() # Calculate the probs of each bat using batDetector.calculateProbs()

# param1 = confidence it is not bat. param2 = confidence it is bat
# if param1 is < 0.9 and param2 is > 0.75. Bat is added to the to filtered count
batDetector.filterBatCount('0.9', '0.75') # Filter the calculated probs using the 2 input parameters
print(f"Bat Count before nn = {batDetector.getCount()}\nFiltered Bat Count = {batDetector.getFilteredCount()}")
thermalImg.showContours(both=True) # Show the contours of all bats before and after the neural network
