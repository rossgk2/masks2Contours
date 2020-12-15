import os
from masks2contoursSA_manual import masks2contoursSA_manual

def main():
    # Set to 1 to create intermediate plots
    PLOT = 1

    # Orientation for viewing plots
    az = 214.5268
    el = -56.0884

    # Image directory
    fldr = input(
        "Please enter the absolute path of the folder in which images are stored. To use the default folder and skip "
        "this step, press ENTER.")
    if fldr.strip() == "":
        fldr = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\"

    # Number of frame to be segmented
    frameNum = 1

    # Create results directory
    resultsDir = fldr + "out"
    try:
        os.mkdir(resultsDir)
    except FileExistsError:
        pass

    #
    # SHORT AXIS
    #

    # Determine the names of the image and label files (.nii files).
    convention = input(
        "Enter the root word in the convention used to name image files and label files. Press ENTER to use the "
        "default.\n (The default image name is \"CINE_SAX.nii\" and the default label name is \"CINE_SAX_" + str(
            frameNum) + ".nii\"), so the default such root is \"CINE_SAX\".")

    # Create variables to hold these names.
    if convention.strip() == "":
        convention = "CINE_SAX"

    imgName = fldr + convention + ".nii"
    segName = fldr + convention + "_" + str(frameNum) + ".nii"

    # Run main function to get contour points from masks - output will be saved in mat files in the results directory
    # [update comment]
    masks2contoursSA_manual(segName, imgName, resultsDir, frameNum, PLOT)

# [Load results by returning from function]

# This function displays "prompt" to the user and waits for the user to enter input. As long as the user enters
# incorrect input, "errorMessage" and "prompt" are displayed.
#
# The argument "goodInput" must be a boolean function accepting one argument. The intention is that, given a string
# variable called input, goodInput(input) is True when input is valid and false otherwise.
def get_correct_input(goodInput, prompt, errorMessage):
    validInput = False
    while not validInput:
        returnThis = input(prompt)
        validInput = goodInput(returnThis)
        if not validInput:
            print(errorMessage)

    return returnThis

main()
