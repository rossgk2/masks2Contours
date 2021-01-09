import os
from typing import NamedTuple
from masks2Contours import masks2ContoursSA, masks2ContoursLA

def main():
    use_default_filepaths = True
    PLOT = True # Set to True to create intermediate plots
    LOAD_MATLAB_VARS = True

    # Orientation for viewing plots
    az = 214.5268
    el = -56.0884

    # Configure some settings.
    config = Config(rvWallThickness = 3, downsample = 3, upperBdNumContourPts = 200, PLOT = True, LOAD_MATLAB_VARS = True)
    frameNum = 1 # Index of frame to be segmented.

    # Get filepaths ready.
    fldr = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\"
    resultsDir = fldr + "out\\"

    # These are filepaths for the SA.
    imgName = fldr + "CINE_SAX.nii"
    segName = fldr + "CINE_SAX_" + str(frameNum) + ".nii"

    # These are filepaths for the LA.
    LA_segs = [fldr + "LAX_" + str(i) + "ch_1.nii" for i in range(2, 4 + 1)]
    LA_names = [fldr + "RReg_LAX_" + str(i) + "ch_to_Aligned_SA.nii" for i in range(2, 4 + 1)]

    # The following function produces contour points from masks for the short axis image files, displays them
    # if PLOT == True, and returns (endoLVContours, epiLVContours, endoRVFWContours, epiRVFWContours, RVSContours, RVInserts, RVInsertsWeights).
    # Each variable in this tuple, except for the last two, is a m x 2 ndarray for some m.

    masks2ContoursSA(segName, imgName, resultsDir, frameNum, config)

    masks2ContoursLA(LA_names, LA_segs, resultsDir, frameNum, config)

class Config(NamedTuple):
    rvWallThickness: int # RV wall thickness, in [mm] (don't have contours); e.g. 3 ==> downsample by taking every third point
    downsample: int
    upperBdNumContourPts: int # An upper bound on the number of contour points.
    PLOT: bool
    LOAD_MATLAB_VARS: bool

# [Load results by returning from function]

# This function displays "prompt" to the user and waits for the user to enter input. As long as the user enters
# incorrect input, "errorMessage" and "prompt" are displayed.
#
# The argument "goodInput" must be a boolean function accepting one argument. The intention is that, given a string
# variable called input, goodInput(input) is True when input is valid and false otherwise.
def getCorrectInput(goodInput, prompt, errorMessage):
    validInput = False
    while not validInput:
        returnThis = input(prompt)
        validInput = goodInput(returnThis)
        if not validInput:
            print(errorMessage)

    return returnThis

main()
