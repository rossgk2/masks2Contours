from glob import glob
import nibabel as nib
import valvePoints as vp
import numpy as np

import os
from typing import NamedTuple
from masks2Contours import masks2ContoursSA, masks2ContoursLA

def main():
    use_default_filepaths = True

    # Orientation for viewing plots
    az = 214.5268
    el = -56.0884

    # Configure some settings.
    config = Config(rvWallThickness = 3, downsample = 3, upperBdNumContourPts = 200, PLOT = True, LOAD_MATLAB_VARS = False)
    frameNum = 1 # Index of frame to be segmented.

    # Get filepaths ready.
    fldr = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\"
    resultsDir = fldr + "out\\"

    # These are filepaths for the SA.
    imgName = fldr + "CINE_SAX.nii"
    segName = fldr + "CINE_SAX_" + str(frameNum) + ".nii"

    # These are lists of filepaths for the LA.
    LA_segs = glob(fldr + "LAX_[0-9]ch_1.nii")
    LA_names = glob(fldr + "RReg_LAX_[0-9]ch_to_Aligned_SA.nii")

    # The following function produces contour points from masks for the short axis image files, displays them
    # if PLOT == True, and returns (endoLVContours, epiLVContours, endoRVFWContours, epiRVFWContours, RVSContours, RVInserts, RVInsertsWeights).
    # Each variable in this tuple, except for the last two, is a m x 2 ndarray for some m.

    # masks2ContoursSA(segName, imgName, resultsDir, frameNum, config)

    # masks2ContoursLA(LA_names, LA_segs, resultsDir, frameNum, config)

    # Get valve points for mitral valve (mv), tricuspid valve (tv), aortic valve (av), and pulmonary valve (pv).
    # Note: the valve points computed by this Python script are slightly different than those produced by MATLAB because
    # the interpolation function used in this code, np.interp(), uses a different interplation method than the MATLAB interp1().
    numFrames = nib.load(imgName).get_fdata().shape[3]  # all good
    (mv, tv, av, pv) = vp.manuallyCompileValvePoints(fldr, numFrames, frameNum)

    def nonzeroRows(xv): # Returns result of removing rows of xv that are all 0.
        return xv[np.any(xv, axis = 1), :]

    mv = np.reshape(mv, (-1, 3)) # Reshape mv to have shape m x 3 for some m (the -1 indicates an unspecified value).
    mv = nonzeroRows(mv)
    tv = nonzeroRows(tv)
    av = nonzeroRows(av)
    pv = nonzeroRows(pv)








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
