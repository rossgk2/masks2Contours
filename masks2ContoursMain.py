import sys
sys.path.append("masks2ContoursScripts")

from glob import glob
from typing import NamedTuple
from types import SimpleNamespace

import numpy as np

from masks2Contours import masks2ContoursSA, masks2ContoursLA

class Config(NamedTuple):
    rvWallThickness: int  # RV wall thickness, in [mm] (don't have contours); e.g. 3 ==> downsample by taking every third point
    downsample: int
    upperBdNumContourPts: int  # An upper bound on the number of contour points.

def main(PyQt_objs):
    # For debugging: don't use scientific notation when printing out ndarrays.
    np.set_printoptions(suppress = True)

    # Configure some settings.
    config = Config(rvWallThickness = 3, downsample = 3, upperBdNumContourPts = 200)
    frameNum = 1  # Index of frame to be segmented.

    # Get filepaths ready.
    fldr = "C:\\Users\\Ross\\Documents\\Data\\CMR\\Student_Project\\P3\\"

    # These are filepaths for the SA.
    imgName = fldr + "CINE_SAX.nii"
    segName = fldr + "CINE_SAX_" + str(frameNum) + ".nii"

    # These are lists of filepaths for the LA.
    LA_segs = glob(fldr + "LAX_[0-9]ch_1.nii")
    LA_names = glob(fldr + "RReg_LAX_[0-9]ch_to_Aligned_SA.nii")

    # The following two functions, masks2ContoursSA() and masks2ContoursLA(), produce labeled contour points from masks for the
    # short and long axis image files, respectively.
    #
    # SAcontours and LAcontours are dicts whose keys are strings such as "LVendo" or "RVsept" and whose values are
    # m x 2 ndarrays containing the contour points. SAinserts is a dict with the two keys "RVinserts" and "RVInsertsWeights".
    SAresults = masks2ContoursSA(segName, frameNum, config)

    mainObjs = SimpleNamespace(SAresults = SAresults, frameNum = frameNum, fldr = fldr, imgName = imgName)
    masks2ContoursLA(LA_segs = LA_segs, frameNum = frameNum, numSlices = len(LA_names), config = config,
                     PyQt_objs = PyQt_objs, mainObjs = mainObjs)



class test:
    pass

t = test()
main(t)