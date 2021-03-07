import sys
sys.path.append("masks2ContoursScripts")

import csv
from glob import glob
from typing import NamedTuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import masks2ContoursMainUtil as mut
import masks2ContoursUtil as ut
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
    LAcontours = masks2ContoursLA(LA_segs, frameNum, numSlices = len(LA_names), config = config, PyQt_objs = PyQt_objs)

    finishUp(SAresults, LAcontours, frameNum, fldr, imgName)

def finishUp(SAresults, LAcontours, frameNum, fldr, imgName):
    # Unpack input.
    (SAcontours, SAinserts) = SAresults

    # Get valve points.
    valves = getValvePoints(frameNum, fldr, imgName)

    # Initialize SAsliceIndices to hold the indices of the slices we will soon iterate over.
    # Short axis slices without any contours are omitted.
    SAsliceIndices = getSAsliceIndices(SAcontours)

    # Calculate apex using "method 2" from the MATLAB script.
    LA_LVepiContours = LAcontours["LVepi"]
    epiPts1 = np.squeeze(LA_LVepiContours[:, :, 0])
    epiPts2 = np.squeeze(LA_LVepiContours[:, :, 2])
    apex = mut.calcApex(epiPts1, epiPts2)

    # Plot the results.
    plotResults(SAsliceIndices, SAcontours, SAinserts, LAcontours, valves, apex)

    # Write the results to two text files.
    writeResults(frameNum, SAsliceIndices, SAcontours, SAinserts, LAcontours, valves, apex, fldr)

def getValvePoints(frameNum, fldr, imgName):
    # Get valve points for mitral valve (mv), tricuspid valve (tv), aortic valve (av), and pulmonary valve (pv).
    # Note: the valve points computed by this Python script are slightly different than those produced by MATLAB because
    # the interpolation function used in this code, np.interp(), uses a different interplation method than the MATLAB interp1().
    numFrames = nib.load(imgName).get_fdata().shape[3]  # all good
    (mv, tv, av, pv) = mut.manuallyCompileValvePoints(fldr, numFrames, frameNum)

    # Remove rows that are all zero from mv, tv, av, pv.
    mv = np.reshape(mv, (-1, 3))  # Reshape mv to have shape m x 3 for some m (the -1 indicates an unspecified value).
    mv = ut.removeZerorows(mv)
    tv = ut.removeZerorows(tv)
    av = ut.removeZerorows(av)
    pv = ut.removeZerorows(pv)
    return (mv, tv, av, pv)


def getSAsliceIndices(SAContours):
    SA_LVendoContours = SAContours["LVendo"]
    slicesToKeep = np.squeeze(SA_LVendoContours[0, 0, :]).astype(int)
    slicesToOmit1 = np.array(
        [i for i in range(SA_LVendoContours.shape[2]) if prepareContour(SA_LVendoContours, i).size == 0])
    slicesToOmit2 = np.logical_not(slicesToKeep)  # is 1 wherever slicesToKeep was 0.
    slicesToOmit = np.unique(np.concatenate((slicesToOmit1, slicesToOmit2)).astype(int))
    numSASlices = SA_LVendoContours.shape[2]
    SAslices = np.linspace(1, numSASlices, numSASlices)
    SAslices = ut.deleteHelper(SAslices, slicesToOmit, axis = 0).astype(int)
    return SAslices - 1  # convert to Python indexing


def plotResults(includedSlices, SAContours, SAinserts, LAContours, valves, apex):
    # Unwrap valve points.
    (mv, av, tv, pv) = valves

    # Plot short axis contours.
    fig, ax = plt.subplots(1, 1, subplot_kw = {"projection": "3d"})
    for i in includedSlices:
        LVendo = prepareContour(SAContours["LVendo"], i)
        LVepi = prepareContour(SAContours["LVepi"], i)
        RVFWendo = prepareContour(SAContours["RVFWendo"], i)
        RVFWepi = prepareContour(SAContours["RVFWepi"], i)
        RVsept = prepareContour(SAContours["RVsept"], i)
        RVinserts = prepareContour(SAinserts["RVinserts"], i)

        h0 = ax.scatter(LVendo[:, 0], LVendo[:, 1], LVendo[:, 2], marker = ".", color = "green")
        h1 = ax.scatter(LVepi[:, 0], LVepi[:, 1], LVepi[:, 2], marker = ".", color = "blue")
        h2 = ax.scatter(RVFWendo[:, 0], RVFWendo[:, 1], RVFWendo[:, 2], marker = ".", color = "red")
        h3 = ax.scatter(RVFWepi[:, 0], RVFWepi[:, 1], RVFWepi[:, 2], marker = ".", color = "blue")
        h4 = ax.scatter(RVsept[:, 0], RVsept[:, 1], RVsept[:, 2], marker = ".", color = "yellow")
        h5 = ax.scatter(RVinserts[:, 0], RVinserts[:, 1], RVinserts[:, 2], s = 50, color = "red")

    # Plot valve points.
    h6 = ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s = 50, color = "black")
    h7 = ax.scatter(av[:, 0], av[:, 1], av[:, 2], s = 50, color = "cyan")
    h8 = ax.scatter(tv[:, 0], tv[:, 1], tv[:, 2], s = 50, color = "magenta")

    # Plot apex.
    h9 = ax.scatter(apex[0], apex[1], apex[2], color = "black")

    # Plot long axis contours.
    numLASlices = LAContours["LVendo"].shape[2]
    for i in range(numLASlices):
        LVendo = prepareContour(LAContours["LVendo"], i)
        LVepi = prepareContour(LAContours["LVepi"], i)
        RVFWendo = prepareContour(LAContours["RVFWendo"], i)
        RVFWepi = prepareContour(LAContours["RVFWepi"], i)
        RVsept = prepareContour(LAContours["RVsept"], i)

        ax.scatter(LVendo[:, 0], LVendo[:, 1], LVendo[:, 2], marker = ".", color = "green")
        ax.scatter(LVepi[:, 0], LVepi[:, 1], LVepi[:, 2], marker = ".", color = "blue")
        ax.scatter(RVFWendo[:, 0], RVFWendo[:, 1], RVFWendo[:, 2], marker = "d", s = 30, color = "red") #using s = 30 for debug
        ax.scatter(RVFWepi[:, 0], RVFWepi[:, 1], RVFWepi[:, 2], marker = ".", color = "blue")
        ax.scatter(RVsept[:, 0], RVsept[:, 1], RVsept[:, 2], marker = "x", color = "yellow")

    ax.view_init()
    ax.legend((h0, h1, h2, h3, h4, h5, h6, h7, h8, h9),
              ("LV endo", "Epi", "RVFW endo", "RVFW epi", "RV sept", "RV inserts",
               "Mitral valve", "Aortic valve", "Tricuspid valve", "Apex"))
    plt.show()  # Must use plt.show() instead of fig.show(). Might have something to do with https://github.com/matplotlib/matplotlib/issues/13101#issuecomment-452032924


def writeResults(frameNum, includedSlices, SAContours, SAinserts, LAContours, valves, apex, fldr):
    # Unwrap valve points.
    (mv, av, tv, pv) = valves

    # Set up file writers.
    try:
        file1 = open(fldr + "GPFile_py.txt", "w", newline = "", encoding = "utf-8")
        file2 = open(fldr + "Case1_FR1_py.txt", "w", newline = "", encoding = "utf-8")
    except Exception as e:
        print(e)
        exit()

    writer1 = csv.writer(file1, delimiter = "\t")
    writer2 = csv.writer(file2, delimiter = " ")

    # TO-DO: precompute the results of prepareContour() and store them before plotting things, since results of
    # prepareContour() are used twice.

    def writePoint(point, j, name1, name2, weight = 1):
        writer2.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
                          name1, "{:d}".format(j + 1), "{:0.4f}".format(weight)])
        writer1.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
                         name2, "{:d}".format(j + 1), "{:0.4f}".format(weight), "{:d}".format(frameNum)])

    def writeContour(mask, i, j, name1, name2, rvi_weights = False):
        if rvi_weights:
            RVInsertsWeights = SAinserts["RVinsertsWeights"]
            multiIndex = np.unravel_index(i, RVInsertsWeights.shape, order = "F")
            weight = RVInsertsWeights[multiIndex]
        else:
            weight = 1

        for k in range(0, mask.shape[0]):
            writePoint(mask[k, :], j, name1, name2, weight)

    # Write short axis contours to text files
    writer1.writerow(["x", "y", "z", "contour type", "slice", "weight", "time frame"])
    for j, i in enumerate(
            includedSlices):  # i will be the ith included slice, and j will live in range(len(includedSlices))
        # LV endo
        LVendo = prepareContour(SAContours["LVendo"], i)
        writeContour(LVendo, i, j, "saendocardialContour", "SAX_LV_ENDOCARDIAL")

        # RV free wall
        RVFW = prepareContour(SAContours["RVFWendo"], i)
        writeContour(RVFW, i, j, "RVFW", "SAX_RV_FREEWALL")

        # Epicardium
        LVepi = prepareContour(SAContours["LVepi"], i)
        RVepi = prepareContour(SAContours["RVFWepi"], i)
        epi = np.vstack((LVepi, RVepi))
        writeContour(epi, i, j, "saepicardialContour", "SAX_LV_EPICARDIAL")

        # RV septum
        RVsept = prepareContour(SAContours["RVsept"], i)
        writeContour(RVsept, i, j, "RVS", "SAX_RV_SEPTUM")

        # RV inserts
        RVI = prepareContour(SAinserts["RVinserts"], i)
        writeContour(RVI, i, j, "RV_insert", "RV_INSERT", rvi_weights = True)

    # Now do long axis contours
    numLASlices = LAContours["LVendo"].shape[2]
    last_SA_j = len(includedSlices) - 1
    first_LA_j = last_SA_j + 1
    LA_i_range = range(numLASlices)  # we will use LA_i_range and LA_j_range after the loop
    LA_j_range = range(first_LA_j, first_LA_j + numLASlices)

    for i, j in zip(LA_i_range, LA_j_range):
        # LV endo
        LVendo = prepareContour(LAContours["LVendo"], i)
        writeContour(LVendo, i, j, "saendocardialContour", "LAX_LV_ENDOCARDIAL")

        # RV free wall
        RVFW = prepareContour(LAContours["RVFWendo"], i)
        writeContour(LVendo, i, j, "RVFW", "LAX_RV_FREEWALL")

        # Epicardium
        LVepi = prepareContour(LAContours["LVepi"], i)
        RVepi = prepareContour(LAContours["RVFWepi"], i)
        epi = np.vstack((LVepi, RVepi))
        writeContour(epi, i, j, "saepicardialContour", "LAX_LV_EPICARDIAL")

        # RV septum
        RVsept = prepareContour(LAContours["RVsept"], i)  # not doing isempty check
        writeContour(RVsept, i, j, "RVS", "LAX_RV_SEPTUM")

        # Valves (tricuspid, mitral, aortic, pulmonary)
        writeContour(tv, i, j, "Tricuspid_Valve", "TRICUSPID_VALVE")
        writeContour(mv, i, j, "BP_point", "MITRAL_VALVE")
        writeContour(av, i, j, "Aorta", "AORTA_VALVE")
        writeContour(pv, i, j, "Pulmonary", "PULMONARY_VALVE")

    # Finally, write out the apex
    writePoint(apex, LA_j_range[-1], "LA_Apex_Point", "APEX_POINT")

def prepareContour(mask, sliceIndex):
    result = np.squeeze(mask[:, :, sliceIndex])
    return ut.removeZerorows(result)

class test:
    pass

t = test()
main(t)