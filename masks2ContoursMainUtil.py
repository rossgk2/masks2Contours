import numpy as np
import masks2ContoursUtil as ut
from glob import glob
import scipy.io as sio
from scipy.spatial.distance import cdist

# Return the point p1 in epiPts1 such that p1 minimizes the distance between p1 and p2, where p2 can be
# any point in epiPts2.
def calcApex(epiPts1, epiPts2):
    epiPts1 = ut.removeZerorows(epiPts1)
    epiPts2 = ut.removeZerorows(epiPts2)
    dist = cdist(epiPts1, epiPts2) # Compute pairwise distances.
    apexIndex = np.unravel_index(np.argmin(dist), dist.shape)[0]
    if apexIndex.shape != ():
        apexIndex = apexIndex[0]

    return epiPts1[apexIndex, :]

# fldr is the folder where the valve points .mat files are stored.
# numFrames is the number of frames in the SA image.
#
# Returns the tuple (mv, tv, av, pv), where...
# - mv is a 3D ndarray such that mv[i, :, :] is the 2D ndarray containing the mitral valve points for the ith mitral valve
# file in fldr.
# - tv, av, and pv are 2D ndarrays containing the tricuspid, aortic, and pulmonary valve points. tv, av, and pv may
# all be None.
def manuallyCompileValvePoints(fldr, numFrames, frameNum):
    mat_files = glob(fldr + "valve-motion-predicted-LA_[0-9]CH.mat")

    mv = np.zeros((len(mat_files), 2, 3))
    tv = np.zeros((2, 3))
    av = np.zeros((2, 3))
    pv = np.zeros((2, 3))

    for i, file in enumerate(mat_files):
        # Load the variable point_coords_mv from the MATLAB file.
        mat_file = sio.loadmat(file)
        point_coords_mv = mat_file["point_coords_mv"]

        # Interpolate onto applicable number of frames if needed.
        tmp = interpTime(point_coords_mv, numFrames)
        mv[i, :, :] = tmp[frameNum, :, :]

    def get_interp_mat_var(mat_var):
        var_name = "point_coords_" + mat_var
        result = None
        if var_name in mat_file:
            point_coords_mat_var = mat_file[var_name]
            result = interpTime(point_coords_mat_var, numFrames)
            result = np.squeeze(result[frameNum, :, :])

        return result

    # Return stuff.
    _tv = get_interp_mat_var("tv")
    _av = get_interp_mat_var("av")
    _pv = get_interp_mat_var("pv")

    if _tv is not None:
        tv = _tv
    if _av is not None:
        av = _av
    if _pv is not None:
        pv = _pv

    return (mv, tv, av, pv)

# oldValvePts is a m x 2 ndarray, and newNumInterpSamples is a positive integer.
# Returns the new, interpolated valve positions.
def interpTime(oldValvePts, newNumInterpSamples):
    result = np.zeros((newNumInterpSamples, oldValvePts.shape[1], oldValvePts.shape[2]))

    for i in range(0, oldValvePts.shape[1]):
        for j in range(0, oldValvePts.shape[2]):
            sampledValues = np.squeeze(oldValvePts[:, i, j])
            sampledValues = ut.deleteHelper(sampledValues, sampledValues == 0, axis = 0) # Remove all zeros from sampledValues
            if sampledValues.size != 0:
                samplePts = np.linspace(1, 100, len(sampledValues))
                queryPts = np.linspace(1, 100, newNumInterpSamples)
                result[:, i, j] = np.interp(queryPts, samplePts, sampledValues)
            else:
                result[:, i, j] = np.zeros(newNumInterpSamples, 1)

    return result