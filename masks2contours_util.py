import numpy as np

# "points" is a m1 x 2 ndarray for some m1.
#
# Returns an 1 x m2 ndarray, for some m2, containing the indices of endoRVFWContours that correspond to the RV insert points.
def getRVinsertIndices(points):
     distances = pointDistances(points)
     upperThreshold = np.mean(distances) + 3 * np.std(distances)
     largeDists = distances > upperThreshold

    # Find the index (in "distances") of the point that is furthest from its neighbor. Return an ndarray consisting of
    # this point and *its* neighbor.
     if largeDists.size != 0:
         largestDistIndex = np.argmax(largeDists)
         if largestDistIndex == len(points) - 1: # if the point furthest from its neighbor is the last point...
             return np.array([0, largestDistIndex]) #the neighbor to largestDistIndex is 0 in this case
         else:
             return np.array([largestDistIndex, largestDistIndex + 1])
     else:
         return np.array([])

# "oldPoints" is a m1 x 2 ndarray for some m1.
#
# Returns a m2 x 2 ndarray, for some m2, that is the result of removing points that are "too far" from the majority of
# the points in "oldPoints".
def removeFarPoints(oldPoints):
    # If there are 2 or less points, then no points will ever be considered "too far". (The condition for "too far" is
    # given in the computation of far_indices, below).
    if oldPoints.shape[0] <= 2:
        return oldPoints

    # Record the distances between consecutive points (wrap around from the last point to the first point) in oldPoints.
    distances = pointDistances(oldPoints)

    # Find points that are far away from other points. In the below, note that simply calling np.nonzero() doesn't do
    # the trick; np.nonzero() returns a tuple, and we want the first element of that tuple.
    far_indices = np.nonzero(distances > (np.mean(distances) + 2 * np.std(distances)))[0]

    # The point must be far from its two nearest neighbours, therefore, if there are extraneous points, the length of
    # 'far_indices' must be even. If the length of 'far_indices' is odd, the contour is open (but there still may be
    # extraneous points).

    if np.size(far_indices) > 1:
        # Find these extraneous points and remove them.
        indicesToRemove = []
        for i in range(0, np.size(far_indices) - 1): # i covers the range {0, ..., np.size(far_indices) - 2} since np.size(far_indices) - 1 isn't included
            if (far_indices[i] == 0) and (far_indices[-1] == np.size(oldPoints) - 1):
                indicesToRemove.append(far_indices[i])
            elif far_indices[i + 1] - far_indices[i] == 0:
                indicesToRemove.append(far_indices[i + 1])

        # Remove the extraneous points.
        return deleteHelper(oldPoints, indicesToRemove)
    else:
        return oldPoints # Remove nothing in this case.

# "points" is an m x 2 ndarray for some m.
#
# Returns an m x 1 ndarray in which the ith entry is the distance from the ith point in "points" to the (i + 1)st point.
# The last entry is the distance from the last point to the first point.
def pointDistances(points):
    numPts = points.shape[0]
    distances = []
    for i in range(0, numPts - 1): # i covers the range {0, ..., numPts - 2} since numPts - 1 isn't included
        if i == numPts - 1:  # If on last point, compare last point with first point
            distances.append(np.linalg.norm(points[i, :] - points[0, :]))
        else:
            distances.append(np.linalg.norm(points[i + 1, :] - points[i, :]))

    return np.array(distances)

# Helper function for deleting parts of ndarrays that, unlike np.delete(), works when either "arr" or "indices" is empty.
#
# "arr" may either be a list or an ndarray.
def deleteHelper(arr, indices, axis = 0):
    #np.delete() does not handle the case in which arr is an empty list or an empty ndarray.
    if (type(arr) is np.ndarray and arr.size == 0) or (type(arr) is list and len(list) == 0):
        return arr
    elif type(indices) is np.ndarray and indices.size == 0: # np.delete() does not work as expected if arr is an empty ndarray. This case fixes that. (If indices is an empty list, np.delete() works as expected).
        return arr
    else:
        return np.delete(arr, indices, axis)

# "arr1" and "arr2" must be m x n ndarrays for some m and n. (So they can't be m x n x s ndarrays).
#
# Returns the list [_sharedRows, sharedIndicesArr1, sharedIndicesArr2].
# "_sharedRows" is a matrix whose rows are those that are shared by "arr1" and "arr2".
# "sharedIndicesArr1" is a list of the row-indices in arr1 whose rows appear (not necessarily with the same indices) in
# arr2.
# "sharedIndicesArr2" is a similar list of row-indices that pertains to "arr2".
def sharedRows(arr1, arr2):
    if arr1.size == 0 or arr2.size == 0: #If either array is empty, return a list containing three empty ndarrays.
        return[np.array([]), np.array([]), np.array([])]

    # First, just get the indices of the shared rows.
    sharedIndicesArr1 = []
    sharedIndicesArr2 = []
    jRange = list(range(0, arr2.shape[1])) # Needs to be a list so we can use remove()
    for i in range(0, arr1.shape[0]):
        for j in jRange:
            if np.all(arr1[i, :] == arr2[j, :]): #If (ith row in arr1) == (ith row in arr2)
                sharedIndicesArr1.append(i)
                sharedIndicesArr2.append(j)
                jRange.remove(j) # After we visit a row in arr2, we don't need to visit it again.

    # Use these indices to build the matrix of shared rows.
    _sharedRows = [arr1[i] for i in sharedIndicesArr1]

    # Convert everything to ndarrays.
    _sharedRows = np.array(_sharedRows)
    sharedIndicesArr1 = np.array(sharedIndicesArr1)
    sharedIndicesArr2 = np.array(sharedIndicesArr2)

    return [_sharedRows, sharedIndicesArr1, sharedIndicesArr2]