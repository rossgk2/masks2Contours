import numpy as np

# "vertices" is a m x 2 ndarray for some m, and "lines" is a (m - 1) x 2 ndarray of indices
# in the "vertices" ndarray such that the ith row of lines contains the indices of the points that
# constitute the ith line segment.
#
# Returns an m x 2 ndarray, where the ith row contains the components of the normal vector corresponding to the direction
# of the ith line segment.
#
# This function was ported over from a MATLAB function written by D. Kroon from University of Twente in August 2011.
def lineNormals2D(vertices, lineIndices = None):
    numVertices = vertices.shape[0]

    if numVertices != 0 and numVertices < 2:
        raise ValueError("\"vertices\" must have either 0 rows or 2 or more rows; that is, there must be eiher 0 vertices or more than 2 vertices.")

    if vertices is None or (type(vertices) == np.ndarray and vertices.size == 0):
        return np.array([])

    # If nothing is passed in for "lines", initialize lines to the ndarray [[0, 1], [1, 2], ..., [numVertices - 2, numVertices - 1]].
    if lineIndices is None or (type(lineIndices) == np.ndarray and lineIndices.size == 0):
        lineIndices = np.column_stack((np.array(range(0, numVertices - 1)), np.array(range(1, (numVertices)))))

    # Calculate tangent vectors.
    DT = vertices[lineIndices[:, 0], :] - vertices[lineIndices[:, 1], :]

    # Divide each tangent vector by its length. ("Weighted central difference"?)
    LL = np.sqrt(np.power(DT[:, 0], 2) + np.power(DT[:, 1], 2))
    eps = np.spacing(1) # Returns the distance from 1 to the nearest number representible by the computer. I.e., is really small.
    LL2 = np.maximum(np.power(LL, 2), eps) # This is essentially the elementwise second power of LL.
    DT[:, 0] = np.divide(DT[:, 0], LL2) # The scaling happens in this line and the one below.
    DT[:, 1] = np.divide(DT[:, 1], LL2)

    # ???
    D1 = np.zeros(vertices.shape)
    D1[lineIndices[:, 0], :] = DT
    D2 = np.zeros(vertices.shape)
    D2[lineIndices[:, 1], :] = DT
    D = D1 + D2

    # Normalize the normals.
    LL = np.sqrt(np.power(D[:, 0], 2) + np.power(D[:, 1], 2))
    LL = np.maximum(LL, eps) # The equivalent of this line was not in the MATLAB original, but is necessary to prevent division by zero ocurring as a result of division by too-small.
    normals = np.zeros(D.shape) # Note that D.shape == vertices.shape.
    normals[:, 0] = np.divide(-D[:, 1], LL)
    normals[:, 1] = np.divide(D[:, 0], LL)

    return normals


# "points" is a m1 x 2 ndarray for some m1.
#
# Returns an 1 x m2 ndarray, for some m2, containing the indices of endoRVFWContours that correspond to the RV insert points.
def getRVinsertIndices(points):
     distances = pointDistances(points)
     upperThreshold = np.mean(distances) + 3 * np.std(distances, ddof = 1) # We need to use ddof = 1 to use Bessel's correction (so we need it to get the same std as is calculated in MATLAB).
     largeDists = distances > upperThreshold

    # Find the index (in "distances") of the point that is furthest from its neighbor. Return an ndarray consisting of
    # this point and *its* neighbor.
     if largeDists.size != 0:
         largestDistIndex = np.argmax(distances)
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

    # Comment from the MATLAB script: "The point must be far from its two nearest neighbours, therefore, if
    # there are extraneous points, the length of variable 'far' must be even.
    # If the length of 'far' is odd, the contour is open (but there still may
    # be extraneous points)." What???

    # Every point that is far from *both* of its neighbors is considered to be a "far point". We return the result
    # of removing all such far points from oldPoints.
    if np.size(far_indices) > 1: # Using > 1 instead of != 0 probably fixes some edge case.
        indicesToRemove = []
        for i in range(0, np.size(far_indices) - 1): # i covers the range {0, ..., np.size(far_indices) - 2} since np.size(far_indices) - 1 isn't included
            if (far_indices[i] == 0) and (far_indices[-1] == np.size(oldPoints) - 1):
                indicesToRemove.append(far_indices[i])
            elif far_indices[i + 1] - far_indices[i] == 1:
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
    for i in range(0, numPts): # i covers the range {0, ..., numPts - 1} since numPts isn't included
        if i == numPts - 1:  # If on last point, compare last point with first point
            distances.append(np.linalg.norm(points[i, :] - points[0, :]))
        else:
            distances.append(np.linalg.norm(points[i + 1, :] - points[i, :]))

    return np.array(distances)

# Helper function for deleting parts of ndarrays that, unlike np.delete(), works when either "arr" or "indices" is empty.
# (np.delete() does not handle the case in which arr is an empty list or an empty ndarray).
#
# "arr" may either be a list or an ndarray.
def deleteHelper(arr, indices, axis = 0):
    def emptyNdarrayCheck(x):
        return type(x) is np.ndarray and ((x == np.array(None)).any() or x.size == 0)

    def emptyListCheck(x):
        return type(x) is list and len(x) == 0

    if emptyNdarrayCheck(arr) or emptyListCheck(arr):
        return arr

    # np.delete() does not work as expected if arr is an empty ndarray. This case fixes that.
    # (If indices is an empty list, np.delete() works as expected, so there is no need to call emptyListCheck() here).
    if emptyNdarrayCheck(indices):
        return arr

    return np.delete(arr, indices, axis)

# "arr1" and "arr2" must be m x n ndarrays for some m and n. (So they can't be m x n x s ndarrays).
#
# Returns the list [_sharedRows, sharedIndicesArr1, sharedIndicesArr2].
# "_sharedRows" is a matrix whose rows are those that are shared by "arr1" and "arr2".
# "sharedIndicesArr1" is a list of the row-indices in arr1 whose rows appear (not necessarily with the same indices) in
# arr2.
# "sharedIndicesArr2" is a similar list of row-indices that pertains to "arr2".
def sharedRows(arr1, arr2):
    if arr1 is None or arr1.size == 0 or arr2 is None or arr2.size == 0: #If either array is empty, return a list containing three empty ndarrays.
        return[np.array([]), np.array([]), np.array([])]

    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError("Arrays must have same number of columns.")

    # Get the indices of the shared rows.
    sharedIndicesArr1 = []
    sharedIndicesArr2 = []

    for i in range(0, arr1.shape[0]):
        for j in range(0, arr2.shape[0]):
            if i in sharedIndicesArr1 or j in sharedIndicesArr2:
                continue
            elif np.all(arr1[i, :] == arr2[j, :]): #If (ith row in arr1) == (jth row in arr2)
                sharedIndicesArr1.append(i)
                sharedIndicesArr2.append(j)

    # Use these indices to build the matrix of shared rows.
    _sharedRows = [arr1[i] for i in sharedIndicesArr1]

    # Convert everything to ndarrays.
    _sharedRows = np.array(_sharedRows)
    sharedIndicesArr1 = np.array(sharedIndicesArr1)
    sharedIndicesArr2 = np.array(sharedIndicesArr2)

    return [_sharedRows, sharedIndicesArr1, sharedIndicesArr2]
