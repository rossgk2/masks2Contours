# masks2Contours

## Summary

This is a Python port of MATLAB code for a joint University of Michigan and King's College London research group. The MATLAB code took 3D selections as input (typically, we are interested in selecting the 3D geometry of a human heart) and produced contour points of those selections as output. The output contour points are also labeled according which chamber in the heart they reside in. 

"3D selections" are more commonly called "3D masks"- that is why this program is called `masks2Contours`.

The Python port improves upon the organization of the MATLAB code, and is integrated within the GUI [Eidolon](https://github.com/ericspod/Eidolon), a program written by Dr. Eric Kerfoot of King's College London. Dr. Renee Miller, also from King's College, wrote the MATLAB code, and I am the author of this Python port.

This is the Eidolon interface the user uses to select the NIFTI files that should be the input to the masks2Contours program.

<img src = "images/fileSelect.PNG" width = 300>

After the user starts masks2Contours (File -> Open Script), they use this interface to remove undesired points from nonempty slices in the long axis segmentation:

<img src = "images/lassoSelector.PNG" width = 300>

In the above, the user clicks and drags to lasso select points. Here's the full 3D output for some sample data.

<img src = "images/output.PNG" width = 500>

## How to run the script

1. Install [Eidolon](https://github.com/ericspod/Eidolon).
2. Download this project, and then place the masks2ContoursScripts subdirectory from this project inside the Eidolon folder.
3. Start Eidolon, click File -> Open Script, navigate to inside masks2ContoursScripts, and click on masks2ContoursEidolonScript.

## Differences between this Python port and the MATLAB original 

In the MATLAB original, the function `masks2contoursSA_manual()` converted masks to contours for the short axis (SA) images, and the function `masks2contoursLA_manual()` converted masks to contours for the long axis (LA) images. These functions are replaced by `masks2ContoursSA()` and `masks2ContoursLA()` in the Python port.

The Python port is more abstracted than the MATLAB original. **Before the commits on Jan. 11, 2021**, the Python port mostly mirrored the MATLAB code, with the main differences being that the Python code had the following abstractions:
- plotting functions (`subplotHelper` and `subplotHelperMulti`)
- a function for cleaning a mask and a function for getting contours from a mask slice
- a function to read from NIFTI files
- a function for converting to image coordinates

The Python port also has the significant difference (at least in the `oo_plotting` branch, which is the default branch) that it uses `matplotlib`'s object oriented plotting interface instead of its state-based interface.

**A major change was implemented in the commits on Jan. 11, 2021.** That change is that the `masks2ContoursSA()` and `masks2ContoursLA()` functions now both rely on the `slice2Contours()` function, and are therefore much simpler and shorter. In the MATLAB code, `masks2ContoursSA()` and `masks2ContoursLA()` did not call a common function.

Here is an overview of how `masks2ContoursSA()` and `masks2ContoursLA()` work. In general, a 3D image for the short axis (SA) is stored in a single file, while multiple 2D long axis (LA) images taken from different angles are stored in multiple files. The main difference between the short axis and the long axis is that *each 2D slice of the short axis 3D image has the same geometry metadata, while each long axis 2D image has different geometry metadata*. We can ignore this difference by pretending that each 2D slice of the short axis 3D image has varying geometry metadata- we just input the same metadata every iteration! So:

- In both `masks2ContoursSA()` and `masks2ContoursLA()`, we loop over 2D images and call `slice2Contours()` each iteration. 
- In `masks2ContoursSA()`, the 2D images are the 2D slices of a 3D short axis image.
- In `masks2ContoursLA()`, the 2D images come from the several long axis 2D image files (each of which is a snapshot taken from a different angle).
- In `masks2ContoursSA()`, the same geometry metadata is plugged into `slice2Contours()` in every iteration of the loop.
- In `masks2ContoursLA()`, all geometry metadata is precomputed before the loop over the 2D images. In the loop over 2D images, geometry metadata that changes with each 2D image is plugged into `slice2Contours()`.
