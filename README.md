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

### The Python port abstracts the process of converting a segmentation *slice* to contours

The Python port is more abstracted than the MATLAB original. In the MATLAB code, though `masks2contoursSA_manual()` and `masks2contoursLA_manual()` performed very similar sequences of tasks, this shared behaivor was not made obvious by a common dependency on a helper function. In the Python port, `masks2ContoursSA()` and `masks2ContoursLA()` do depend on a common helper function.

To understand precisely what `masks2ContoursSA()` and `masks2ContoursLA()` have in common, we must know that each 2D slice of the short axis 3D image has the same geometry metadata, while each long axis 2D image has different geometry metadata. Since "same geometry metadata in each slice" is a special case of "varying geometry metadata in each slice", we can use the following pseudocode for our program:

```
def masks2ContoursSA():
  geometryMetadata = getGeometryMetadataSA()
  for each short axis slice s:
    slice2Contours(geometryMetadata) # geometryMetadata is the same each iteration
    
def masks2ContoursLA():
  for each long axis slice s:
    geometryMetadata = getGeometryMetadataLA(s) 
    slice2Contours(geometryMetadata) # geometryMetadata depends on the long axis slice, s
```

Notice that in this pseudocode, `masks2ContoursSA()` and `masks2ContoursLA()` depend on the common helper function `slice2Contours()`.

### Iterative when `SA_LA.lower() == "sa"`, recursive when `SA_LA.lower() == "la"`

Since the Eidolon GUI runs on one thread and the Python script on another, we can't return control from the lasso selector back to the function which created the lasso selector by waiting; if we want some function `f` to execute after the lasso selector is done doing its thing, we must explicitly call `f` from within one of the lasso selector's callback functions.

This means that, when processing the long axis segmentation, we have to split up the `slice2Contours()` process into a `slice2ContoursPt1()` process and a `slice2ContoursPt2()` process. For reasons explained soon, we will actually split up the `slice2Contours()` process into a `slice2ContoursPt1(SA_LA = "la")` process and a `slice2ContoursPt2(SA_LA = "la")` process. 

The `masks2ContoursLA()` function proceeds as follows:

```
def masks2ContoursLA():
  slice2ContoursPt1(SA_LA_ = "la")
  
def slice2ContoursPt1(SA_LA):
  execute some tasks
  if SA_LA == "la":
    set up a lasso selector if necessary (if not, call slice2ContoursPt2()).
    # the lasso selector responds to user input, and then calls slice2ContoursPt2()

def slice2ContoursPt2(SA_LA):
  execute some more tasks
  if SA_LA == "la":
    call slice2ContoursPt1() if the slice just processed was not the last; call finishUp() otherwise
```

Note, the `slice2Contours(SA_LA = "la")` process is recursive, since `slice2ContoursPt1(SA_LA = "la")` and `slice2ContoursPt2(SA_LA = "la")` call each other. 

Now, we explain why we have parameterzied `slice2ContoursPt1()` and `slice2ContoursPt2()` by `SA_LA`. The intention of this is to allow `slice2ContoursPt1(SA_LA = "la")` and `slice2ContoursPt2(SA_LA = "la")` to "collectively" constitue a recursive function, while having `slice2ContoursPt1(SA_LA = "sa")` and `slice2ContoursPt2(SA_LA = "sa")` collectively constitute a traditional imperative function that can be looped over, like this:

```
def masks2ContoursSA():
  for each short axis slice s:
    slice2ContoursPt1(s, SA_LA_ = "sa")
```

n port, we need some way of returning control from the lasso selector that is used to interact with slices of the long axis segmentation. Unfortunately, since the Python `masks2Contours` script is integrated with a GUI, we can't just wait for control to return to whatever function set up the lasso selector

Since the Python port of masks2Contours is integrated into a GUI framework, it needs to be able to run asynchronously.

complicates things a little bit

This means that we *cannot* return control from the lasso selector 

It may be a little confusing that that helper function which `masks2ContoursSA()` and `masks2ContoursLA()` depend on is `slice2ContoursPt1()'.





**Before the commits on Jan. 11, 2021**, the Python port mostly mirrored the MATLAB code, with the main differences being that the Python code had the following abstractions:
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
