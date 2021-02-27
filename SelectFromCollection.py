# This file is from https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html.
import sys
import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, fig, ax, collection, alpha_other=0.3):
        self.fig = fig
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        self.is_subtract = True

        self.pts = collection.get_offsets()
        self.Npts = len(self.pts)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # All points are initially selected.
        self.fc[:, -1] = 1 # 1 corresponds to "filled in" instead of translucent
        self.collection.set_facecolors(self.fc)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        # Register the callback functions.
        fig.canvas.mpl_connect("key_press_event", self.key_press)
        fig.canvas.mpl_connect("key_release_event", self.key_release)
        ax.set_title("Press enter to accept selected points.")

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.pts))[0]

        # Subtract or add selections depending on whether shift is pressed or not.
        if self.is_subtract:
            self.fc[self.ind, -1] = self.alpha_other
        else:
            self.fc[self.ind, -1] = 1

        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def key_press(self, event):
        if event.key == "shift":
            self.is_subtract = False
        elif event.key == "enter":
            print("Removed points:")
            print(self.pts[self.ind])
            self.disconnect()
            self.ax.set_title("")
            self.fig.canvas.draw()
            plt.close("all")  # This closes all matplotlib windows and destroys their figure managers.

    def key_release(self, event):
        if event.key == "shift":
            self.is_subtract = True

# An example on how to use the above class:

if __name__ == '__main__':
    # Select random grid points.
    np.random.seed(19680801)
    data = np.random.rand(100, 2)

    # Useless second plot window.
    f1 = plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=80)

    # Set up first plot window with lasso selector.
    subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
    fig, ax = plt.subplots(subplot_kw=subplot_kw)
    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    selector = SelectFromCollection(fig, ax, pts)

    plt.show()

    print("Done")
