import os
import numpy as np
import PIL
import PIL.Image

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"

def batch_vizseg(img, labels, labels2=None, colormap=None, gridsize=(3,8), saveto=None):
    # Only use the number of images needed to fill grid
    rows, cols = gridsize
    assert rows>0 and cols>0, "rows and cols must be positive integers"
    n_cells = (rows*cols)
    n_samples = img.shape[0]

    if colormap is None:
        colormap = [[0,0,0], [255,79,64], [115,173,33],[48,126,199]]

    # Group the images into pairs/triplets
    output = []
    for i in range(min(n_cells, n_samples)):
        x = img[i]
        y = labels[i]

        # TODO: make the following more generic so it works with RGB images too
        # Convert X to 3 channels
        x = np.repeat(x, 3, axis=2)

        # Apply colormap to labels and predictions
        y = np.array(colormap)[y].astype(np.uint8)

        if labels2 is not None:
            y2 = labels2[i]
            y2 = np.array(colormap)[y2].astype(np.uint8)
            output.append(np.concatenate([x,y,y2], axis=0))
        else:
            output.append(np.concatenate([x,y], axis=0))

    output = np.array(output, dtype=np.uint8)

    # Prepare dimensions of the grid
    n_batch, img_height, img_width, n_channels = output.shape

    # Handle case where there is not enough images in batch to fill grid
    n_gap = n_cells - n_batch
    output = np.pad(output, pad_width=[(0,n_gap),(0,0), (0,0), (0,0)], mode="constant", constant_values=0)

    # Reshape into grid
    output = output.reshape(rows,cols,img_height,img_width,n_channels).swapaxes(1,2)
    output = output.reshape(rows*img_height,cols*img_width,n_channels)

    output = PIL.Image.fromarray(output.squeeze())

    # Optionally save image
    if saveto is not None:
        # Create necessary file structure
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "": # ensure pardir is not an empty string
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        output.save(saveto, "JPEG")

    return output
