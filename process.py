# Process and visualize data

import numpy as np
import matplotlib.pyplot as plt
import os

size = 128

datadir = "data/output"
imagedir = "data/images/circle"

for file in os.listdir(datadir):
    file_path = os.path.join(datadir, file)

    array = np.fromfile(file_path, dtype=np.float64)
    array = array.reshape(size, size)

    im = plt.imshow(array)
    plt.colorbar(im)
    plt.savefig(os.path.join(imagedir, file))
    plt.clf()
