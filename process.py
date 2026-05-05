# Process and visualize data

import numpy as np
import matplotlib.pyplot as plt
import os

datadir = "data/output"
imagedir = "data/images/beam"
for file in os.listdir(datadir):
    file_path = os.path.join(datadir, file)

    array = np.fromfile(file_path, dtype=np.float64)
    array = array.reshape(32, 32)

    im = plt.imshow(array)
    plt.colorbar(im)
    plt.savefig(os.path.join(imagedir, file))
    plt.clf()
