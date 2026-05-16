# Process and visualize data

import numpy as np
import matplotlib.pyplot as plt
import os

w = 2
h = 2
d = 2

datadir = "data/output"
imagedir = "data/images"

for file in os.listdir(datadir):
    file_path = os.path.join(datadir, file)

    array = np.fromfile(file_path, dtype=np.float64)
    array = array.reshape(w, h * d)

    im = plt.imshow(array)
    plt.colorbar(im)
    plt.savefig(os.path.join(imagedir, file))
    plt.clf()
