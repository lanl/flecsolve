#!/usr/bin/env python
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from glob import glob
import os
import imageio

nx = eval(sys.argv[1])
ny = eval(sys.argv[2])

nsteps = len(glob('timestep*-0.dat'))

for step in range(nsteps):
    cmd = f'cat timestep{step}-*.dat > timestep{step}.dat'
    os.system(cmd)

osteps = 20
skip = nsteps // osteps

fnames = []
for step in range(0, nsteps, skip):
    fname = f'timestep{step}.dat'
    data = np.loadtxt(fname)

    ind = np.lexsort((data[:, 0], data[:,1]))
    u = data[:, -1][ind].reshape(nx, ny)

    plt.clf()
    plt.pcolormesh(u, cmap=plt.cm.turbo, vmin=0, vmax=50.)
    plt.colorbar()
    ofname = f'figs/sol-{step}.png'

    plt.savefig(ofname)
    plt.close()

    fnames.append(ofname)

with imageio.get_writer('figs/solution.gif', mode='i') as writer:
    for fname in fnames:
        image = imageio.imread(fname)
        writer.append_data(image)
