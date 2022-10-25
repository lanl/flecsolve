#!/usr/bin/env python

from pyevtk.hl import gridToVTK
import numpy as np
import sys
import os
from glob import glob

nx = eval(sys.argv[1])
ny = eval(sys.argv[2])

nsteps = len(glob('timestep*-0.dat'))

for step in range(nsteps):
    cmd = f'cat timestep{step}-*.dat > timestep{step}.dat'
    os.system(cmd)

for step in range(nsteps):
    fname = f'timestep{step}.dat'
    data = np.loadtxt(fname)

    x = np.zeros((nx, ny, 1))
    y = np.zeros((nx, ny, 1))
    z = np.zeros((nx, ny, 1))
    t = np.zeros((nx, ny, 1))
    x[:,:,0] = data[:, 0].reshape((nx, ny))
    y[:,:,0] = data[:, 1].reshape((nx, ny))
    t[:,:,0] = data[:, 2].reshape((nx, ny))

    ofname = f'vtk/step-{step}'
    gridToVTK(ofname, x, y, z, pointData = {"temperature": t})
