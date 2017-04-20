#!/usr/bin/env python
"""
Usage:
  preprocess-ecco-data.py <ifile> <ofile>

<ifile>	input data file
<ofile> output data file

Takes in a 3 dimensional ecco file of velocities and averages the the top ten layers to give a 2d velocity field
	

"""
import numpy as np
import sys

try:
	ifile = sys.argv.pop(1)
	ofile = sys.argv.pop(1)

except IndexError:
	sys.exit(__doc__)

uvel = np.fromfile(ifile, ">f4")
uvel = uvel.reshape(-1, 510, 3060)[0:10]

uvel = np.sum(uvel, 0)/10

uvel = uvel.reshape(-1)
print(np.sum(uvel))

np.save(ofile, uvel)


saved = np.load(ofile)
print(np.sum(uvel-saved))
