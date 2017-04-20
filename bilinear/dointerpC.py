#!/usr/bin/env python
'''
Usage:
  dointerpC.py <interpdir> <res> <ifile> <ofile>

<interpdir>   directory with interpolation data
<res>         number of output points per degree

Example:
  ./dointerpC.py interp/ll4x4invconf 4 offline/ETAN/day.0000000288.data ll_ETAN.data
'''
import sys
sys.path[:0] = ['.']
import numpy as np
from h5py import File
from csinterp import CSInterp

ncs = 510
gridfile = '/home/jahn/l/h5/cube84/ro/grid.h5'

try:
    interpdir = sys.argv.pop(1)
    r         = int(sys.argv.pop(1))
    ifile     = sys.argv.pop(1)
    ofile     = sys.argv.pop(1)
except IndexError:
    sys.exit(__doc__)

sfx = '_{0}x{0}'.format(r)
csi = CSInterp.frombin('fij'+sfx, 'w'+sfx, ncs, dir=interpdir)

p = np.fromfile(ifile, '>f4').reshape(-1, 510, 3060)
nk = len(p)

with File(gridfile, 'r') as h:
    hFacC = h['hFacC'][:nk]

smcs = 0 != hFacC
wetfrac = csi(smcs)
sm = wetfrac >= .5
w = np.zeros_like(wetfrac)
w[sm] = 1./wetfrac[sm]
del wetfrac

pi = csi(p*smcs)
pi *= w
pi[w == 0] = np.nan

if nk == 1:
    pi, = pi

pi.astype('>f4').tofile(ofile)

