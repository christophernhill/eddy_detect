#!/usr/bin/env python
'''
Usage:
  dointerpvel.py <interpdir> <res> <ifile> <ofile>

<interpdir>   directory with interpolation data
<res>         number of output points per degree

Examples:
  ./dointerpvel.py interp/ll4x4invconf 4 UE.0000000288.data ll_UE.data
  ./dointerpvel.py interp/ll4x4invconf 4 VN.0000000288.data ll_VN.data

only difference to dointerppart.py should be different landmask
'''
import sys
sys.path[:0] = ['.']
import numpy as np
from h5py import File
from csinterp import CSInterp

nf = 6
ncs = 510

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

smcs = 0 == np.fromfile('lm510velc_i1.bin', 'i1').reshape(1, ncs, nf*ncs)
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

