#!/usr/bin/env python
'''
Usage:
  dorotateuv.py <ufile> <vfile> <uefile> <vnfile>

Example:
  ./dorotateuv.py offline/UVELMASS/day.0000000288.data offline/VVELMASS/day.0000000288.data UE.0000000288.data VN.0000000288.data
'''
import sys
import numpy as np
import facets as fa
from exchange import Exchange
from h5py import File

nf = 6 # num faces on cube-sphere
ncs = 510 # dimensions of one face (510x510)
gridfile = '/home/jahn/l/h5/cube84/ro/grid.h5'

try:
    ufile = sys.argv.pop(1)
    vfile = sys.argv.pop(1)
    uefile = sys.argv.pop(1)
    vnfile = sys.argv.pop(1)
except IndexError:
    sys.exit(__doc__)

dims = 2*nf*[ncs]
exch = Exchange.cs()

with File(gridfile, 'r') as h:
    cs = fa.fromglobal(h['AngleCS'], dims=dims, dtype=np.float64)
    sn = fa.fromglobal(h['AngleSN'], dims=dims, dtype=np.float64)
    smw = fa.fromglobal(0 != h['hFacW'][0], dims=dims, extrau=1, dtype=np.int8)
    sms = fa.fromglobal(0 != h['hFacS'][0], dims=dims, extrav=1, dtype=np.int8)

exch.ws(smw.F, sms.F)

smc = (smw[..., :, :-1] + smw[..., :, 1:])*(sms[..., :-1, :] + sms[..., 1:, :]) != 0.0
(0==smc).toglobal().astype('i1').tofile('lm510velc_i1.bin')

#np.fromfile count paraeter gives the number of items to read
# nf*ncs*ncs = total number of indices to read in = total number of grid cells
 
udata = np.load(ufile)
vdata = np.load(vfile)

u = fa.fromglobal(udata, dims=dims, extrau=1, dtype=np.float64)
v = fa.fromglobal(vdata, dims=dims, extrav=1, dtype=np.float64)
u *= smw
v *= sms
exch.uv(u.F, v.F)

# interpolate to grid cell center
uc = u[..., :, :-1] + u[..., :, 1:]
vc = v[..., :-1, :] + v[..., 1:, :]
uc[smc] /= (smw[..., :, :-1] + smw[..., :, 1:])[smc]
vc[smc] /= (sms[..., :-1, :] + sms[..., 1:, :])[smc]

# rotate
uE = cs*uc - sn*vc
vN = sn*uc + cs*vc
uE.toglobal().astype('>f4').tofile(uefile)
vN.toglobal().astype('>f4').tofile(vnfile)

