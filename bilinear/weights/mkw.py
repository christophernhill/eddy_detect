#!/usr/bin/env python
'''
Usage: ./mkw.py <fxyfile>
'''
import sys
import os
import numpy as np
from num import loadbin, savebin

ncs = 510

fxyname, = sys.argv[1:]
idir, name = os.path.split(fxyname)
sfx = name.replace('fxy_', '')

f,x,y = loadbin(fxyname).transpose(2, 0, 1)
f = f.astype(int)

i,X = divmod(ncs*x+.5, 1.)
j,Y = divmod(ncs*y+.5, 1.)
i = i.astype(int)
j = j.astype(int)

w = np.zeros((2,2)+f.shape)
for iw,(ii,jj,ff,xx,yy) in enumerate(zip(i.flat, j.flat, f.flat, X.flat, Y.flat)):
    w[0,0].flat[iw] = (1-xx)*(1-yy)
    w[0,1].flat[iw] = (xx)*(1-yy)
    w[1,0].flat[iw] = (1-xx)*(yy)
    w[1,1].flat[iw] = (xx)*(yy)

wtot = w[0,0] + w[0,1] + w[1,0] + w[1,1]
w /= wtot

savebin(idir + '/w_' + sfx, w)
savebin(idir + '/fij_' + sfx, np.array([f,j,i]), 'i2')

