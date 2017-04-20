#!/usr/bin/env python
import sys
import numpy as np
from mapcs_float import ll2fxy
from num import savebin

n,dir = sys.argv[1:]
n = int(n)

ncs = 510
llat,llon = np.mgrid[-90+.5/n:90-.5/n:n*180j,-180+.5/n:180-.5/n:n*360j]

csindl = [ll2fxy(lon,lat) for lon,lat in zip(llon.flat,llat.flat)]

csind = np.array(csindl).reshape(n*180,n*360,3)

savebin('{}/fxy_{}x{}'.format(dir,n,n), csind)

