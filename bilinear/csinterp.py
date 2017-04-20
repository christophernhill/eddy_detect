#!/usr/bin/env python
import os
import numpy as np
from facets import FacetArray
from exchange import Exchange
from num import loadbin

class CSInterp(object):
    def __init__(self, f, j, i, w, ncs):
        self.f = f
        self.i = i
        self.j = j
        self.w = w
        self.ncs = ncs
        self.exch = Exchange.cs()
        # workspace to hold input array
        self.fa = FacetArray.zeros((), dims=12*[ncs], halo=1)

    @classmethod
    def frombin(cls, fname, wname, ncs, dir='', slice=np.s_[:,]):
        f,j,i = loadbin(os.path.join(dir, fname))[np.s_[:,] + slice]
        w = loadbin(os.path.join(dir, wname))[np.s_[:,:] + slice]
        return CSInterp(f, j, i, w, ncs)

    def __call__(self, a):
        if np.ndim(a) > 2:
            sh = a.shape[:-2]
            res = np.zeros(sh + self.f.shape)
            for idx in np.ndindex(sh):
                res[idx] = self.__call__(a[idx])
        else:
            self.fa.set(a, halo=1)
            self.exch(self.fa.F)
            res = np.zeros(self.f.shape)
            for jj in [0, 1]:
                for ii in [0, 1]:
                    res += self.w[jj, ii] * self.fa.F[self.f, self.j + jj, self.i + ii]

        return res
 
    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        f = self.f[idx]
        i = self.i[idx]
        j = self.j[idx]
        w = self.w[(slice(None), slice(None)) + idx]
        return CSInterp(f, j, i, w, self.ncs)

    def sourcemask(self):
        ncs = self.ncs
        msk = np.zeros((ncs,6,ncs), bool)
        msk[self.j-1, self.f, self.i-1] = True
        i = self.i + 1
        j = self.j + 0
        f = self.f + 0
        self.exch.coords(i, j, f, ncs, .5)
        print i.min(), j.min()
        msk[j-1, f, i-1] = True
        j += 1
        self.exch.coords(i, j, f, ncs, .5)
        print i.min(), j.min()
        msk[j-1, f, i-1] = True
        i -= 1
        self.exch.coords(i, j, f, ncs, .5)
        print i.min(), j.min()
        msk[j-1, f, i-1] = True
        return msk.reshape((ncs, 6*ncs))

