import __builtin__
import operator
import itertools as itools
import numpy as np
import numpy.core.umath as umath
try:
    from fractions import gcd
except ImportError:
    try:
        import numpy.core._internal._gcd as gcd
    except ImportError:
        def gcd(a, b):
            """Calculate the greatest common divisor of a and b"""
            while b:
                a, b = b, a%b
            return a

class _FacetUnaryOperation(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, *args, **kwargs):
        try:
            facets = a.facets
        except AttributeError:
            return self.f(a, *args, **kwargs)
        else:
            return FacetArray( self.f(f, *args, **kwargs) for f in facets )

    def __str__ (self):
        return "FacetArray version of %s" % str(self.f)

class _FacetUnaryOperation2(object):
    " Unary operation with 2 return values "
    def __init__(self, func):
        self.f = func

    def __call__(self, a, *args, **kwargs):
        try:
            facets = a.facets
        except AttributeError:
            return self.f(a, *args, **kwargs)
        else:
            listofpairs = [ self.f(f, *args, **kwargs) for f in facets ]
            facets1,facets2 = zip(*listofpairs)
            return FacetArray( f for f in facets1 ), FacetArray( f for f in facets2 )

    def __str__ (self):
        return "FacetArray version of %s" % str(self.f)

class _FacetBinaryOperation(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, b, *args, **kwargs):
        try:
            afacets = a.facets
        except AttributeError:
            try:
                bfacets = b.facets
            except AttributeError:
                return self.f(a, b, *args, **kwargs)
            else:
                return FacetArray( self.f(a, bf, *args, **kwargs) for bf in bfacets )
        else:
            try:
                bfacets = b.facets
            except AttributeError:
                return FacetArray( self.f(af, b, *args, **kwargs) for af in afacets )
            else:
                return FacetArray( self.f(af, bf, *args, **kwargs) for af,bf in zip(afacets,bfacets) )

    def reduce(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.reduce(target,axis,dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None or dtype == facets[0].dtype:
                    res = facets[0].copy()
                else:
                    res = facets[0].astype(dtype)

                for f in facets[1:]:
                    res[:] = self.f(res, f)

                return res
            else:
                if axis > 0:
                    axis -= 1
                return FacetArray( self.f.reduce(f, axis, dtype) for f in facets )

    def accumulate(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.accumulate(target, axis, dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None:
                    dtype = facets[0].dtype

                tmp = facets[0].astype(dtype)
                res = [ tmp ]
                for f in facets[1:]:
                    tmp = self.f(tmp, f)
                    res.append( tmp )

                return FacetArray(res)
            else:
                if axis > 0:
                    axis -= 1
                return FacetArray( self.f.accumulate(f, axis, dtype) for f in facets )

    def outer (self, a, b):
        raise NotImplementedError('outer product operations for FacetArray')

    def __str__ (self):
        return "FacetArray version of %s" % str(self.f)

class _FacetBinaryOperation2(object):
    def __init__(self, func):
        self.f = func

    def __call__(self, a, b, *args, **kwargs):
        try:
            afacets = a.facets
        except AttributeError:
            try:
                bfacets = b.facets
            except AttributeError:
                return self.f(a, b, *args, **kwargs)
            else:
                listofpairs = [ self.f(a, bf, *args, **kwargs) for bf in bfacets ]
                facets1,facets2 = zip(*listofpairs)
                return FacetArray( f for f in facets1 ), FacetArray( f for f in facets2 )
        else:
            try:
                bfacets = b.facets
            except AttributeError:
                listofpairs = [ self.f(af, b, *args, **kwargs) for af in afacets ]
                facets1,facets2 = zip(*listofpairs)
                return FacetArray( f for f in facets1 ), FacetArray( f for f in facets2 )
            else:
                listofpairs = [ self.f(af, bf, *args, **kwargs) for af,bf in zip(afacets,bfacets) ]
                facets1,facets2 = zip(*listofpairs)
                return FacetArray( f for f in facets1 ), FacetArray( f for f in facets2 )

    def reduce(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.reduce(target,axis,dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None or dtype == facets[0].dtype:
                    res = facets[0].copy()
                else:
                    res = facets[0].astype(dtype)

                for f in facets[1:]:
                    res[:] = self.f(res, f)

                return res
            else:
                if axis > 0:
                    axis -= 1
                return FacetArray( self.f.reduce(f, axis, dtype) for f in facets )

    def accumulate(self, target, axis=None, dtype=None):
        try:
            facets = target.facets
        except AttributeError:
            if axis is None:
                axis = 0
            return self.f.accumulate(target, axis, dtype)
        else:
            if axis is None:
                axis = 1
            if axis in [0,-len(facets)]:
                if dtype is None:
                    dtype = facets[0].dtype

                tmp = facets[0].astype(dtype)
                res = [ tmp ]
                for f in facets[1:]:
                    tmp = self.f(tmp, f)
                    res.append( tmp )

                return FacetArray(res)
            else:
                if axis > 0:
                    axis -= 1
                return FacetArray( self.f.accumulate(f, axis, dtype) for f in facets )

    def outer (self, a, b):
        raise NotImplementedError('outer product operations for FacetArray')

    def __str__ (self):
        return "FacetArray version of %s" % str(self.f)


abs = absolute = _FacetUnaryOperation(umath.absolute)
arccos      = _FacetUnaryOperation(umath.arccos)
arccosh     = _FacetUnaryOperation(umath.arccosh)
arcsin      = _FacetUnaryOperation(umath.arcsin)
arcsinh     = _FacetUnaryOperation(umath.arcsinh)
arctan      = _FacetUnaryOperation(umath.arctan)
arctanh     = _FacetUnaryOperation(umath.arctanh)
around      = _FacetUnaryOperation(np.round_)
ceil        = _FacetUnaryOperation(umath.ceil)
conj = conjugate = _FacetUnaryOperation(umath.conjugate)
cos         = _FacetUnaryOperation(umath.cos)
cosh        = _FacetUnaryOperation(umath.cosh)
exp         = _FacetUnaryOperation(umath.exp)
fabs        = _FacetUnaryOperation(umath.fabs)
floor       = _FacetUnaryOperation(umath.floor)
log10       = _FacetUnaryOperation(umath.log10)
log2        = _FacetUnaryOperation(umath.log2)
log         = _FacetUnaryOperation(umath.log)
logical_not = _FacetUnaryOperation(umath.logical_not)
negative    = _FacetUnaryOperation(umath.negative)
sin         = _FacetUnaryOperation(umath.sin)
sinh        = _FacetUnaryOperation(umath.sinh)
sqrt        = _FacetUnaryOperation(umath.sqrt)
tan         = _FacetUnaryOperation(umath.tan)
tan         = _FacetUnaryOperation(umath.tan)
tanh        = _FacetUnaryOperation(umath.tanh)

deg2rad     = _FacetUnaryOperation(umath.deg2rad)
degrees     = _FacetUnaryOperation(umath.degrees)
exp2        = _FacetUnaryOperation(umath.exp2      )
expm1       = _FacetUnaryOperation(umath.expm1     )
invert      = _FacetUnaryOperation(umath.invert    )
isfinite    = _FacetUnaryOperation(umath.isfinite  )
isinf       = _FacetUnaryOperation(umath.isinf     )
isnan       = _FacetUnaryOperation(umath.isnan     )
log1p       = _FacetUnaryOperation(umath.log1p     )
ones_like   = np.ones_like
rad2deg     = _FacetUnaryOperation(umath.rad2deg   )
radians     = _FacetUnaryOperation(umath.radians   )
reciprocal  = _FacetUnaryOperation(umath.reciprocal)
rint        = _FacetUnaryOperation(umath.rint      )
sign        = _FacetUnaryOperation(umath.sign      )
signbit     = _FacetUnaryOperation(umath.signbit   )
spacing     = _FacetUnaryOperation(umath.spacing   )
square      = _FacetUnaryOperation(umath.square    )
trunc       = _FacetUnaryOperation(umath.trunc     )

# 2 return values
frexp       = _FacetUnaryOperation2(umath.frexp     )
modf        = _FacetUnaryOperation2(umath.modf      )
bitwise_not = invert
# Binary ufuncs ...............................................................
add                  = _FacetBinaryOperation(umath.add)
arctan2              = _FacetBinaryOperation(umath.arctan2)
bitwise_and          = _FacetBinaryOperation(umath.bitwise_and)
bitwise_or           = _FacetBinaryOperation(umath.bitwise_or)
bitwise_xor          = _FacetBinaryOperation(umath.bitwise_xor)
divide               = _FacetBinaryOperation(umath.divide)
equal                = _FacetBinaryOperation(umath.equal)
floor_divide         = _FacetBinaryOperation(umath.floor_divide)
fmod                 = _FacetBinaryOperation(umath.fmod)
greater_equal        = _FacetBinaryOperation(umath.greater_equal)
greater              = _FacetBinaryOperation(umath.greater)
hypot                = _FacetBinaryOperation(umath.hypot)
less_equal           = _FacetBinaryOperation(umath.less_equal)
less                 = _FacetBinaryOperation(umath.less)
logical_and          = _FacetBinaryOperation(umath.logical_and)
logical_or           = _FacetBinaryOperation(umath.logical_or)
logical_xor          = _FacetBinaryOperation(umath.logical_xor)
mod                  = _FacetBinaryOperation(umath.mod)
multiply             = _FacetBinaryOperation(umath.multiply)
not_equal            = _FacetBinaryOperation(umath.not_equal)
power                = _FacetBinaryOperation(umath.power)
remainder            = _FacetBinaryOperation(umath.remainder)
subtract             = _FacetBinaryOperation(umath.subtract)
true_divide          = _FacetBinaryOperation(umath.true_divide)

copysign    = _FacetBinaryOperation(umath.copysign   )
fmax        = _FacetBinaryOperation(umath.fmax       )
fmin        = _FacetBinaryOperation(umath.fmin       )
ldexp       = _FacetBinaryOperation(umath.ldexp      )
left_shift  = _FacetBinaryOperation(umath.left_shift )
maximum     = _FacetBinaryOperation(umath.maximum    )
minimum     = _FacetBinaryOperation(umath.minimum    )
nextafter   = _FacetBinaryOperation(umath.nextafter  )
right_shift = _FacetBinaryOperation(umath.right_shift)

divmod = _FacetBinaryOperation2(divmod)

equal.reduce = None
greater_equal.reduce = None
greater.reduce = None
less_equal.reduce = None
less.reduce = None
not_equal.reduce = None
alltrue = logical_and.reduce
sometrue = logical_or.reduce

def max(arr, axis=None, out=None):
    try:
        return arr.max(axis=axis, out=out)
    except AttributeError:
        return np.asanyarray(arr).max(axis=axis, out=out)

def min(arr, axis=None, out=None):
    try:
        return arr.min(axis=axis, out=out)
    except AttributeError:
        return np.asanyarray(arr).min(axis=axis, out=out)

def where(cnd, a=None, b=None):
    if isinstance(cnd, np.ndarray):
        return np.where(cnd, a, b)
    if a is not None:
        raise ImplementationError()
    nd = cnd.ndim
    idx = [[] for _ in range(nd)]
    for f in range(len(cnd)):
        idxf = np.where(cnd[f])
        idx[0].append(len(idxf[0])*[f])
        for i in range(1,nd):
            idx[i].append(idxf[i-1])
    return tuple(np.concatenate(ii) for ii in idx)


def calc_shapes(shape=(), dims=None, halo=0, extrau=0, extrav=0):
    if dims is None:
        # shape is a tuple of dimensions or lists of dimensions
        # simple dimension get repeated for all faces
        nfacet = __builtin__.max( np.iterable(d) and len(d) or 1 for d in shape )
        shape = [ np.iterable(d) and d or nfacet*[d] for d in shape ]
        shape[-1] = [ x+2*halo+extrau for x in shape[-1] ]
        shape[-2] = [ x+2*halo+extrav for x in shape[-2] ]
        shapes = zip(*shape)
    else:
        nxs = dims[0::2]
        nys = dims[1::2]
        shapes = [ shape+(ny+2*halo+extrav,nx+2*halo+extrau) for nx,ny in zip(nxs,nys) ]
    return shapes

def calc_shapes2(shape=(), dims=None, halo=0, extrau=0, extrav=0):
    if dims is None:
        # shape is a tuple of dimensions or lists of dimensions
        # first dimension is number of facets
        # simple dimension get repeated for all faces
        nfacet = shape[0]
        shape = [ np.iterable(d) and d or nfacet*[d] for d in shape[1:] ]
        shape[-1] = [ x+2*halo+extrau for x in shape[-1] ]
        shape[-2] = [ x+2*halo+extrav for x in shape[-2] ]
        shapes = zip(*shape)
    else:
        nxs = dims[0::2]
        nys = dims[1::2]
        shapes = [ shape+(ny+2*halo+extrav,nx+2*halo+extrau) for nx,ny in zip(nxs,nys) ]
    return shapes


class Facets(object):
    """Facets: object for indexing a FacetArray over the facet index (and other indices)

    see FacetArray for details.

    """
    def __init__(self, facetarray):
        self.fa = facetarray

    def __getitem__(self, indx):
        if type(indx) == type(()):
            if indx[0] is Ellipsis:
                indx = (self.ndim-len(indx)+1+indx.count(None))*np.s_[:,] + indx[1:]
            if np.iterable(indx[0]):
                # advanced indexing (only pure for now)
                #a = np.broadcast_arrays(*[a for a in indx if np.iterable(a)])
                a = np.broadcast_arrays(*indx)
                res = np.zeros(a[0].shape, self.fa.dtype)
                for f in range(self.fa.nfacet):
                    msk = a[0] == f
                    b = tuple(x[msk] for x in a[1:])
                    res[msk] = self.fa.facets[f][b]
                return res
            else:
                facets = self.fa.facets[indx[0]]
                if type(facets) == type(()):
                    return FacetArray( a[indx[1:]] for a in facets )
                else:
                    return facets[indx[1:]]
        else:
            if indx is Ellipsis:
                indx = np.s_[:]
            facets = self.fa.facets[indx]
            ## if indx selects several facets, turn into FacetArray
            if type(facets) == type(()):
                return FacetArray(facets)
            else:
                return facets

    def __getslice__(self,i,j):
        return FacetArray(self.fa.facets[i:j])

    def __setitem__(self, indx, val):
        if type(indx) != type(()):
            indx = (indx,)

        if indx[0] is Ellipsis:
            indx = (self.ndim-len(indx)+1+indx.count(None))*np.s_[:,] + indx[1:]

        facets = self.fa.facets[indx[0]]
        if type(facets) == type(()):
            if hasattr(val,'facets'):
                assert len(facets) == len(val.facets)
                for i,facet in enumerate(facets):
                    facet[indx[1:]] = val[i]
            else:
                for facet in facets:
                    facet[indx[1:]] = val
        else:
            # single facet
            facets[indx[1:]] = val

    def __setslice__(self, i, j, y):
        return self.__setitem__(slice(i, j) ,y)

    def __len__(self):
        return len(self.fa.facets)

    @property
    def ndim(self):
        return self.fa.ndim + 1

    @property
    def shape(self):
        return (self.fa.nfacet,) + self.fa.shape

    def __repr__(self):
        return 'Facets(' + ',\n\n       '.join( '(' + str(f).replace('\n','\n       ') + ",dtype='" + str(f.dtype) + "')" for f in self.fa.facets ) + ')'

    def __str__(self):
        return '[' + '\n\n '.join( f.__str__().replace('\n','\n ') for f in self.fa.facets ) + ']'


class FacetArray(object):
    """
    an array with multiple facets (e.g. cubed-sphere faces).

    To create:

    1. from arrays for facets:

        f1 = [[1., 2.], [3., 4.]]
        f2 = [[5., 6.], [7., 8.]]
        a = FacetArray([f1, f2])   # 2 facets, each 2x2

    2. empty with given shape:

        u = FacetArray.empty((10,), 'f', dims=6*[510, 510], halo=1, extrau=1)

       creates a facet array with 6 facets of shape (10, 512, 513) (to be
       interpreted as 510x510 with a halo of 1 and an extra row in x appropriate
       for velocity along x).

    3. zeros:

        u = FacetArray.zeros((10,), 'f', dims=6*[510, 510], halo=1, extrau=1)

    4. from a global array (using one of MITgcm/exch2's maps):

        arr = np.zeros((10, 510, 3060))
        a = FacetArray.fromglobal(arr, dims=6*[510, 510], map=-1)

    5. from a binary file:

        a = FacetArray.fromfile('THETA.data', '>f4', (50,), dims=6*[510, 510])

    Facet arrays can mostly be used like normal numpy arrays.  Indices apply
    to all facets.  Use negative indices to refer to slices near the upper boundaries, e.g.

        b = a[..., :-1]

    drops the last row in x from all facets of a.

    Use the .F member to select facets, e.g.,

        f0 = a.F[0]

    returns the first facet as a regular numpy array.

    Selecting multiple facets gives a new FacetArray,

        ae = a.F[::2]

    Other dimension can be indexed at the same time as the facet index,

        b = a.F[::2, ..., :-1]

    gives a FacetArray of every even facet of a  with the last row in x dropped
    from all facets.

    """
    __array_priority__ = 20

    def __init__(self, arrs, masks=None):
        if isinstance(arrs, FacetArray):
            arrs = arrs.F
        if masks is not None:
            self.facets = tuple(np.ma.MaskedArray(arr, mask) for arr,mask in zip(arrs,masks))
        else:
            self.facets = tuple(np.asanyarray(arr) for arr in arrs)
        self.F = Facets(self)
        self.f = self.F
        if self.facets:
            self.dtype = self.facets[0].dtype
            for f in self.facets[1:]:
                if f.dtype != self.dtype:
                    raise TypeError('FacetArray must have same dtype')

    @classmethod
    def empty(cls, shape=(), dtype=None, dims=None, halo=0, extrau=0, extrav=0):
        shapes = calc_shapes2(shape, dims, halo, extrau, extrav)
        return cls( np.empty(sh, dtype) for sh in shapes )

    @classmethod
    def zeros(cls, shape=(), dtype=None, dims=None, halo=0, extrau=0, extrav=0, mask=None):
        shapes = calc_shapes2(shape, dims, halo, extrau, extrav)
        if mask is None:
            zeros = np.zeros
        else:
            zeros = np.ma.zeros
        obj = cls( zeros(sh, dtype) for sh in shapes )
        if mask:
            obj.mask = True
        return obj

    @classmethod
    def fromglobal(cls, arr, shape=None, dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1, dtype=None):
        if not (halo or extrau or extrav) and dims is None and missing is None and dtype is None:
            return view(arr, shape, map)
        arr = np.asanyarray(arr)
        if shape is None and dims is not None:
            shape = arr.shape[:-2]
        if dtype is None:
            dtype = arr.dtype
        hasmask = hasattr(arr, 'mask') or None
        obj = cls.zeros(shape, dtype, dims, halo, extrau, extrav, mask=hasmask)
        obj.set(arr, map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    @classmethod
    def fromfile(cls, fname, dtype, shape=(), dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1):
        if not (halo or extrau or extrav) and dims is None and missing is None:
            return view(np.fromfile(fname, dtype), shape, map)
        obj = cls.zeros(shape, dtype, dims, halo, extrau, extrav, missing)
        obj.set(np.fromfile(fname, dtype), map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    @classmethod
    def frombin(cls, fname, dims=None, halo=0, extrau=0, extrav=0, missing=None, map=-1):
        from oj.num import loadbin
        data = loadbin(fname)
        dtype = data.dtype
        shape = data.shape[:-2]
        if not (halo or extrau or extrav) and dims is None and missing is None:
            return view(data, shape, map)
        obj = cls.zeros(shape, dtype, dims, halo, extrau, extrav, missing)
        obj.set(data, map, halo, extrau, extrav)
        if missing is not None:
            obj.mask = obj == missing
        return obj

    def copy(self):
        return self.__class__( f.copy() for f in self.facets )

    @property
    def size(self):
        return sum(f.size for f in self.facets)

    def contiguize(self):
        data = np.zeros(self.size, self.dtype)
        facets = []
        off = 0
        for f in range(self.nfacet):
            sz = self.facets[f].size
            data[off:off+sz] = self.facets[f].flat
            facets.append(data[off:off+sz])
            off += sz
        return self.__class__(facets)

    @property
    def data(self):
        return FacetArray(f.data for f in self.facets)

    @data.setter
    def data(self, value):
        if hasattr(value, 'facets'):
            for f in range(self.nfacet):
                self.facets[f].data[...] = value.facets[f]
        else:
            for f in self.facets:
                f.data[...] = value

    @property
    def mask(self):
        return FacetArray(f.mask for f in self.facets)

    @mask.setter
    def mask(self, value):
        if hasattr(value, 'facets'):
            for f in range(self.nfacet):
                self.facets[f].mask = value.facets[f]
        else:
            for f in self.facets:
                f.mask = value

    def filled(self, fill_value=None):
        return self.__class__( f.filled(fill_value) for f in self.facets )

    def astype(self, newtype):
        """
        Returns a copy of the FacetArray cast to given newtype.

        Returns
        -------
        output : FacetArray
            A copy of self cast to input newtype.
            The returned record shape matches self.shape.

        """
        newtype = np.dtype(newtype)
        output = self.__class__( f.astype(newtype) for f in self.facets )
        return output

    def __getitem__(self, indx):
        if hasattr(indx,'facets'):
            if indx.facets[0].dtype == bool:
                return FacetArray( f[i] for f,i in zip(self.facets,indx.facets) )
            else:
                return NotImplemented
        else:
            return FacetArray( a[indx] for a in self.facets )

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def __setitem__(self, indx, val):
        if type(indx) != type(()):
            indx = (indx,)

        if isinstance(indx[0], FacetArray) and indx[0].dtype == bool:
            findx = indx[0]
            pindx = indx[1:]
            # boolean advanced indexing
            if np.ndim(val) == 0:
                for f in range(self.nfacet):
                    self.facets[f][(findx.facets[f],)+pindx] = val
            elif isinstance(val, FacetArray):
                for f in range(self.nfacet):
                    self.facets[f][(findx.facets[f],)+pindx] = val.facets[f]
            else:
                off = 0
                for f in range(self.nfacet):
                    n = self.facets[f][(findx.facets[f],)+pindx].size
                    self.facets[f][(findx.facets[f],)+pindx] = val[off:off+n]
                    off += n
            return

        l = []
        for i in indx:
            if hasattr(i, 'facets'):
                l.append(i.F)
            else:
                l.append(self.nfacet*[i])

        indxs = zip(*l)

        if hasattr(val,'facets'):
            assert self.nfacet == val.nfacet
            for f in range(self.nfacet):
                self.facets[f][indx] = val.facets[f]
        else:
            for facet, i in zip(self.facets, indxs):
                facet[i] = val

    def __setslice__(self, i, j, y):
        return self.__setitem__(slice(i, j) ,y)

    @property
    def nfacet(self):
        return len(self.facets)

    def __len__(self):
        return len(self.facets)

    @property
    def ndim(self):
        try:
            facet = self.facets[0]
        except IndexError:
            return 0

        return facet.ndim

    @property
    def shapes(self):
        return [ f.shape for f in self.facets ]

    @property
    def dims(self):
        return [x for shape in self.shapes for x in shape[:-3:-1]]

    @property
    def shape(self):
        dims = zip(*self.shapes)
        return tuple( np.std(d) and d or d[0] for d in dims )

#    @property
#    def flat(self):
#        return itools.chain(*(f.flat for f in self.facets))

#    @flat.setter
#    def flat(self, val):
#        val = iter(val)
#        itools.chain(*(f.flat for f in self.facets)) = val

    def set(self, arr, map=-1, halo=0, extrau=0, extrav=0):
        arr = np.asanyarray(arr)

        # compute shape of and slices into global array
        gshape,slices = globalmap(self.shapes, map, halo, extrau, extrav)

        if arr.ndim == 1:
            try:
                arr = arr.reshape(gshape)
            except ValueError:
                raise ValueError('Cannot reshape array of {} elements to shape {}'.format(arr.size, gshape))
        else:
            # check for correct shape
            if map > 0:
                # stacked (and folded) in y -- flatten last 2 to make equiv to map==0
                arr = arr.reshape(arr.shape[:-2]+(-1,))
            if arr.shape != gshape:
                raise ValueError('Unexpected shape: ' + str(arr.shape) + ' ' + str(gshape))
        for i,s in enumerate(slices):
            ny,nx = self.facets[i].shape[-2:]
            #nyi = ny-2*halo-extrav
            #nxi = nx-2*halo-extrau
            #self[i][..., halo:ny-halo-extrav, halo:nx-halo-extrau].flat = arr[s].reshape(nyi*nxi)
            self.facets[i][..., halo:ny-halo-extrav, halo:nx-halo-extrau] = arr[s]

    def setfromfile(self, fname, dtype, map=-1, halo=0, extrau=0, extrav=0, offset=None):
        """set facet array from a global file

        if offset is given, start reading at this byte offset in the file
        and read only as much data as required

        """
        if offset is not None:
            gshape,slices = globalmap(self.shapes, map, halo, extrau, extrav)
            count = reduce(operator.mul, gshape)
            with open(fname) as f:
                f.seek(offset)
                data = np.fromfile(f, dtype, count=count)
                self.set(data, map, halo, extrau, extrav)
        else:
            self.set(np.fromfile(fname, dtype), map, halo, extrau, extrav)

    def __repr__(self):
        #return '<FacetArray(' + ', '.join( str(f.shape) for f in self.facets ) + ') at 0x{0:x}>'.format(id(self))
#        return 'FacetArray(' + ',\n\n       '.join( str(f).replace('\n','\n       ') for f in self.facets ) + ')'
        return 'FacetArray(' + ',\n\n       '.join( '(' + str(f).replace('\n','\n       ') + ",dtype='" + str(f.dtype) + "')" for f in self.facets ) + ')'

    def __str__(self):
        return '[' + '\n\n '.join( f.__str__().replace('\n','\n ') for f in self.facets ) + ']'

    def max(self, axis=None, out=None, fill_value=None):
        if hasattr(self, 'mask'):
            if fill_value is None:
                fill_value = np.ma.maximum_fill_value(self)
            arr = self.filled(fill_value)
        else:
            arr = self
        if axis is None:
            res = np.max([ np.max(f) for f in arr.facets ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(arr.facets) ]:
            # won't work unless facets all have same shape
            return np.maximum.reduce(arr.facets, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return FacetArray( f.max(axis=axis) for f in arr.facets )
            else:
                for i,f in enumerate(arr.facets):
                    out[i] = f.max(axis=axis)
                return out

    def min(self, axis=None, out=None, fill_value=None):
        if hasattr(self, 'mask'):
            if fill_value is None:
                fill_value = np.ma.minimum_fill_value(self)
            arr = self.filled(fill_value)
        else:
            arr = self
        if axis is None:
            res = np.min([ np.min(f) for f in arr.facets ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(arr.facets) ]:
            # won't work unless facets all have same shape
            return np.minimum.reduce(arr.facets, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return FacetArray( f.min(axis=axis) for f in arr.facets )
            else:
                for i,f in enumerate(arr.facets):
                    out[i] = f.min(axis=axis)
                return out

    def sum(self, axis=None, dtype=None, out=None):
        if axis is None:
            res = np.sum([ np.sum(f, dtype=dtype) for f in self.facets ])
            if out is not None:
                out[:] = res
            return res
        elif axis in [0, -len(self.facets) ]:
            # won't work unless facets all have same shape
            return np.add.reduce(self.facets, dtype=dtype, out=out)
        else:
            if axis > 0:
                axis = axis - 1
            if out is None:
                return FacetArray( f.sum(axis=axis, dtype=dtype) for f in self.facets )
            else:
                for i,f in enumerate(self.facets):
                    out[i] = f.sum(dtype=dtype, axis=axis)
                return out

    def __abs__(self): return abs(self)

    def __add__(self, other):
        "Add other to self."
        return add(self, other)
    #
    def __radd__(self, other):
        "Add self to other."
        return add(other, self)
    #
    def __sub__(self, other):
        "Subtract other from self."
        return subtract(self, other)
    #
    def __rsub__(self, other):
        "Subtract self from other."
        return subtract(other, self)
    #
    def __mul__(self, other):
        "Multiply self by other."
        return multiply(self, other)
    #
    def __rmul__(self, other):
        "Multiply other by self."
        return multiply(other, self)
    #
    def __div__(self, other):
        "Divide self by other."
        return divide(self, other)
    #
    def __rdiv__(self, other):
        "Divide other by self."
        return divide(other, self)
    #
    def __truediv__(self, other):
        "Divide self by other."
        return true_divide(self, other)
    #
    def __rtruediv__(self, other):
        "Divide other by self."
        return true_divide(other, self)
    #
    def __floordiv__(self, other):
        "Divide self by other."
        return floor_divide(self, other)
    #
    def __rfloordiv__(self, other):
        "Divide other by self."
        return floor_divide(other, self)
    #
    def __divmod__(self, other):
        "divmod self by other."
        return divmod(self, other)
    #
    def __rdivmod__(self, other):
        "divmod other by self."
        return divmod(other, self)
    #
    def __pow__(self, other):
        "Raise self to the power other."
        return power(self, other)
    #
    def __rpow__(self, other):
        "Raise other to the power self."
        return power(other, self)
    #
    def __eq__(self, other): return equal(self, other)
    def __ne__(self, other): return not_equal(self, other)
    def __lt__(self, other): return less (self, other)
    def __gt__(self, other): return greater(self, other)
    def __le__(self, other): return less_equal(self, other)
    def __ge__(self, other): return greater_equal(self, other)
    def __and__(self, other): return bitwise_and(self, other)
    def __rand__(self, other): return bitwise_and(self, other)
    def __or__(self, other): return bitwise_or(self, other)
    def __ror__(self, other): return bitwise_or(self, other)
    def __xor__(self, other): return bitwise_xor(self, other)
    def __rxor__(self, other): return bitwise_xor(self, other)
    def __mod__(self, other): return mod(self, other)
    def __rmod__(self, other): return mod(other, self)
    def __pos__(self): return self
    def __neg__(self): return negative(self)
    def __invert__(self): return invert(self)
    ## ............................................
    #
    def __iadd__(self, other):
        "Add other to self in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__iadd__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__iadd__(of)
        return self
    #
    def __isub__(self, other):
        "Subtract other from self in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__isub__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__isub__(of)
        return self
    #
    def __imul__(self, other):
        "Multiply self by other in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__imul__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__imul__(of)
        return self
    #
    def __idiv__(self, other):
        "Divide self by other in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__idiv__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__idiv__(of)
        return self
    #
    def __ifloordiv__(self, other):
        "Floor divide self by other in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__ifloordiv__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__ifloordiv__(of)
        return self
    #
    def __itruediv__(self, other):
        "True divide self by other in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__itruediv__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__itruediv__(of)
        return self
    #
    def __ipow__(self, other):
        "Raise self to the power other, in place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__ipow__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__ipow__(of)
        return self
    #
    def __iand__(self, other):
        "And in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__iand__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__iand__(of)
        return self
    #
    def __ior__(self, other):
        "Or in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__ior__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__ior__(of)
        return self
    #
    def __ixor__(self, other):
        "Xor in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__ixor__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__ixor__(of)
        return self
    #
    def __imod__(self, other):
        "Mod in-place."
        try:
            facets = other.facets
        except AttributeError:
            for f in self.facets:
                f.__imod__(other)
        else:
            for f,of in zip(self.facets, facets):
                f.__imod__(of)
        return self

    def addhalo(self,extra=[]):
        try:
            iter(extra)
        except TypeError:
            # if just a number, assume equal halos in last 2 dimensions
            extra = 2*[extra]

        extra = np.r_[(self.ndim-len(extra))*[0],  extra]
        res = self.__class__( np.ndarray.__new__(f.__class__, sh+2*extra, f.dtype) for sh,f in zip(self.shapes,self.facets) )

        s = tuple( e and np.s_[e:-e] or np.s_[:] for e in extra )
        for i in range(self.nfacet):
            res.facets[i][s] = self.facets[i]

        return res

    def toglobal(self,out=None,dtype=None,map=-1,halo=0,extrau=0,extrav=0):
        if halo or extrau or extrav:
            self = self[..., halo:(-halo-extrav or None), halo:(-halo-extrau or None)]

        if dtype is None:
            dtype = self.facets[0].dtype

        if map == -1:
            # stacked in x
            nflatdim = 1
        else:
            # facets concatenated
            nflatdim = 2

        facetshapes = self.shapes
        dimlists = zip(*facetshapes)
        gshape,slices = globalmap(facetshapes,map)
        if out is None:
            res = np.zeros(gshape,dtype).view(self.facets[0].__class__)
        else:
            if out.ndim == 1:
                res = out.reshape(gshape)
            else:
                if map == 0:
                    res = out
                else:
                    # stacked (and folded) in y -- flatten 2d
                    res = out.reshape(arr.shape[:-2]+(-1,))

                if res.shape != gshape:
                    print maxdims
                    print flatdims
                    print flatbounds
                    raise ValueError('Unexpected shape: ' + str(res.shape) + ' ' + str(gshape))

        for s,f in zip(slices,self.facets):
            if map >= 0:
                f = f.reshape(f.shape[:-2] + (-1,))

            try:
                res[s] = f
            except ValueError:
                raise ValueError('shape mismatch: ' + str(res[s].shape) + ' ' + str(f.shape))

        if map > 0:
            nx = reduce(gcd, dimlists[-1])
            res = res.reshape(res.shape[:-1] + (-1,nx))

        return res

    def transpose(self, *axes):
        return FacetArray([f.transpose(*axes) for f in self.facets])


def globalmap(facetshapes, map=-1, halo=0, extrau=0, extrav=0):
    if map == -1:
        # stacked in x
        nflatdim = 1
    else:
        # facets concatenated
        nflatdim = 2

    if halo > 0 or extrau > 0 or extrav > 0:
        ndim = len(facetshapes[0])
        halo0 = (ndim-2)*[0] + [halo, halo]
        halo1 = (ndim-2)*[0] + [halo+extrav, halo+extrau]
        facetshapes = [
                [ d-h0-h1 for d,h0,h1 in zip(shape, halo0, halo1) ]
                for shape in facetshapes ]

    dimlists = zip(*facetshapes)
    maxdims = tuple( max(d) for d in dimlists[:-nflatdim] )
    flatdims = [ int(np.prod(sh[-nflatdim:])) for sh in facetshapes ]
    flatbounds = [0] + list(np.cumsum(flatdims))
    gshape = maxdims + (flatbounds[-1],)

#    starts = [ (ndim-nflatdim)*(0,) + (s,) for s in flatbounds[:-1] ]
#    ends = [ sh[:-nflatdim] + (e,) for sh,e in zip(facetshapes,flatbounds[1:]) ]
#    facetslices = [ tuple( np.s_[s:e] for s,e in zip(ss,ee) ) for ss,ee in zip(starts,ends) ]
    slices = [ tuple( np.s_[0:d] for d in sh[:-nflatdim] ) + np.s_[s:e,]
               for sh,s,e in zip(facetshapes,flatbounds[:-1],flatbounds[1:]) ]

    return gshape, slices


array = FacetArray
empty = FacetArray.empty
zeros = FacetArray.zeros
fromfile = FacetArray.fromfile
frombin = FacetArray.frombin
fromglobal = FacetArray.fromglobal
U = _FacetUnaryOperation
U2 = _FacetUnaryOperation2
B = _FacetBinaryOperation
B2 = _FacetBinaryOperation2

def view(arr, shape, map=-1):
    """return a view of a numpy array as a FacetArray array"""
#    dimlists = np.broadcast_arrays(*shape[1:])
#    facetshapes = zip(*dimlists)
#    # compute shape of and slices into global array
    facetshapes = calc_shapes2(shape)
    gshape,slices = globalmap(facetshapes,map)

    if arr.ndim == 1:
        arr = arr.reshape(gshape)
    else:
        # check for correct shape
        if map > 0:
            # stacked (and folded) in y -- flatten last 2 to make equiv to map==0
            arr = arr.reshape(arr.shape[:-2]+(-1,))

        if arr.shape != gshape:
            raise ValueError('Unexpected shape: ' + str(arr.shape) + ' ' + str(gshape))

    self = FacetArray( arr[s] for s in slices )
    return self


def apply(func, *args, **kwargs):
    def mkargs(args, f):
        for arg in args:
            if hasattr(arg, 'facets'):
                yield arg.facets[f]
            else:
                yield arg

    fargs = [ arg for arg in args if hasattr(arg, 'facets') ]
    if len(fargs):
        nfacet = fargs[0].nfacet
        res = [ func(*mkargs(args, f), **kwargs) for f in range(nfacet) ]
        if type(res[0]) == type(()):
            return tuple(FacetArray(arrs) for arrs in zip(*res))
        else:
            try:
                return FacetArray(res)
            except AttributeError:
                return res
    else:
        return func(*args, **kwargs)

def masked(arr, mask):
    return FacetArray(np.ma.MaskedArray(arr[f], mask[f]) for f in range(len(arr)))

def diff(a, n=1, axis=-1):
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
                "order must be non-negative but got " + repr(n))
    if not isinstance(a, FacetArray):
        a = FacetArray(a)
    nd = a.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return diff(a[slice1]-a[slice2], n-1, axis=axis)
    else:
        return a[slice1]-a[slice2]


class MITGrid(object):
    """Example:

    grid = MITGrid('tile{0:03d}.mitgrid', 6*[510, 510])
    xg = grid.xg
    yg = grid.yg
    ac = grid.load('ac')
    az = FacetArray.zeros((), 'f', dims=grid.dims, extrau=1, extrav=1, halo=1)
    grid.set(az, 'az', halo=1)

    """
    _fldnames = ['xc', 'yc', 'dxf', 'dyf', 'ac', 'xg', 'yg', 'dxv', 'dyu', 'az', 'dxc', 'dyc', 'aw', 'as', 'dxg', 'dyg', 'anglecs', 'anglesn']
    _smate    = {'dxc':'dyc', 'dyg':'dxg', 'aw':'as'}
    _wmate    = {'dyc':'dxc', 'dxg':'dyg', 'as':'aw'}
    _zfields  = ['xg','yg','dxv','dyu','az']
    _cfields  = ['xc','yc','dxf','dyf','ac','anglecs','anglesn']
    _end = dict.fromkeys(_cfields, (-1,-1))
    _end.update(dict.fromkeys(_smate, (-1,None)))
    _end.update(dict.fromkeys(_wmate, (None,-1)))
    _end.update(dict.fromkeys(_zfields, (None,None)))
    _extra = dict((k, tuple(e is None and 1 or e+1 for e in ee)) for k,ee in _end.items())

    def __init__(self, files, dims, dtype='>f8'):
        self.files = files
        self.file_dtype = np.dtype(dtype)
        self.dims = dims
        self.nx = dims[0::2]
        self.ny = dims[1::2]
        self.shapes = [(ny+1,nx+1) for nx,ny in zip(self.nx, self.ny)]
        self._count = [(nx+1)*(ny+1) for nx,ny in zip(self.nx, self.ny)]
        self.nfaces = len(self.nx)
        self._fields = dict()

        if len(self.files) != self.nfaces:
            self.files = [ files.format(i+1) for i in range(self.nfaces) ]

    def load(self, name, dtype=float):
        skip = self._fldnames.index(name)
        endy,endx = self._end[name]
        arrs = []
        for f in range(self.nfaces):
            if self.nx[f] > 0 and self.ny[f] > 0:
                count = self._count[f]
                with open(self.files[f]) as fid:
                    fid.seek(skip*count*self.file_dtype.itemsize)
                    arr = np.fromfile(fid, self.file_dtype, count=count)
                try:
                    arr = arr.reshape(self.shapes[f])
                except ValueError:
                    raise IOError("fa.MITGrid: could not read enough data for %s" % name)
                arrs.append(arr[:endy, :endx].astype(dtype))
        return FacetArray(arrs)

    def set(self, farr, name, halo=0):
        skip = self._fldnames.index(name)
        extray, extrax = self._extra[name]
        for f in range(self.nfaces):
            if self.nx[f] > 0 and self.ny[f] > 0:
                count = self._count[f]
                with open(self.files[f]) as fid:
                    fid.seek(skip*count*self.file_dtype.itemsize)
                    arr = np.fromfile(fid, self.file_dtype, count=count)
                try:
                    arr = arr.reshape(self.shapes[f])
                except ValueError:
                    raise IOError("fa.MITGrid: could not read enough data for %s" % name)
                nx = __builtin__.min(farr[f].shape[-1] - 2*halo, self.nx[f] + extrax)
                ny = __builtin__.min(farr[f].shape[-2] - 2*halo, self.ny[f] + extray)
                print f, halo, ny, nx
                farr[f, halo:halo+ny, halo:halo+nx] = arr[:ny, :nx]
 
    def __getattr__(self, name):
        if name.lower() in self._fldnames:
            return self.load(name.lower())
        else:
            raise AttributeError("'MITGrid' object has no attribute '" + name + "'")

