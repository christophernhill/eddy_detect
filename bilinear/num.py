import sys
import re
from glob import glob
import numpy as np

_typemap = {'>f4':'R4', '>f8':'R8', '>c8':'C8', '>c16':'C16', '>i4':'I4', '>i2':'I2'}
_invtypemap = dict((v,k) for k, v in _typemap.iteritems())
binpatt = re.compile(r'\.(([<>a-zA-Z0-9:,]*)_)?([-0-9x]*)\.bin$')

def str2type(type):
    if ':' in type:
        type = [ tuple(s.split(':')) for s in type.split(',') ]
        type = np.dtype([(k, str2type(v)) for k,v in type])
    else:
        try:
            type = _invtypemap[type]
        except KeyError:
            m = re.match(r'([A-Z])', type)
            if m:
                l = m.group(1)
                type = re.sub(r'^' + l, '>' + l.lower(), type)
    return type


def type2str(dt):
    dtp = np.dtype(dt)
    if dtp.fields:
        dtypes = ','.join('{0}:{1}'.format(k,type2str(dtp.fields[k][0])) for k in dtp.names)
    else:
        dtypes = str(dt)
        if '>' in dtypes:
            try:
                dtypes = _typemap[dtypes]
            except KeyError:
                m = re.match(r'>([a-z])', dtypes)
                if m:
                    l = m.group(1)
                    dtypes = re.sub(r'>' + l, l.upper(), dtypes)

    return dtypes


def loadbin(f, astype=None):
    '''
    a = loadbin(f, ...)

    Parameters:
    astype     convert to this type
    '''

    # does f have grid info in it already?
    m = binpatt.search(f)
    if m:
        fname = f
    else:
        if not '*' in f and not '?' in f:
            fglob = f + '.[<>a-zA-Z0-9:,]*_*.bin'

        fnames = glob(fglob)
        patt = re.compile(r'\.(([<>a-zA-Z0-9:,]*)_)?([-0-9x]*)\.bin$')
        matches = []
        for fname in fnames:
            if not '*' in f and not '?' in f:
                m1 = patt.match(fname[len(f):])
            else:
                m1 = patt.search(fname)
            if m1:
                matches.append(fname)
                m = m1

        if len(matches) > 1:
            sys.stderr.write('Warning: loadbin: multiple matches for ' + fglob + '\n')
            sys.stderr.write('Warning: loadbin: using ' + matches[-1] + '\n')

        if m:
            fname = matches[-1]

    if not m:
        raise IOError('file not found: ' +  fglob)

    dims = m.group(3)
    if dims:
        dims = [ int(s) for s in dims.split('x')[::-1] ]
    else:
        dims = []
    tp = str2type(m.group(2))
    dtype = np.dtype(tp)

    a = np.fromfile(fname, dtype=dtype)
    try:
        a = a.reshape(dims)
    except ValueError:
        raise IOError('Wrong dimensions for file size: {} {} {}\n'.format(
                           fname, dims, a.size))

    if astype is not None:
        a = a.astype(astype)

    return a


def savebin(f, a, dtype=None):
    a = np.asanyarray(a)
    if dtype is not None:
        try:
            dtype = str2type(dtype)
        except TypeError:
            dtype = np.dtype(dtype)

        a = a.astype(dtype)

    dtypes = type2str(a.dtype)
    shapes = 'x'.join([str(i) for i in a.shape[::-1]])
    fname = '{0}.{1}_{2}.bin'.format(f, dtypes, shapes)
    a.tofile(fname)

