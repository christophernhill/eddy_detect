import numpy as np
import scipy.special as sf

# work around numpy 1.4.0rc1 bug
def pow(x,p):
    return np.exp(p*np.log(x))

#####################################################################
# cs face -> square

_2o3 = 2./3
_tan2o3 = np.tan(2./3)
_r4 = np.sqrt(1j)
_rt3 = np.sqrt(3)
_2rt3 = 2*_rt3
_2prt3 = 2 + _rt3
_2mrt3 = 2 - _rt3
_Agcs = 2*np.sqrt(3j*_rt3)
_Afsquare = sf.gamma(.75)/(sf.gamma(1.25)*sf.gamma(.5))

#def stereo2half(x,y):
#    """ map cs face (symmetric to 0) to half plane
#        (+1+i)*xd -> -1
#        (-1+i)*xd -> 0
#        (-1-i)*xd -> 1
#        (+1-i)*xd -> inf
#        xd = (sqrt(3)-1)/2
#    """
#    z = y-1j*x
#    z2 = z*z
#    app = (2+_rt3+1j*z2)**1.5
#    apm = (2+_rt3-1j*z2)**1.5
#    amp = (2-_rt3+1j*z2)**1.5
#    amm = (2-_rt3-1j*z2)**1.5
#    if -x > y:
#        return 1j*app*amm/(apm*amp+_Agcs*z*(z2*z2-1))
#    else:
#        return 1j*(apm*amp-_Agcs*z*(z2*z2-1))/(app*amm)

def stereo2half(x,y):
    """ map cs face (symmetric to 0) to half plane
        (+1+i)*xd -> inf
        (-1+i)*xd -> -1
        (-1-i)*xd -> 0
        (+1-i)*xd -> 1
        xd = (sqrt(3)-1)/2
    """
    z = y - 1j*x
    z2j = 1j*z*z
    mz4 = z2j*z2j
    appamm = complex((_2prt3+z2j)*(_2mrt3-z2j))**1.5
    apmamp = complex((_2prt3-z2j)*(_2mrt3+z2j))**1.5
#    appamm = (1 - mz4 - _2rt3*z2j)**1.5
#    apmzmp = (1 - mz4 + _2rt3*z2j)**1.5
    if -x > y:
        return 1j*appamm/(apmamp-_Agcs*z*(mz4+1))
    else:
        return 1j*(apmamp+_Agcs*z*(mz4+1))/(appamm)


def half2square(z):
    """ map from half plane to square 0,1,1+i,i using 2F1
        (could use ellipticF)
        maps -1,0,1,inf -> i,0,1,1+i
        don't use near unit circle (away from -1+)!
    """
#    print np.abs(z), np.angle(z)/np.pi*2
    z2 = z*z
    res = 1j**.5*_Afsquare*complex(z/1j)**.5*sf.hyp2f1(.25,.5,1.25,z2)
    x,y = res.real, res.imag
    if z.real < 0 and y < x:
        x,y = y,x
    return x,y


def stereo2square(x,y):
    """ map cs face in stereographic coordinates to square
        (+1+i)*xd -> i
        (-1+i)*xd -> 0
        (-1-i)*xd -> 1
        (+1-i)*xd -> 1+i
        xd = (sqrt(3)-1)/2
    """
    # make sure we stay where half2square is fast and reliable
    if y <= 0.:
        if x <= 0.:
            x,y = half2square(stereo2half(x,y))
            return (x,y)
        else:
            x,y = half2square(stereo2half(-x,y))
            return (1-x,y)
    else:
        if x <= 0.:
            x,y = half2square(stereo2half(x,-y))
            return (x,1-y)
        else:
            x,y = half2square(stereo2half(-x,-y))
            return (1-x,1-y)

#####################################################################

def _tanunscale(t):
    tpm = 2*t-1
    return .5*(1+1.5*np.arctan(tpm*_tan2o3))

tanunscale = np.frompyfunc(_tanunscale,1,1)


def stereo2tansquare(x,y):
    x,y = stereo2square(x,y)
    x = tanunscale(x)
    y = tanunscale(y)
    return x,y

#####################################################################

def stereo2xyz(X,Y):
    """ inverse of stereographic projection: X,Y -> x,y,z """
    r2 = X*X + Y*Y
    f = 1./(1. + r2)
    x = 2*X*f
    y = 2*Y*f
    z = (1.-r2)*f
    # fix normalization, just in case
    r3 = np.sqrt(x*x+y*y+z*z)
    return x/r3, y/r3, z/r3


def xyz2stereo(x,y,z):
    """ stereographic projection: x,y,z -> X,Y """
    f = 1./(1. + z)
    return (f*x,f*y)


def xyz2facestereo(x,y,z):
    """ f,X,Y = xyz2facestereo(x,y,z)
    
find face of point on sphere and project stereographically """
    ax = np.abs(x)
    ay = np.abs(y)
    az = np.abs(z)
    if x >= ay and x >= az and y != x and z != x:
        f = 0
        x,y,z = y,z,x
    elif -x >= ay and -x >= az and y != x and z != x:
        f = 3
        x,y,z = -z,-y,-x
    elif y >= az and z != y:
        f = 1
        x,y,z = -x,z,y
    elif -y >= az and z != y:
        f = 4
        x,y,z = -z,x,-y
    elif z > 0:
        f = 2
        x,y,z = -x,-y,z
    else:
        f = 5
        x,y,z = y,x,-z

    X,Y = xyz2stereo(x,y,z)

    return f,X,Y


def xyz2fxy(x,y,z):
    """ f,zcs = xyz2fxy(x,y,z)

map 3d hom. coordinates on sphere to face number (0,..,5) and x,y in [0,1] on cube """

    f,X,Y = xyz2facestereo(x,y,z)
    X,Y = stereo2tansquare(X,Y)
    return np.asarray(f,int),np.asfarray(X),np.asfarray(Y)


######################################################################

def ll2xyz(lon,lat):
    z = np.sin(np.pi*lat/180)
    r2 = np.cos(np.pi*lat/180)
    x = r2*np.cos(np.pi*lon/180)
    y = r2*np.sin(np.pi*lon/180)
    return x,y,z


def ll2fxy(lon,lat):
    """ f,x,y = ll2fxy(lon,lat)

map lon-lat coordinates to face number (0,..,5) and x,y in [0,1] on cube """
    return xyz2fxy(*ll2xyz(lon,lat))


def ll2jfi(lon,lat,ncs):
    """ j,f,i = ll2jfi(lon,lat)

map lon-lat coordinates to indices j,face,i on ncs cube """
    f,x,y = ll2fxy(lon,lat)
    i = np.clip(np.asarray(np.floor(ncs*x),int), 0,ncs-1)
    j = np.clip(np.asarray(np.floor(ncs*y),int), 0,ncs-1)
    return j,f,i


def ll2csflat(lon,lat,ncs):
    """ csind = ll2csflat(lon,lat)

map lon-lat coordinates to flat index on cube with sides ncs """
    j,f,i = ll2jfi(lon,lat,ncs)
    return (j*6+f)*ncs+i

