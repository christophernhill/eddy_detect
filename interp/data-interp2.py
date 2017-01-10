import matplotlib.pyplot as plt
from numpy import *
import scipy.interpolate as interp
import pdb

phases_in = load("final-phases.npy")

#x_dim = 720
#y_dim = 1440

x_dim = 720
y_dim = 1000

x0 = 0
y0 = 0

# NOTE: must be float to avoid integer division later on
scaling_factor = 5.0

# interpolate on just a window of the total dataset
phases = phases_in[x0:x0+x_dim,y0:y0+y_dim]

# data points coordinates to give to griddata function
x = arange(x_dim)
y = arange(y_dim)
# just getting it in the the right format (nx2) where each row is a x,y pair
xx, yy = meshgrid(x,y)
points = hstack((xx.reshape(-1,1), yy.reshape(-1,1)))

# points at which to interpolate data
x1 = arange(x_dim*scaling_factor)/scaling_factor
y1 = arange(y_dim*scaling_factor)/scaling_factor
# again getting into right format
xx1, yy1 = meshgrid(x1,y1)
interp_points = hstack((xx1.reshape(-1,1), yy1.reshape(-1,1)))
  
fig1 = plt.figure(1)
fig1.suptitle("Original")
plt.imshow(phases, interpolation="none")
plt.colorbar()

for j, method in enumerate(('nearest', 'linear', 'cubic')):
  i = interp.griddata(points, phases.reshape(-1), interp_points, method=method)
  figure = plt.figure(j+2)
  figure.suptitle(method)
  i = nan_to_num(i)
  i = i.reshape(int(x_dim*scaling_factor),int(y_dim*scaling_factor))
  plt.imshow(i, interpolation="none")
  plt.colorbar()

plt.show(block="true")
plt.savefig("123.png")
