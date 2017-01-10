import matplotlib.pyplot as plt
from numpy import *
import scipy.interpolate as interp
import pdb

# takes a numpy array and pads with 0's to make the dimensions square
def pad_to_square(arr):
  diff = arr.shape[0] - arr.shape[1]
  if diff < 0:
    pad_rows = abs(diff)/2
    padding = zeros(pad_rows * arr.shape[1]).reshape(pad_rows, -1)
    arr = vstack((padding, arr, padding))
  elif diff > 0:
    pad_cols = diff/2
    padding = zeros(pad_cols * arr.shape[0]).reshape(-1, pad_cols)
    arr = hstack((padding, arr, padding))
  return arr
  


phases_in = load("final-phases.npy")

phases_in = pad_to_square(phases_in)

#x_dim = 720
#y_dim = 1440

x_dim = 1440
y_dim = 1440

x0 = 00
y0 = 00

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
plt.savefig("1440x1440-orig.png")
print("Saved: 1440x1440-orig.png")

for j, method in enumerate(('nearest', 'linear', 'cubic')):
  i = interp.griddata(points, phases.reshape(-1), interp_points, method=method)
  figure = plt.figure(j+2)
  figure.suptitle(method)
  i = nan_to_num(i)
  i = i.reshape(int(x_dim*scaling_factor),int(y_dim*scaling_factor))
  plt.imshow(i, interpolation="none")
  plt.colorbar()
  plt.savefig("1440x1440-{0}.png".format(method))
  print("Saved: 1440x1440-{0}.png".format(method))

plt.show(block="true")
