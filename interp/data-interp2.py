import matplotlib.pyplot as plt
from numpy import *
import scipy.interpolate as interp
import pdb
import sys

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
  

# NOTE: This takes y as the first parameter and x as the second
# Returns DEGREE values
def convert_phase(v, u):
  if u > 0 and v >= 0:
      return degrees(math.atan(v/u))
  elif u > 0 and v <= 0:
      return degrees(math.atan(v/u) + math.pi*2)
  elif u == 0 and v > 0:
      return degrees(math.pi/2)
  elif u == 0 and v < 0:
      return degrees(math.pi*3/2)
  elif u < 0:
     return degrees(math.atan(v/u) + math.pi)

# convert radians to degrees
def degrees(radians):
  return (radians * 360 / (math.pi * 2)) % 360
  
if len(sys.argv) != 6:
  x_dim = 100
  y_dim = 100
  x0 = 500
  y0 = 500
  # must be float to avoid integer division later
  scaling_factor = 5.0
else:
  x_dim = int(sys.argv[1])
  y_dim = int(sys.argv[2])
  x0 = int(sys.argv[3])
  y0 = int(sys.argv[4])
  # must be float to avoid integer division later
  scaling_factor = float(sys.argv[5])

print("""
=============
Running with:
x_dim = {0}
y_dim = {1}
x0 = {2}
y0 = {3}
scaling_factor = {4}
============""".format(x_dim, y_dim, x0, y0, scaling_factor))

# full data set 
phases_in = load("final-phases.npy")

# full data set padded to be a square matrix
phases_in = pad_to_square(phases_in)


# interpolate on just a window of the total dataset
phases = phases_in[x0:x0+x_dim,y0:y0+y_dim]

# Now, instead of interpolating on the phases, split the phases into componenets and interpolate on these
phase_x = cos(phases)
phase_y = sin(phases)

# to hold interpolated component scalars
x_interp = {}
y_interp = {}

for data, output in zip([phase_x, phase_y], [x_interp, y_interp]):
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

  # interpolate data using each of the three available methods
  for j, method in enumerate(('nearest', 'linear', 'cubic')):
    i = interp.griddata(points, data.reshape(-1), interp_points, method=method)
    i = nan_to_num(i)
    i = i.reshape(int(x_dim*scaling_factor),int(y_dim*scaling_factor))
    output[method] = i


# vectorized function to convert two component matrices to one phase matrix
phase_conversion_vectorized = vectorize(convert_phase)

# to hold interpolated data (matrix forms)
phases_interp = {}

# display the original data for reference
figure_orig = plt.figure(0)
figure_orig.suptitle("Original")
plt.imshow(phases, interpolation="none")
plt.colorbar()

for i, method in enumerate(('nearest', 'linear', 'cubic')):
  phases_interp[method] = phase_conversion_vectorized(y_interp[method], x_interp[method])

  #display data and write to files
  figure_method = plt.figure(1+i)
  figure_method.suptitle(method)
  plt.imshow(phases_interp[method], interpolation="none")
  plt.colorbar()
  plt.savefig("{0}x{1}-{2}.png".format(int(x_dim*scaling_factor), int(y_dim*scaling_factor), method))
  save("{0}x{1}-{2}.npy".format(int(x_dim*scaling_factor), int(y_dim*scaling_factor), method), phases_interp[method])
  print("Saved {0}x{1}-{2}.png".format(int(x_dim*scaling_factor), int(y_dim*scaling_factor),method)) 
  print("Saved {0}x{1}-{2}.npy".format(int(x_dim*scaling_factor), int(y_dim*scaling_factor),method)) 

plt.show(block="true")

