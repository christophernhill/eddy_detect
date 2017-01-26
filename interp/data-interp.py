import pylab
from numpy import *
import scipy.interpolate as interp

phases_in = load("final-phases.npy")

dim = 100
x_start = 800
y_start = 800

phases = phases_in[x_start:x_start+dim, y_start:y_start+dim]



sf = 4# scaling factor

x = arange(dim*sf, step=sf)
y = arange(dim*sf, step=sf)

xx, yy = meshgrid(x,y)

points = hstack((xx.reshape(-1, 1), yy.reshape(-1,1)))

values = phases.reshape(-1)

grid_x, grid_y = mgrid[0:dim*sf, 0:dim*sf]

x2 = arange(dim*sf)
y2 = arange(dim*sf)

XX, YY = meshgrid(x2, y2)

xi = hstack((XX.reshape(-1,1), YY.reshape(-1,1)))

grid_x, grid_y = mgrid[0:dim*sf, 0:dim*sf]

#nearest = interp.griddata(points, values, (grid_y, grid_x), method="nearest")
nearest = interp.griddata((xx,yy), values, (grid_y, grid_x), method="nearest")

nearest = nan_to_num(nearest)

linear = interp.griddata(points, values, (grid_y, grid_x), method="linear")
linear = nan_to_num(linear)\

cubic = interp.griddata(points, values, (grid_y, grid_x), method="cubic")
cubic = nan_to_num(cubic)

import matplotlib.pyplot as plt

plt.subplot(221)
plt.plot(points[:,0], points[:,1], 'k.', ms=3)
plt.title("Original")

plt.subplot(222)
plt.imshow(linear.T, extent=(0, dim*sf, 0, dim*sf), origin='lower', interpolation="none")
plt.title('Linear')

plt.subplot(223)
plt.imshow(nearest.T, extent=(0, dim*sf, 0, dim*sf), origin='lower', interpolation="none")
plt.title('Nearest')

plt.subplot(224)
plt.imshow(cubic.T, extent=(0, dim*sf, 0, dim*sf), origin='lower', interpolation="none")
plt.title('Cubic')

plt.show()

save("interped-data.npy", output)

print("finished")
