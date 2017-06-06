import matplotlib.pyplot as plt
from numpy import *
import scipy.interpolate as interp
import pdb
import sys
import logging

class Interpolation:
    """
    Takes a numpy array and pads with 0's to make the dimensions square

    @returns: tuple of the padded array, and the (x,y) pair representing the adjustment
    to coordinates on the original array
    """ 
    def pad_to_square(self, arr):
      # width = arr.shape[0], height = arr.shape[1]
      diff = arr.shape[0] - arr.shape[1]
      shift = (0,0)
      if diff < 0:
        pad_rows = abs(diff)/2
        padding = zeros(pad_rows * arr.shape[1]).reshape(pad_rows, -1)
        arr = vstack((padding, arr, padding))
        shift = (pad_rows, 0)
      elif diff > 0:
        pad_cols = diff/2
        padding = zeros(pad_cols * arr.shape[0]).reshape(-1, pad_cols)
        arr = hstack((padding, arr, padding))
        shift = (0,pad_cols)
      return (arr, shift)

    """
    This takes y as the first parameter and x as the second, calculates phase

    @returns: DEGREE values
    """
    def convert_phase(self, v, u):
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

    """
    Convert radians to degrres

    @param radians: radians to convert
    @returns: degree value
    """
    def degrees(self, radians):
      return (radians * 360 / (math.pi * 2)) % 360

    """
    Interpolates on a patch of a given 2D float array

    @param parameters: holds lat\lng-idinal width of the path, lat/lng-tidinal location (in 1/4th of degrees) and scaling factor
    @param phases: the underlying 2D data to interpolate
    @param debug: whether to show blocking graphics during script execution
    @raises Exception: when invalid number of parameters
    """
    def interpolate(self, parameters, phases, debug = False):
        if len(parameters) != 5:
            raise Exception("paramters expects 5 elements, not {0}".format(len(parameters)))
        else:
          lat_dim = int(parameters[0])
          lng_dim = int(parameters[1])
          lat = int(parameters[2])
          lng = int(parameters[3])
          # must be float to avoid integer division later
          scaling_factor = float(parameters[4])

        logging.debug("""
        =============
        Running with:
        lat_dim = {0}
        lng_dim = {1}
        lat = {2}
        lng = {3}
        scaling_factor = {4}
        ============""".format(lat_dim, lng_dim, lat, lng, scaling_factor))

        # full data set
        phases_in = phases

        # full data set padded to be a square matrix - this is important, because for some reason,
        # the interpolation function breaks when given non-square matrices
        phases_in, shift  = self.pad_to_square(phases_in)

        # Update coordinates as needed after padding
        lat += shift[0]
        lng += shift[1]

        # Interpolate on just a window of the total dataset
        phases = phases_in[lat:lat+lat_dim,lng:lng+lng_dim]
        print("{0}, {1}, {2}, {3}".format(lat, lng, lat_dim, lng_dim))

        if debug:
            figure0 = plt.figure(0)
            figure0.suptitle("Original")
            plt.imshow(phases_in, origin='lower')
            figure1 = plt.figure(1)
            figure1.suptitle("Patch")
            plt.imshow(phases, origin='lower')
            plt.show()


        # INTERPOLATION
        # Now, instead of interpolating on the phases, split the phases into componenets and interpolate on these
        phase_x = cos(phases)
        phase_y = sin(phases)

        # to hold interpolated component scalars
        x_interp = {}
        y_interp = {}

        for data, output in zip([phase_x, phase_y], [x_interp, y_interp]):
          # data points coordinates to give to griddata function
          x = arange(lat_dim)
          y = arange(lng_dim)
          # just getting it in the the right format (nx2) where each row is a x,y pair
          xx, yy = meshgrid(x,y)
          points = hstack((xx.reshape(-1,1), yy.reshape(-1,1)))

          # points at which to interpolate data
          x1 = arange(lat_dim*scaling_factor)/scaling_factor
          y1 = arange(lng_dim*scaling_factor)/scaling_factor
          # again getting into right format
          xx1, yy1 = meshgrid(x1,y1)
          interp_points = hstack((xx1.reshape(-1,1), yy1.reshape(-1,1)))

          # interpolate data using each of the three available methods
          for j, method in enumerate(('nearest', 'linear', 'cubic')):
            i = interp.griddata(points, data.reshape(-1), interp_points, method=method)
            i = nan_to_num(i)
            i = i.reshape(int(lat_dim*scaling_factor),int(lng_dim*scaling_factor))
            output[method] = i
            logging.debug("Finished {0} interpolation".format(method))


        # vectorized function to convert two component matrices to one phase matrix
        phase_conversion_vectorized = vectorize(self.convert_phase)

        # to hold interpolated data (matrix forms)
        phases_interp = {}

        # display the original data for reference
        if debug:
            figure_orig = plt.figure(0)
            figure_orig.suptitle("Original")
            plt.imshow(phases, interpolation="none")
            plt.colorbar()

        for i, method in enumerate(('nearest', 'linear', 'cubic')):
          phases_interp[method] = phase_conversion_vectorized(y_interp[method], x_interp[method])

          # If Debug flag - display data and write to files
          if debug:
              figure_method = plt.figure(1+i)
              figure_method.suptitle(method)
              plt.imshow(phases_interp[method], interpolation="none")
              plt.colorbar()
              plt.savefig("{0}x{1}-{2}.png".format(int(lat_dim*scaling_factor), int(lng_dim*scaling_factor), method))
              save("{0}x{1}-{2}.npy".format(int(lat_dim*scaling_factor), int(lng_dim*scaling_factor), method), phases_interp[method])
              print("Saved {0}x{1}-{2}.png".format(int(lat_dim*scaling_factor), int(lng_dim*scaling_factor),method)) 
              print("Saved {0}x{1}-{2}.npy".format(int(lat_dim*scaling_factor), int(lng_dim*scaling_factor),method)) 


        # only display blocking plot (meaning script execution will pause until pyplot windows are manually closed) if debug flag is set
        if debug:
            logging.debug("Displaying plots of interpolated data")
            plt.show()

        return phases_interp

if __name__ == "__main__":
    interpolator = Interpolation()
    interpolator.interpolate(sys.argv, np.load("final-phases.npy"))
