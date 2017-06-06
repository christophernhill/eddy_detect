#!/usr/bin/env python
"""
Usage: 
    ./detect.py <day> <lat> <long> <latwidth> <longwidth>
<day> simulation day of interest
<lat> latitude location of patch to run eddy detection on (pos = N, neg = S)
<long> longtitude location of path to run eddy detection on (pos = E, neg = W)
<latwidth> longitudinal (in degrees) width of patch
<longwidth> latitudinal (in degrees) width of patch

Runs eddy detection on a given patch of velocity data from a simulation day
"""

import sys
import pdb
import numpy as np
sys.path.insert(0, '../')
from bilinear.generate_phases import GeneratePhases
from interp.data_interp import Interpolation
from interp.eddy_detection_viz import detect_and_visualize
import logging

"""  Constants and other script settings  """
# Denotes how much to increase the resolution from the quarter degree resolution per pixel 
# This was set to be ~20 - based off of eyeballing what edML was trained on, and how big typical eddies appeared in quarter degrees
# This can definitely and should probably be tuned, or at least re-examined #TODO
INTERPOLATION_FACTOR = 20

# Whether or not to tell the script to output more verbose logging statements, or stop to plot the data as it 
# is transformed along the pipeline. Setting this to true will cause pyplot plots to be plotted that block script execution,
# so unless you're testing something, want to see how the data is transformed, or something else, you probably want this to be False.
DEBUG_FLAG = True
LOG_LEVEL = logging.DEBUG


"""
Takes lat lng coordinates (lat: [-90, 90], lng: [-180,180]) and converts to 
the coordinate system used in the phase data. This is a 720x1440 grid, where each pixel is a 
1/4 degree x 1/4 degree, and 0N, 0E is at (360,720)

@param lat: latitude in degrees 
@param lng: longtitude in degrees
@returns: tuple of (x1, x2) coordinates
"""
def from_lat_lng(lat, lng):
    x1 = float(lat)*4 + 360
    x2 = float(lng)*4 + 720
    return (x1,x2)

"""
Identifies any eddies in a window over the phase data for the given day (where day corresponds to the 
10 digit name for simulaiton step. For example "0000231552.data".  

@param day: 10 digit string of integers that denotes the day to pull velocity data from (phase data will be constructed from this)
@param lat: latitude of bottom left corner of window to work on
@param lng: lng of bottom left corner of window to work on
@param latwidth: latitudinal width in degrees of the window
@param lngwidth: longitudinal width in degrees of the window

@returns: 3-tuple of: array of eddy center locations, array of eddy polarities, eddy radii
          Units for these values will be in degrees.

"""
def detect(day, lat, lng, latwidth, lngwidth):
    logging.debug("entered #detect() with day = {0}, x1={1}, x2={2}, x1width={3}, x2width={4}".format(day, lat, lng, latwidth, lngwidth))
    logging.debug("converting lat lng to grid coordinates")

    # Convert input values (in lat/lng) into those used for the phase data
    x1,x2 = from_lat_lng(lat,lng)
    x1width = int(latwidth)
    x2width = int(lngwidth)
    x1width *= 4
    x2width *= 4

    # PREPROCESSING RAW VELOCITY DATA
    # Run generate_phases.py on the given day to get a .npy array of the phases, and average the top 10 layers
    # TODO: The script currently averages the top 10 ocean layers from the raw ECCO data to get the velocity grid,
    # this was rather arbitrary in the hopes of getting a better signal for eddies, and is something you should be aware of
    # when using the script to detect eddies, and when drawing comparisons to other things.
    gen = GeneratePhases()
    phases, etn = gen.generate(day, layers=10)

    # INTERPOLATING PHASE DATA
    # Now that we have the phase data, we need to interpolate it to a higher resolution to actually run through the eddy classifier
    # We interpolate 3 ways: nearest neighbor, linear, and cubic intepolation.
    # Scipy.interpolate#griddata() is the interpolation library
    interp = Interpolation()
    parameters = [x1width, x2width, x1, x2, INTERPOLATION_FACTOR]
    phases_interp = interp.interpolate(parameters, phases, debug=False)
    # ETAN (ssh) data is interpolated as well and passed into the pipeline, but is currently not used (and miiight be slightly broken, but should be easily fixable))
    etn_interp = interp.interpolate(parameters, etn) 

    # DETECTING EDDIES
    # Run eddy_detection_viz.py on the interpolated phase npy file to identify any eddies
    # Uses a wrapped version of Mohammad's edML code and model to classify eddies (pack_edML.py)
    # This wrapped version is has minimal changes from the original edML classifier - just some print statements,
    # and modifications to make it an importable Python module. 
    # Apologies for the messy way that was jammed into the pipeline here - if there is a change to Mohammad's thing, you 
    # should still pretty easily be able to update the version here
    parameters.append(day)
    eddy_centers, eddy_polarity, eddy_radius = detect_and_visualize(phases_interp['cubic'], etn_interp, parameters, False)

    # The returned eddy metrics are given in the units of the interpolated data, so we need to convert it back to quarter degrees
    pdb.set_trace()
    eddy_centers /= (INTERPOLATION_FACTOR * 4)
    eddy_radius /= (INTERPOLATION_FACTOR * 4)

    return eddy_centers, eddy_polarity, eddy_radius

if __name__ == '__main__':
    logging.getLogger().setLevel(LOG_LEVEL)
    logging.info("Started detect.py")
    try:
        day = sys.argv.pop(1)
        x1 =  sys.argv.pop(1)
        x2 =  sys.argv.pop(1)
        x1width =  sys.argv.pop(1)
        x2width =  sys.argv.pop(1)
    except IndexError:
        #sys.exit(__doc__)
        day = "0000231552" # leading zeros are needed for filename resolution 
        x1 = 50
        x2 = 50
        x1width = 100
        x2width = 100

    detect(day, x1, x2, x1width, x2width)

