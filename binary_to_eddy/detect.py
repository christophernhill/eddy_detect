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

def detect(day, lat, lng, latwidth, lngwidth):
    logging.debug("entered #detect() with day = {0}, x1={1}, x2={2}, x1width={3}, x2width={4}".format(day, lat, lng, latwidth, lngwidth))
    logging.debug("converting lat lng to grid coordinates")


    #latlng stuff
    x1,x2 = from_lat_lng(lat,lng)
    x1width = int(latwidth)
    x2width = int(lngwidth)
    x1width *= 4
    x2width *= 4

    # Run generate_phases.py on the given day to get a .npy array of the phases
    gen = GeneratePhases()
    phases, etn = gen.generate(day, layers=10)

    # Run data-interp.py on the phase npy file (and over the given patch) to interpolate the phase data to a higher resolution
    interp = Interpolation()
    interpolation_factor = 2
    parameters = [x1width, x2width, x1, x2, interpolation_factor]
    phases_interp = interp.interpolate(parameters, phases, debug=True)
    etn_interp = interp.interpolate(parameters, etn)

    # Run eddy_detection_viz.py on the interpolated phase npy file to identify any eddies
    detect_and_visualize(phases_interp['cubic'], etn_interp, parameters)


    # Display and output the eddy detection results (perhaps pull some of this out of eddy_detection_viz.py
    return None

if __name__ == '__main__':
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Started detect.py")
    try:
        day = sys.argv.pop(1)
        x1 =  sys.argv.pop(1)
        x2 =  sys.argv.pop(1)
        x1width =  sys.argv.pop(1)
        x2width =  sys.argv.pop(1)
    except IndexError:
        sys.exit(__doc__)
        day = "0000231552" # leading zeros are needed for filename resolution 
        x1 = 50
        x2 = 50
        x1width = 100
        x2width = 100

    detect(day, x1, x2, x1width, x2width)

