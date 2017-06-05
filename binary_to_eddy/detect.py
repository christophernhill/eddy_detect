#!/usr/bin/env python
"""
Usage: 
    ./detect.py <day> <xwidth> <ywidth> <x> <y> 
<day> simulation day of interest
<x> x location of patch to run eddy detection on
<y> y location of path to run eddy detection on
<xwidth> x width of patch
<ywidth> y width of patch

Runs eddy detection on a given patch of velocity data from a simulation day
"""

import sys
import pdb
sys.path.insert(0, '../')
from bilinear.generate_phases import GeneratePhases
from interp.data_interp import Interpolation
from interp.eddy_detection_viz import detect_and_visualize

def from_lat_lng(lat, lng):
    x = float(lat)*4 + 360
    y = -float(lng)*4 + 720
    return (x,y)

def detect(day, x, y, xwidth, ywidth):
    logging.debug("entered #detect() with day = {0}".format(day))
    logging.debug("converting lat lng to grid coordinates")

    #latlng stuff
    x,y = from_lat_lng(x,y)
    xwidth = int(xwidth)
    ywidth = int(ywidth)
    xwidth *= 4
    ywidth *= 4
    pdb.set_trace()

    # Run generate_phases.py on the given day to get a .npy array of the phases
    gen = GeneratePhases()
    phases, etn = gen.generate(day, layers=10)

    # Run data-interp.py on the phase npy file (and over the given patch) to interpolate the phase data to a higher resolution
    interp = Interpolation()
    # TODO: edit data_interp.py so that it doesn't throw away the first element in the parameter array
    interpolation_factor = 2
    parameters = ["placeholder",xwidth, ywidth, x, y, interpolation_factor]
    pdb.set_trace()
    phases_interp = interp.interpolate(parameters, phases)
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
        x =  sys.argv.pop(1)
        y =  sys.argv.pop(1)
        xwidth =  sys.argv.pop(1)
        ywidth =  sys.argv.pop(1)
    except IndexError:
        day = "0000231552" # leading zeros are needed for filename resolution 
        x = 50
        y = 50
        xwidth = 100
        ywidth = 100

    detect(day, x, y, xwidth, ywidth)

