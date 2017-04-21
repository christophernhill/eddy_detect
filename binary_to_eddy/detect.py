#!/usr/bin/env python
"""
Usage: 
    ./detect.py <day> <x> <y> <xwidth> <ywidth>

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

def detect(day, x, y, xwidth, ywidth):
    logging.debug("entered #detect() with day = {0}".format(day))
    # Run generate_phases.py on the given day to get a .npy array of the phases
    gen = GeneratePhases()
    phases = gen.generate(day, layers=10)

    # Run data-interp.py on the phase npy file (and over the given patch) to interpolate the phase data to a higher resolution
    interp = Interpolation()
    parameters = [x, y, xwidth, ywidth, 5.0]
    phases_interp = interp.interpolate(parameters)

    # Run eddy_detection_viz.py on the interpolated phase npy file to identify any eddies
    detect_and_visualize(phases_interp['cubic'])


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
        x = 100
        y = 100
        xwidth = 500
        ywidth = 500
        #sys.exit(__doc__)

    detect(day, x, y, xwidth, ywidth)

