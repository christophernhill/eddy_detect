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

def detect(day, x, y, xwidth, ywidth):

    # Run generate_phases.py on the given day to get a .npy array of the phases


    # Run data-interp.py on the phase npy file (and over the given patch) to interpolate the phase data to a higher resolution


    # Run eddy_detection_viz.py on the interpolated phase npy file to identify any eddies


    # Display and output the eddy detection results (perhaps pull some of this out of eddy_detection_viz.py
    return None

try:
    day = sys.argv.pop(1)
    x =  sys.argv.pop(1)
    y =  sys.argv.pop(1)
    xwidth =  sys.argv.pop(1)
    ywidth =  sys.argv.pop(1)
except IndexError:
    sys.exit(__doc__)

detect(day, x, y, xwidth, ywidth)

