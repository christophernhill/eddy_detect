# eddy_detect
Detecting eddies in MITgcm 

# Overview
Provides a pipeline for easily extracting eddy information from an ECCO
UVEL and VVEL dataset.

## How to Run It
You want to run the detect(...) method in binary_to_eddy/detect.py. This
is a module, so you can use:

  <code> from binary_to_eddy.detect import detect </code>
  
and then execute the function simply as:

  <code> detect(day,  lat, lng, latwidth, lngwidth) </code>

This exact same thing is done in test.py, and was verified to work.

Also, you will need to change the 'offline' filepath located in /bilinear/generate_phases.py.
This gives the location to a folder that should contain the raw ECCO data flies of interest.

The script expects the folder to have 3 subfolders: ETAN, UVELMASS, and VVELMASS. ETAN should
hold the sea surface height data, and the other two should have the velocity data (2D).

The files in each should be formatted according to "day.<10 digit day code>.data". Ex:
"day.0000253155.data". 


## The Pipeline
1. Uses Oliver's interpolation scripts (/bilinear) to convert the ECCO
datasets from their original cube-sphere format to a rectangular grid
(1440x720). At this point, each grid element represents a quarter
degree.

2. Velocity data is preprocessed a bit (currently this is implemented as
   averaging the top 10 layers) and then phases are calculated from the
  2D velocity vecotrs.

3. Since Mohammad's edML classifier expects higher resolution than the
   quarter degree resolution we have, we run it through an interpolation
  class. The current scaling factor is 20x.

4. We then feed the scaled up phase data into Mohammad's classifer and
   return the results.

## Future Work
If you find yourself needing to debug or perform future work on the
script, the description here in the README and hopefully well-commmented
code should make it pretty straight forward. 

### Project Structure
Root
  - /bilinear: holds Oliver's code and anything needed to turn raw ECCO
    data into the initial phase data. (Steps 1-2)
  - /interp: interpolation code (Step 3) and eddy detection code (Step
    4)
  - /binary_to_eddy: main script to run (detect.py) that combines
    everything. detect.py is your entry point into everything. 

### TODOs + Things to be aware of
- The 20x scaling factor was picked somewhat arbitrarily and is based
  off the difference between what Oliver's code outputs (quarter degree
resolution) and what Mohammad's edML classifer expects. Whether you
revisit this, tweak it, or not - it's probably a good thing to keep in
mind.
- I tried to make any filepaths relative and relatively robust, but I
  think some things might break if you start calling things from other
places. The fixes should be pretty obvious and easy, but let me know if
anything annoying happens.


