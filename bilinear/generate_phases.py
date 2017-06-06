import math
import numpy as np
from pylab import *
import subprocess
import pdb
import os
import logging

class GeneratePhases:

    # Pulls down the  working directory of the file, used to nail down relative file paths so that they 
    # are relative to the location of generate_phases.py
    filedir = os.path.dirname(os.path.abspath(__file__))
    offline_path = os.path.join(filedir, "../interp/offline")
    logging.debug("offline_path: {0}".format(offline_path))

    """
    Converts a u-vel and v-vel into a phase (radians) value

    @param u: velocity in u direction
    @param v: velocity in v direction
    @return phase: corresponding phase value
    """
    def convert_phase(self, u, v):
        if u > 0 and v >= 0:
            return math.atan(v/u)
        elif u > 0 and v <= 0:
            return math.atan(v/u) + math.pi*2
        elif u == 0 and v > 0:
            return math.pi/2
        elif u == 0 and v < 0:
            return math.pi*3/2
        elif u < 0:
            return math.atan(v/u) + math.pi

    """
    Generates phase data by averaging the top layers of a day's velocity data

    @param day: the day to generate phase data for
    @param layers: the number of top layers to average over
    @return: 2D np array with phase values
    """
    def generate(self, day, layers):
        logging.debug("Entered GeneratePhases#generate() with day = {0}, layers = {1}".format(day, layers))
        logging.debug("offline_path: {0}".format(self.offline_path))
        logging.debug("Current working directory: {0}".format(os.getcwd()))
        logging.debug("generate_phases.py directory: {0}".format(self.filedir))


        cmds = []
        # average top layers of UVEL data
        cmds.append("python {2}/preprocess-ecco-data.py {1}/UVELMASS/day.{0}.data ./avgUVEL-day.{0}.npy".format(day, self.offline_path, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished averaging UVEL")

        # average top layers of VVEL data
        cmds.append("python {2}/preprocess-ecco-data.py {1}/VVELMASS/day.{0}.data ./avgVVEL-day.{0}.npy".format(day, self.offline_path, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished averaging VVEL")

        # convert UVEL and VVEL data to 1440x720 grid
        cmds.append("python {1}/dorotateuv.py ./avgUVEL-day.{0}.npy ./avgVVEL-day.{0}.npy ./UE.{0}.data ./VN.{0}.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dorotateuv.py")

        cmds.append("{1}/dointerpvel.py {1}/interp/ll4x4invconf 4 ./UE.{0}.data ./generate-temp-u.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dointerpvel for UVEL")

        cmds.append("{1}/dointerpvel.py {1}/interp/ll4x4invconf 4 ./VN.{0}.data ./generate-temp-v.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dointerpvel for VVEL")

        # get ETAN data and convert to 1440x720 grid
        etn = np.fromfile("{1}/ETAN/day.{0}.data".format(day, self.offline_path), ">f4")
        np.save("etan.day.{0}.npy".format(day), etn)
        logging.debug("Converted ETAN to npy")
        dorotateuv_cmd = "python {1}/dorotateuv.py etan.day.{0}.npy etan.day.{0}.npy etan-done.{0}.data etan-done.data".format(day, self.filedir)
        subprocess.call(dorotateuv_cmd, shell = True)
        dointerpvel_cmd ="python {1}/dointerpvel.py {1}/interp/ll4x4invconf 4 ./etan-done.{0}.data ./etan-done.{0}.data".format(day, self.filedir)
        subprocess.call(dointerpvel_cmd, shell = True)
        etn = np.fromfile("./etan-done.{0}.data".format(day), ">f4")
        etn = etn.reshape(720, 1440)
        logging.debug("Converted ETAN to 1440x720")

        logging.info("...Finished bilinear interpolation and conversion of binary data into vel npy files (and ETAN)")

        # Pull in the velocity fields we just processed above
        uvel = fromfile("./generate-temp-u.data", ">f4")
        vvel = fromfile("./generate-temp-v.data", ">f4")

        # Reshaping into 1d so we can easily iterate through
        uvel = uvel.reshape(-1)
        vvel = vvel.reshape(-1)

        # Calculating phases from two sets of velocities
        phases = np.array([])
        values = []
        for i in range(len(uvel)):
            val = self.convert_phase(uvel[i], vvel[i])
            values.append(val)

        # Formatting phase data
        phases = np.array(values, dtype = np.float).reshape(720,1440)
        phases = np.nan_to_num(phases)

        logging.info("...Finished computing phase field.")

        return (phases, etn)

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    gen = GeneratePhases()
    out = gen.generate("0000189648", 10)
