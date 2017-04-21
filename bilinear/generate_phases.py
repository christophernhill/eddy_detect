import math
import numpy as np
from pylab import *
import subprocess
import pdb
import os
import logging

class GeneratePhases:

    # working directory of the file, used to nail down relative file paths so that they 
    # are relative to the location of generate_phases.py
    filedir = os.path.dirname(os.path.abspath(__file__))
    offline_path = os.path.join(filedir, "../interp/offline")
    logging.debug("offline_path: {0}".format(offline_path))

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
    def generate(self, day, layers):
        logging.debug("Entered GeneratePhases#generate() with day = {0}, layers = {1}".format(day, layers))
        # get the UVELMASS and VVELMASS files and convert them using Oliver's methods
        logging.debug("offline_path: {0}".format(self.offline_path))
        logging.debug("Current working directory: {0}".format(os.getcwd()))
        logging.debug("generate_phases.py directory: {0}".format(self.filedir))


        cmds = []
        cmds.append("python {2}/preprocess-ecco-data.py {1}/UVELMASS/day.{0}.data ./avgUVEL-day.{0}.npy".format(day, self.offline_path, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished averaging UVEL")
        cmds.append("python {2}/preprocess-ecco-data.py {1}/VVELMASS/day.{0}.data ./avgVVEL-day.{0}.npy".format(day, self.offline_path, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished averaging VVEL")
        cmds.append("python {1}/dorotateuv.py ./avgUVEL-day.{0}.npy ./avgVVEL-day.{0}.npy ./UE.{0}.data ./VN.{0}.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dorotateuv.py")
        cmds.append("{1}/dointerpvel.py ./interp/ll4x4invconf 4 ./UE.{0}.data ./generate-temp-u.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dointerpvel for UVEL")
        cmds.append("{1}/dointerpvel.py ./interp/ll4x4invconf 4 ./VN.{0}.data ./generate-temp-v.data".format(day, self.filedir))
        subprocess.call(cmds.pop(), shell = True)
        logging.debug("Finished dointerpvel for VVEL")

        #for cmd in cmds:
        # Note on subprocess.call usage: set shell = True, to allow for expansion of '.' to working directory
        #    subprocess.call(cmd, shell=True)

        logging.info("...Finished bilinear interpolation and conversion of binary data into vel npy files")

        uvel = fromfile("./generate-temp-u.data", ">f4")
        vvel = fromfile("./generate-temp-v.data", ">f4")

        uvel = uvel.reshape(1440, 720)
        vvel = vvel.reshape(1440, 720)

        uvel = uvel.reshape(-1)
        vvel = vvel.reshape(-1)

        phases = np.array([])
        values = []
        for i in range(len(uvel)):
            val = self.convert_phase(uvel[i], vvel[i])
            values.append(val)

        phases = np.array(values, dtype = np.float).reshape(720,1440)
        phases = np.nan_to_num(phases)

        logging.info("...Finished computing phase field.")

        return phases

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    gen = GeneratePhases()
    out = gen.generate("0000189648", 10)
    pdb.set_trace()
