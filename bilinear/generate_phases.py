import math
import numpy as np
from pylab import *
import subprocess
import pdb

class GeneratePhases:

    offline_path = "../interp/offline"

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
        # get the UVELMASS and VVELMASS files and convert them using Oliver's methods
        cmds = []
        cmds.append("python ./preprocess-ecco-data.py {1}/UVELMASS/day.{0}.data ./avgUVEL-day.{0}.npy".format(day, self.offline_path))
        cmds.append("python ./preprocess-ecco-data.py {1}/VVELMASS/day.{0}.data ./avgVVEL-day.{0}.npy".format(day, self.offline_path))
        cmds.append("python ./dorotateuv.py ./avgUVEL-day.{0}.npy ./avgVVEL-day.{0}.npy ./UE.{0}.data ./VN.{0}.data".format(day))
        cmds.append("./dointerpvel.py ./interp/ll4x4invconf 4 ./UE.{0}.data ./generate-temp-u.data".format(day))
        cmds.append("./dointerpvel.py ./interp/ll4x4invconf 4 ./VN.{0}.data ./generate-temp-v.data".format(day))

        for cmd in cmds:
        # Note on subprocess.call usage: set shell = True, to allow for expansion of '.' to working directory
            subprocess.call(cmd, shell=True)

        print("...Finished bilinear interpolation and conversion of binary data into vel npy files")

        uvel = fromfile("generate-temp-u.data", ">f4")
        vvel = fromfile("generate-temp-v.data", ">f4")

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


        print("...Finished computing phase field.")

        return phases

gen = GeneratePhases()
out = gen.generate("0000189648", 10)
pdb.set_trace()
