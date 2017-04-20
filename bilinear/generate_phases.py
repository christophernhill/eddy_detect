import math
import numpy as np
from pylab import *
import subprocess

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
        cmd1 = "./dorotateuv.py {1}/UVELMASS/day.{0}.data {1}/VVELMASS/day.{0}.data UE.{0}.data VN.{0}.data".format(day, self.offline_path)
        cmd2 = "./dointerpvel.py interp/ll4x4invconf 4 UE.{0}.data generate-temp-u.data".format(day)
        cmd3 = "./dointerpvel.py interp/ll4x4invconf 4 VN.{0}.data generate-temp-v.data".format(day)

        subprocess.call([cmd1])
        subprocess.call([cmd2])
        subprocess.call([cmd3])

        uvel = fromfile("generate-temp-u.data", ">f4")
        vvel = fromfile("generate-temp-v.data", ">f4")

        uvel = uvel.reshape(1440, 720)
        vvel = vvel.reshape(1440, 720)

        print(uvel.shape)
        print(vvel.shape)

        uvel = uvel.reshape(-1)
        vvel = vvel.reshape(-1)

        phases = np.array([])
        values = []
        for i in range(len(uvel)):
            val = self.convert_phase(uvel[i], vvel[i])
            values.append(val)

        phases = np.array(values, dtype = np.float).reshape(720,1440)
        phases = np.nan_to_num(phases)

        return phases
