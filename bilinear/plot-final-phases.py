import math
import numpy as np
from pylab import *

def convert_phase(u, v):
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

uvel = fromfile("ll_UE.data", ">f4")
vvel = fromfile("ll_VN.data", ">f4")

uvel = uvel.reshape(1440, 720)
vvel = vvel.reshape(1440, 720)

print(uvel.shape)
print(vvel.shape)

uvel = uvel.reshape(-1)
vvel = vvel.reshape(-1)

phases = np.array([])
values = []
for i in range(len(uvel)):
    val = convert_phase(uvel[i], vvel[i])
    values.append(val)


print(len(values))
phases = np.array(values, dtype = np.float).reshape(720,1440)
phases = np.nan_to_num(phases)

np.save("final-phases.npy", phases)

imshow(phases, origin="lower")
plt.colorbar()

plt.show(block="true")

savefig("final-phases.png", dpi=1000)
