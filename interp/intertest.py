import numpy as np
from pylab import *

data = np.random.rand(10,10)

imshow(data, interpolation="none")
savefig("intertest.png")
