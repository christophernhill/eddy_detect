import pdb
import numpy as np
import pack_edML
import sys
import math
import matplotlib.pyplot as plt


dataset = np.load(sys.argv[1])

classifier = pack_edML.EddyML()

print("Running data through classifier")

eddy_centers, eddy_polarity, eddy_radius = classifier.classify(dataset)

plt.imshow(dataset, alpha=0.4)

x = eddy_centers[:,0]
y = eddy_centers[:,1]

colors = eddy_polarity

area = eddy_radius**2 * np.pi

eddy_info = np.hstack((x.reshape(-1, 1), y.reshape(-1,1), eddy_radius.reshape(-1,1)))

cyclonic_mask = eddy_polarity == 1
x_cyc = x[cyclonic_mask]
y_cyc = y[cyclonic_mask]
area_cyc = area[cyclonic_mask]

anti_mask = eddy_polarity == -1
x_anti = x[anti_mask]
y_anti = y[anti_mask]
area_anti = area[anti_mask]

# note coordinates being flipped
# plot cyclonic eddy structures
plt.scatter(y_cyc,x_cyc, s = area_cyc, marker='o', facecolor='none', edgecolor='white')
plt.scatter(y_cyc,x_cyc, s= 20, marker='+')


# plot anti-cyclonic eddy structures
plt.scatter(y_anti,x_anti, s = area_anti, marker='o', facecolor='none', edgecolor='black')
plt.scatter(y_anti,x_anti, s= 20, marker='+')

plt.legend(['cyclonic radii','cyclonic center', 'anticyclonic radii', 'anticyclonic center'], markerscale=0.1, fontsize = 6, frameon='false')

plt.show(block=True)

filename = sys.argv[1].split("/")[-1].split(".")[-2]

plt.savefig("detection-test/png/{0}.png".format(filename))
np.save("detection-test/npy/{0}.npy".format(filename), dataset)
np.save("detection-test/eddy/{0}.npy".format(filename), eddy_info)


print("Saved image to detection-test/png/{0}.png".format(filename))
print("Saved data to detection-test/npy/{0}.npy".format(filename))
print("Saved classified eddy data to detection-test/eddy/{0}.npy".format(filename))
plt.show(block="true")


