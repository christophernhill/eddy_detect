import numpy as np
import pack_edML
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

def detect_and_visualize(dataset, etn, params, debug=False):
    classifier = pack_edML.EddyML()

    print("Running data through classifier")

    eddy_centers, eddy_polarity, eddy_radius = classifier.classify(dataset)

    # clean up any previous pyplot windows
    plt.close('all')
    phase_fig = plt.figure(0)
    ax = phase_fig.add_subplot(111)
    phase_fig.suptitle("Phases and detected eddies")

    ax.imshow(dataset, alpha=0.4)

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

    # custom formatter for latlng
    def lat_formatter(x,p):
        degrees = abs(x/(4*params[-1]) -90) + params[1]
        if x > 360:
            return "{0} N".format(degrees)
        elif x < 360:
            return "{0} S".format(degrees)
        else:
            return "0"
    def lng_formatter(x,p):
        degrees = abs(x/(4*params[-1]) -90) + params[2]
        if x > 720:
            return "{0} E".format(degrees)
        elif x < 720:
            return "{0} W".format(degrees)
        else:
            return "0"

    ax.yaxis.set_major_formatter(tick.FuncFormatter(lat_formatter))
    ax.xaxis.set_major_formatter(tick.FuncFormatter(lng_formatter))

    # note coordinates being flipped
    # plot cyclonic eddy structures
    ax.scatter(y_cyc,x_cyc, s = area_cyc, marker='o', facecolor='none', edgecolor='white')
    ax.scatter(y_cyc,x_cyc, s= 20, marker='+')


    # plot anti-cyclonic eddy structures
    ax.scatter(y_anti,x_anti, s = area_anti, marker='o', facecolor='none', edgecolor='black')
    ax.scatter(y_anti,x_anti, s= 20, marker='+')

    ax.legend(['cyclonic radii','cyclonic center', 'anticyclonic radii', 'anticyclonic center'], markerscale=0.1, fontsize = 6, frameon='false')

    # display the ETA values on a separate plot
    #figure_ETA = plt.figure(1)
    #figure_ETA.suptitle("ETA Values")
    #plt.imshow(etn)

    if debug: plt.show(block=True)

    #filename = sys.argv[1].split("/")[-1].split(".")[-2]

    #plt.savefig("detection-test/png/{0}.png".format(filename))
    #np.save("detection-test/npy/{0}.npy".format(filename), dataset)
    #np.save("detection-test/eddy/{0}.npy".format(filename), eddy_info)


    #print("Saved image to detection-test/png/{0}.png".format(filename))
    #print("Saved data to detection-test/npy/{0}.npy".format(filename))
    #print("Saved classified eddy data to detection-test/eddy/{0}.npy".format(filename))

    return (eddy_centers, eddy_polarity, eddy_radius)

if __name__ == "__main__":
    detect_and_visualize(np.load(sys.argv[1]))


