import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

x = np.linspace(-1,1,100)
y =  np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)

def f(x, y):
    s = np.hypot(x, y)
    phi = np.arctan2(y, x)
    tau = s + s*(1-s)/5 * np.sin(6*phi) 
    return 5*(1-tau) + tau

T = f(X, Y)
# Choose npts random point from the discrete domain of our model function
npts = 400
px, py = np.random.choice(x, npts), np.random.choice(y, npts)

print px.shape
print py.shape

x2 = np.linspace(-1,1,400)
y2 =  np.linspace(-1,1,400)
X2, Y2 = np.meshgrid(x2,y2)

print X2.reshape(-1)

print T.shape

print X.shape


## fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the model function and the randomly selected sample points
## ax[0,0].contourf(X, Y, T)
## ax[0,0].scatter(px, py, c='k', alpha=0.2, marker='.')
## ax[0,0].set_title('Sample points on f(X,Y)')

# Interpolate using three different methods and plot
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
for i, method in enumerate(('nearest',)):
    Ti = griddata((X.reshape(-1), Y.reshape(-1)), T.reshape(-1), (X2, Y2), method=method)
    r, c = (i+1) // 2, (i+1) % 2
    ## ax[r,c].contourf(X, Y, Ti)
    ## ax[r,c].set_title('method = {}'.format(method))

print X
print X2

print T[10,10]
print Ti[40,40]
print X[10,10]
print X2[40,40]
print Ti.shape

## plt.show()
