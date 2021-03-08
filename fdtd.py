import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from scipy.constants import c, mu_0, epsilon_0

# SPECIFY GEOMETRY AND TIME

IMAX = JMAX = KMAX = 10
NMAX = 100

dx = 1e0
dt = dx/(2*c)

pml_width = 10

# SPECIFY SCATTERER AND SOURCE

sig = np.zeros((IMAX, JMAX))
epsilon_r = np.ones((IMAX, JMAX))

f = 1e15 # Hz

# DEFINE CONSTANTS

R = dt/(epsilon_0)
Ra = (c*dt/dx)**2
Rb = dt/(mu_0*dx)
Ca = (1 - R*sig/epsilon_r)/(1+R*sig/epsilon_r)
Cb = Ra/(epsilon_r+R*sig)

#R = Ra = Rb = Ca = Cb = 1

# INITIALIZE FIELDS

xx, yy = np.meshgrid(np.arange(IMAX), np.arange(JMAX))

# only keep prev. two time steps
# shapes are (time, xx, yy, zz, dir)
H = np.zeros((3, IMAX, JMAX, 3))
E = np.zeros((3, IMAX, JMAX, 3))

# INITIALIZE PLOT

fig = plt.figure()
ax = fig.add_subplot(121)#, projection='3d')
#plot = ax.plot_surface(xx, yy, Ez[0])
plot = ax.imshow(E[0,:,:,2], aspect='auto')
ax2 = fig.add_subplot(122)
plot2 = ax2.imshow(H[0,:,:,0], aspect='auto')

# CURL FUNCS

def curl_E(E):
    curl = np.zeros(E.shape)
    
    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]
    
    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]
    
    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]
    
    return curl

def curl_H(H):

    curl = np.zeros(H.shape)
    
    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]
    
    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]
    
    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]
    
    return curl

npr2 = 0
npr1 = 1
ncur = 2

def update(n):
    global npr2, npr1, ncur, Hx, Hy, Hz, Ex, Ey, Ez
    npr2 = npr1
    npr1 = ncur
    ncur = np.mod(ncur+1, 3)
    
    # UPDATE FIELDS IN BULK
    # many attempts were made
    """
    # 3d using equations from notes
    Hx[ncur] = Hx[npr1] - Ey[npr1] + np.roll(Ey[npr1], 1, axis=0) - np.roll(Ez[npr1], 1, axis=1) + Ez[npr1]
    Hy[ncur] = Hy[npr1] + np.roll(Ez[npr1], 1, axis=0) - Ez[npr1] + Ex[npr1] - np.roll(Ex[npr1], 1, axis=1)
    Hz[ncur] = Hz[npr1] + np.roll(Ex[npr1], 1, axis=1) - Ex[npr1] - np.roll(Ey[npr1], 1, axis=0) + Ey[npr1]
    Ex[ncur] = Ca*Ex[npr1] + Cb*(Hz[npr1] - np.roll(Hz[npr1], -1, axis=1) - Hy[npr1])
    Ey[ncur] = Ca*Ey[npr1] + Cb*(Hx[npr1] - Hz[npr1] + np.roll(Hz[npr1] , -1, axis=0))
    Ez[ncur] = Ca*Ez[npr1] + Cb*(Hy[npr1] - np.roll(Hy[npr1], -1, axis=0) - Hx[npr1] + np.roll(Hx[npr1], -1, axis=1))
    """
    """
    # neglecting the contribution from fields at k+/-1 (2d here)
    Hx[ncur] = Hx[npr1] - np.roll(Ez[npr1], 1, axis=0) + Ez[npr1]
    Hy[ncur] = Hy[npr1] + np.roll(Ez[npr1], -1, axis=1) - Ez[npr1]
    #Hz[ncur] = Hz[npr1] + np.roll(Ex[npr1], 1, axis=1) - Ex[npr1] - np.roll(Ey[npr1], 1, axis=0) + Ey[npr1]
    #Ex[ncur] = Ca*Ex[npr1] + Cb*(Hz[npr1] - np.roll(Hz[npr1], -1, axis=1))
    #Ey[ncur] = Ca*Ey[npr1] + Cb*(-Hz[npr1] + np.roll(Hz[npr1] , -1, axis=0))
    Ez[ncur] = Ca*Ez[npr1] + Cb*(Hy[npr1] - np.roll(Hy[npr1], 1, axis=1) - Hx[npr1] + np.roll(Hx[npr1], -1, axis=0))
    """
    """
    # 3d using curl
    E[ncur] = E[npr1] + Ra*(1/epsilon_0)*(curl_H(H[npr1]))
    H[ncur] = H[npr1] - Ra*(1/mu_0)*curl_E(E[npr1])
    """
    # 2d, class notes, but fields wrapped in arrays
    H[ncur,:,:,0] = H[npr1,:,:,0] - np.roll(E[npr1,:,:,2], 1, axis=0) + E[npr1,:,:,2]
    H[ncur,:,:,1] = H[npr1,:,:,1] + np.roll(E[npr1,:,:,2], -1, axis=1) - E[npr1,:,:,2]
    E[ncur,:,:,2] = Ca*E[npr1,:,:,2] + Cb*(H[npr1,:,:,1] - np.roll(H[npr1,:,:,1], 1, axis=1) - H[npr1,:,:,0] + np.roll(H[npr1,:,:,0], -1, axis=0))


    # TODO: ENFORCE PML BOUNDARY

    # ADD SOURCES
    E[ncur, IMAX//2, JMAX//2, 2] += 1000*np.sin(n*dt*2*np.pi*f)
    
    # PLOT EZ AND HX
    ax.clear()
    ax2.clear()
    #plot = ax.plot_surface(xx, yy, Ez[ncur])
    plot = ax.imshow(E[ncur,:,:,2], aspect='auto')
    plot2 = ax2.imshow(H[ncur,:,:,0], aspect='auto')
    return plot,plot2

ani = animation.FuncAnimation(fig, update, frames=np.arange(NMAX), interval=200)
plt.show()
