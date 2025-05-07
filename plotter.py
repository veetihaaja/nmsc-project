import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

D = np.loadtxt('output/D', dtype=float)
timesteps = np.loadtxt('output/timestep', dtype=float)
vx = np.loadtxt('output/vx', dtype=float)
vy = np.loadtxt('output/vy', dtype=float)

N = 500

fig, ax = plt.subplots(2, 1, figsize=(7, 14), dpi=100)

dplot = ax[0].imshow(D[0].reshape((N, N)).T, cmap='viridis')

def animateD(i):
    Dgrid = D[i].reshape((N, N)).T
    #vxgrid = vx[i].reshape((N, N)).T
    #vygrid = vy[i].reshape((N, N)).T
    #v_mag_grid = np.sqrt(vxgrid**2 + vygrid**2)

    dplot.set_array(Dgrid)

    #ax[0].imshow(Dgrid, cmap='viridis')
    #ax[0].colorbar()
    #ax[0].set_title(f"Time = {timesteps[i]} s")

    #ax[1].quiver(np.arange(N), np.arange(N), vxgrid, vygrid, color='r', scale=10)
    #ax[1].imshow(v_mag_grid, cmap='viridis')
    #ax[1].colorbar()
    #ax[1].set_title(f"Time = {timesteps[i]} s")
    return dplot,

anim = FuncAnimation(fig, animateD, frames=len(timesteps), interval=100, blit=True)
anim.save('animation.mp4', writer='ffmpeg', fps=2)

