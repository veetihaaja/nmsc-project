import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

D = np.loadtxt('../run/output/D', dtype=float)
timesteps = np.loadtxt('../run/output/timestep', dtype=float)

N = 500
if timesteps[4]-timesteps[3] < 0.09:
    fps = 24
    times = 1
else:
    fps = 2
    times = 10

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

dplot = plt.imshow(D[0].reshape((N, N)).T, cmap='viridis', origin='lower')

def animateD(i):
    Dgrid = D[i].reshape((N, N)).T

    dplot.set_array(Dgrid)
    ax.set_title(f'time {timesteps[i]}, timestep {i*times}')
    return dplot,

if timesteps[11]-timesteps[10] < 0.09:
    fps = 24
else:
    fps = 2

anim = FuncAnimation(fig, animateD, frames=len(timesteps), interval=100, blit=True)
anim.save('../run/D_animation.mp4', writer='ffmpeg', fps=fps)

