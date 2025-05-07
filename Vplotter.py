import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

timesteps = np.loadtxt('output/timestep', dtype=float)
vx = np.loadtxt('output/vx', dtype=float)
vy = np.loadtxt('output/vy', dtype=float)

N = 500
arrow_spacing = 50  # Draw arrows every 10 points

fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

X, Y = np.meshgrid(np.arange(0, N, arrow_spacing), np.arange(0, N, arrow_spacing))
vxgrid = vx[0].reshape((N, N))[::arrow_spacing, ::arrow_spacing].T
vygrid = vy[0].reshape((N, N))[::arrow_spacing, ::arrow_spacing].T

Q = ax.quiver(X, Y, vxgrid, vygrid, pivot="mid", color='r', scale=6, units='width')

def animateD(i, Q, X, Y):
    vxgrid = vx[i].reshape((N, N))[::arrow_spacing, ::arrow_spacing].T
    vygrid = vy[i].reshape((N, N))[::arrow_spacing, ::arrow_spacing].T
    
    Q.set_UVC(vxgrid, vygrid)
    ax.set_title(f'time {timesteps[i]}, timestep {i}')
    return Q,

if timesteps[11]-timesteps[10] < 0.09:
    fps = 24
else:
    fps = 2

anim = FuncAnimation(fig, animateD, fargs=(Q, X, Y), frames=len(timesteps), interval=100, blit=False)
anim.save('V_animation.mp4', writer='ffmpeg', fps=fps)