import numpy as np
import matplotlib.pyplot as plt

D = np.loadtxt('output/D', dtype=float)
timesteps = np.loadtxt('output/timestep', dtype=float)
vx = np.loadtxt('output/vx', dtype=float)
vy = np.loadtxt('output/vy', dtype=float)

N = 500

for i, val in enumerate(timesteps):
    Dgrid = D[i].reshape((N, N))
    vxgrid = vx[i].reshape((N, N))
    vygrid = vy[i].reshape((N, N))

    v_mag_grid = np.sqrt(vxgrid**2 + vygrid**2)

    plt.imshow(Dgrid, cmap='viridis')
    plt.colorbar()
    plt.title(f'D at timestep {val:.2f}')
    plt.savefig(f'output/D_plots/D_{i}.png')
    plt.close()

    plt.imshow(v_mag_grid, cmap='viridis')
    plt.colorbar()
    plt.title(f'Velocity Magnitude at timestep {val:.2f}')
    plt.savefig(f'output/v_plots/v_mag_{i}.png')
    plt.close()