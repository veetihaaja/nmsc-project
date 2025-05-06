import numpy as np
import matplotlib.pyplot as plt

D = np.loadtxt('output/D', dtype=float)
timesteps = np.loadtxt('output/timestep', dtype=float)
vx = np.loadtxt('output/vx', dtype=float)
vy = np.loadtxt('output/vy', dtype=float)

N = 500

for i, val in enumerate(timesteps):
    Dgrid = D[i].reshape((N, N)).T
    vxgrid = vx[i].reshape((N, N)).T
    vygrid = vy[i].reshape((N, N)).T

    v_mag_grid = np.sqrt(vxgrid**2 + vygrid**2)

    plt.imshow(Dgrid, cmap='viridis')
    plt.colorbar()
    plt.title(f'D at timestep {val:.4f}')
    if (i < 10):
        plt.savefig(f'output/D_plots/0{i}.png')
    else:
        plt.savefig(f'output/D_plots/{i}.png')
    plt.close()

    plt.imshow(v_mag_grid, cmap='viridis')
    plt.colorbar()
    plt.title(f'Velocity Magnitude at timestep {val:.4f}')
    if (i < 10):
        plt.savefig(f'output/v_plots/0{i}.png')
    else:
        plt.savefig(f'output/v_plots/{i}.png')
    plt.close()