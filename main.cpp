#include <fftw3.h>
#include <math.h>
#include <cstring> // for memset

static fftwf_plan plan_rc, plan_cr;

/*
Final project of NMSC 2025, by Veeti Haaja, based on the simple stable FFT fluid solver by Stam.



*/

/*
This function initializes the FFTW planner routines for real-to-complex and complex-to-real transforms in 2D.
params:
n: size of the grid (number of rows and columns)
*/
void init_FFT(int n)
{
  plan_rc = fftwf_plan_dft_r2c_2d(n, n, nullptr, nullptr, FFTW_ESTIMATE);
  plan_cr = fftwf_plan_dft_c2r_2d(n, n, nullptr, nullptr, FFTW_ESTIMATE);
}

/*
This function deinitializes the FFTW planner routines.
*/
void deinit_FFT()
{
  fftwf_destroy_plan(plan_rc);
  fftwf_destroy_plan(plan_cr);
}

/*
This is a macro to execute the FFTW plan for real-to-complex and complex-to-real transforms.

params:
s: sign of the transform (1 for real-to-complex, -1 for complex-to-real)
u: pointer to the data array
*/
#define FFT(s, u)                                                   \
  if (s == 1)                                                       \
    fftwf_execute_dft_r2c(plan_rc, (float *)u, (fftwf_complex *)u); \
  else                                                              \
    fftwf_execute_dft_c2r(plan_cr, (fftwf_complex *)u, (float *)u)

/*
This is the stable solve algorithm from Jos Stam's paper
u ~ x, v ~ y and external force is entered via u0, v0.

The solver has 4 main steps:
    - Add force field (done in the spatial domain)

    - Advect velocity (done in the spatial domain)

    - Diffuse velocity (done in the frequency domain)

    - Force velocity to conserve mass (done in the frequency domain)

params:
n: size of the grid (number of rows and columns)
u: pointer to the velocity field in the x direction
v: pointer to the velocity field in the y direction

u0: pointer to the external force field in the x direction
v0: pointer to the external force field in the y direction
the above fields are also used for interpolation in the self-advection step

visc: viscosity of the fluid
dt: time step for the simulation
*/

void stable_solve(int n, float *u, float *v, float *u0, float *v0,
                  float visc, float dt)

{

    // initializing variables
    float x, y, x0, y0, f, r, U[2], V[2], s, t;
    int i, j, i0, j0, i1, j1;

    // step 1
    // looping over the velocity arrays
    // updating the velocity fields with the external forces
    // then setting the external velocity fields to match the updated velocity fields (this is used for interpolation in the next step)
    for (i = 0; i < n * n; i++)
    {
        u[i] += dt * u0[i];
        u0[i] = u[i];
        v[i] += dt * v0[i];
        v0[i] = v[i];
    }

    // step 2
    // advecting the velocity fields

    // x and y are the centers of cells' positions
    // i and j are the indices of the cells
    for (x = 0.5 / n, i = 0; i < n; i++, x += 1.0 / n)
    {
        for (y = 0.5 / n, j = 0; j < n; j++, y += 1.0 / n)
        {

            x0 = n * (x - dt * u0[i + n * j]) - 0.5;
            y0 = n * (y - dt * v0[i + n * j]) - 0.5;
            i0 = floor(x0);
            s = x0 - i0;
            i0 = (n + (i0 % n)) % n;
            i1 = (i0 + 1) % n;

            j0 = floor(y0);
            t = y0 - j0;
            j0 = (n + (j0 % n)) % n;
            j1 = (j0 + 1) % n;

            u[i + n * j] = (1 - s) * ((1 - t) * u0[i0 + n * j0] + t * u0[i0 + n * j1]) +
                           s * ((1 - t) * u0[i1 + n * j0] + t * u0[i1 + n * j1]);

            v[i + n * j] = (1 - s) * ((1 - t) * v0[i0 + n * j0] + t * v0[i0 + n * j1]) +
                           s * ((1 - t) * v0[i1 + n * j0] + t * v0[i1 + n * j1]);
        }
    }

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u0[i + (n + 2) * j] = u[i + n * j];
            v0[i + (n + 2) * j] = v[i + n * j];
        }
    
    // transforming the arrays u0 and v0 to the frequency domain
    FFT(1, u0);
    FFT(1, v0);


    // here we solve the viscosity term in the frequency domain
    // we also force the velocity field to conserve mass
    for (i = 0; i <= n; i += 2)
    {
        x = 0.5 * i;

        for (j = 0; j < n; j++)
        {
            int l = j;
            if (j > n / 2)
                l = j - n;

            r = x * x + l * l;
            if (r == 0.0)
                continue;

            f = exp(-r * dt * visc);
            U[0] = u0[i + (n + 2) * j];
            V[0] = v0[i + (n + 2) * j];
            U[1] = u0[i + 1 + (n + 2) * j];
            V[1] = v0[i + 1 + (n + 2) * j];
            u0[i + (n + 2) * j] = f * ((1 - x * x / r) * U[0] - x * l / r * V[0]);
            u0[i + 1 + (n + 2) * j] = f * ((1 - x * x / r) * U[1] - x * l / r * V[1]);

            v0[i + (n + 2) * j] = f * (-l * x / r * U[0] + (1 - l * l / r) * V[0]);
            v0[i + 1 + (n + 2) * j] = f * (-l * x / r * U[1] + (1 - l * l / r) * V[1]);
        }
    }

    // transforming the arrays u0 and v0 back to the spatial domain
    FFT(-1, u0);
    FFT(-1, v0);

    // here we normalize the arrays u and v
    f = 1.0 / (n * n);
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u[i + n * j] = f * u0[i + (n + 2) * j];
            v[i + n * j] = f * v0[i + (n + 2) * j];
        }
}



int main() {

  const int N = 500; // Setting up 500x500 grid for an example
  const int arraysize = N * (N + 2); // fftw uses two extra rows

  // Allocate arrays using fftwf_malloc
  float *u = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));
  float *v = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));

  float *f_x = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));
  float *f_y = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));

  // Initialize arrays to zero (fftwf_malloc does not initialize memory)
  memset(u, 0, sizeof(float) * arraysize);
  memset(v, 0, sizeof(float) * arraysize);
  memset(f_x, 0, sizeof(float) * arraysize);
  memset(f_y, 0, sizeof(float) * arraysize);

  // Your code for the simulation logic goes here


  // Deallocate the arrays, deinitialize FFTW
  fftwf_free(u);
  fftwf_free(v);
  fftwf_free(f_x);
  fftwf_free(f_y);
  deinit_FFT();
}
