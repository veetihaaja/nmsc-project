#include <fftw3.h>
#include <math.h>
#include <cstring> // for memset
#include <iostream>
#include <fstream>

static fftwf_plan plan_rc, plan_cr;

#define DEBUG 0

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

D: pointer to the dye field

visc: viscosity of the fluid
dt: time step for the simulation
*/

void stable_solve(int n, float *u, float *v, float *u0, float *v0,
                  float *D, float *D0, float visc, float dt)

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
        D0[i] = D[i];
    }

    if (DEBUG) std::cout << "step 1, outside force done" << std::endl;

    // step 2
    // advecting the velocity fields    
    for (x = 0.5 / n, i = 0; i < n; i++, x += 1.0 / n)
    {
        for (y = 0.5 / n, j = 0; j < n; j++, y += 1.0 / n)
        {

            // this is interpolation, where x0 and y0 are the cell center positions times n, when interpolated backwards a timestep. 
            x0 = n * (x - dt * u0[i + n * j]) - 0.5;
            y0 = n * (y - dt * v0[i + n * j]) - 0.5;

            // her we calculate i0, i1 and s, which are
            // i0 is the index of the cell at x0
            i0 = floor(x0);
            // s is the distance from the cell center to the position x0
            s = x0 - i0;
            // this accounts for the periodic boundary conditions
            i0 = (n + (i0 % n)) % n;
            // i1 is the index of the next cell in the x direction
            i1 = (i0 + 1) % n;

            // same things here but for the y direction
            j0 = floor(y0);
            t = y0 - j0;
            j0 = (n + (j0 % n)) % n;
            j1 = (j0 + 1) % n;

            u[i + n * j] = (1 - s) * ((1 - t) * u0[i0 + n * j0] + t * u0[i0 + n * j1]) +
                           s * ((1 - t) * u0[i1 + n * j0] + t * u0[i1 + n * j1]);

            v[i + n * j] = (1 - s) * ((1 - t) * v0[i0 + n * j0] + t * v0[i0 + n * j1]) +
                           s * ((1 - t) * v0[i1 + n * j0] + t * v0[i1 + n * j1]);

            D[i + n * j] = (1 - s) * ((1 - t) * D0[i0 + n * j0] + t * D0[i0 + n * j1]) +
                            s * ((1 - t) * D0[i1 + n * j0] + t * D0[i1 + n * j1]); 
        }
    }

    if (DEBUG) std::cout << "step 2, advection done" << std::endl;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u0[i + (n + 2) * j] = u[i + n * j];
            v0[i + (n + 2) * j] = v[i + n * j];
            D0[i + (n + 2) * j] = D[i + n * j];
        }
    
    if (DEBUG) std::cout << "step 3, setting boundaries, done" << std::endl;

    if (DEBUG) std::cout << "transforming into frequency domain" << std::endl;

    // transforming the arrays u0 and v0 to the frequency domain
    FFT(1, u0);
    FFT(1, v0);

    if (DEBUG) std::cout << "transformation done" << std::endl;


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

            f = expf(-r * dt * visc);
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

    if (DEBUG) std::cout << "step 4, viscosity and mass conservation done, transforming back into spatial domain" << std::endl;

    // transforming the arrays u0 and v0 back to the spatial domain
    FFT(-1, u0);
    FFT(-1, v0);

    if (DEBUG) std::cout << "transform back done" << std::endl;

    // here we normalize the arrays u and v
    f = 1.0 / (n * n);
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            u[i + n * j] = f * u0[i + (n + 2) * j];
            v[i + n * j] = f * v0[i + (n + 2) * j];
        }

    if (DEBUG) std::cout << "step 5, normalizing arrays done" << std::endl;
    
    // to change this to solve the dye field advection, we need to make a new D0 array for the interpolation

    if (DEBUG)
    {
        std::cout << "solver done" << std::endl;
    }

}

void initial_fx_field(float *f_x, const int n, const float U_0, const float delta) {

    float y_j;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y_j = (float)j / n;
            f_x[i + n * j] = U_0 * tanhf((y_j - 0.5)/(delta));
        }
    }   
}

void initial_fy_field(float *f_y, const int n, const float A, const float k, const float sigma) {

    float x_i, y_j;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y_j = (float)j / n;
            x_i = (float)i / n;
            f_y[i + n * j] = A * sinf(2 * M_PI * k * x_i) * expf(-(y_j - 0.5)*(y_j - 0.5)/(2 * sigma * sigma));
        }
    }
}

void initial_D_field(float *D, const int n) {

    float y_j;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y_j = (float)j / n;
            if (y_j < 0.5) {
                D[i + n * j] = 0.0;
            } else {
                D[i + n * j] = 1.0;
            }
        }
    }
}

void write_to_file(const float *v_x, const float *v_y, const float *D, const int n, const float time) {

    // writing timestep
    std::ofstream timestepfile;
    timestepfile.open("output/timestep", std::ios::app);

    timestepfile << time << "\n";

    timestepfile.close();

    //writing vx
    std::ofstream vxfile;
    vxfile.open("output/vx", std::ios::app);

    for (int i = 0; i<n; i++) {
        for (int j = 0; j<n; j++) {
            vxfile << v_x[i + n * j] << " ";
        }
    }

    vxfile << "\n";
    vxfile.close();

    //writing vy

    std::ofstream vyfile;
    vyfile.open("output/vy", std::ios::app);

    for (int i = 0; i<n; i++) {
        for (int j = 0; j<n; j++) {
            vyfile << v_y[i + n * j] << " ";
        }
    }

    vyfile << "\n";
    vyfile.close();

    //writing D


    std::ofstream Dfile;
    Dfile.open("output/D", std::ios::app);

    for (int i = 0; i<n; i++) {
        for (int j = 0; j<n; j++) {
            Dfile << D[i + n * j] << " ";
        }
    }

    Dfile << "\n";
    Dfile.close();

}

void clearOutputFiles() {
    std::ofstream timestepfile;
    timestepfile.open("output/timestep", std::ios::trunc);
    timestepfile.close();

    std::ofstream vxfile;
    vxfile.open("output/vx", std::ios::trunc);
    vxfile.close();

    std::ofstream vyfile;
    vyfile.open("output/vy", std::ios::trunc);
    vyfile.close();

    std::ofstream Dfile;
    Dfile.open("output/D", std::ios::trunc);
    Dfile.close();
}

int main() {

    const int N = 500; // Setting up 500x500 grid for an example
    const int arraysize = N * (N + 2); // fftw uses two extra rows

    const float delta_t = 0.01; // time step
    const float visc = 0.001; // viscosity

    const float U_0 = 5.0; 
    const float delta = 0.025; 

    const float A = 1.0; 
    const float sigma = 0.02;
    const float k = 4.0; 

    // Initialize FFTW
    init_FFT(N);

    // Allocate arrays using fftwf_malloc
    float *u = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));
    float *v = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));

    float *f_x = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));
    float *f_y = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));

    float *D = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));
    float *D0 = static_cast<float *>(fftwf_malloc(sizeof(float) * arraysize));

    // Initialize arrays to zero (fftwf_malloc does not initialize memory)
    memset(u, 0, sizeof(float) * arraysize);
    memset(v, 0, sizeof(float) * arraysize);
    memset(f_x, 0, sizeof(float) * arraysize);
    memset(f_y, 0, sizeof(float) * arraysize);

    // Initialize dye field

    initial_D_field(D, N);

    float time = 0.0;
    const float time_end = 1000.0;

    int timestep = 0;
    const int simulation_steps = 500;

    const int write_interval = 10; // write every n steps

    std::cout << "writing initial state and starting simulation" << std::endl;
    // write initial state to file
    clearOutputFiles();
    write_to_file(u, v, D, N, time);
    // main simulation loop
    while (time < time_end && timestep < simulation_steps) {

        if (DEBUG) std::cout << "timestep: " << timestep << ", time: " << time << std::endl;

        // apply force fields
        if (timestep < 10) {
            initial_fx_field(f_x, N, U_0, delta);
            initial_fy_field(f_y, N, A, k, sigma);
        } else {
            memset(f_x, 0.0, sizeof(float) * arraysize);
            memset(f_y, 0.0, sizeof(float) * arraysize);
        }

        if (DEBUG) std::cout << "force fields applied" << std::endl; 

        // apply solver
        stable_solve(N, u, v, f_x, f_y, D, D0, visc, delta_t);

        if (DEBUG) std::cout << "solver applied" << std::endl;

        // write to file every write_interval steps
        if (timestep % write_interval == 0) {
            write_to_file(u, v, D, N, time);
        }

        if (DEBUG) std::cout << "file written" << std::endl;

        time += delta_t;
        timestep++;
        std::cout << "timestep: " << timestep << ", time: " << time << std::endl;

    }


    // Deallocate the arrays, deinitialize FFTW
    fftwf_free(u);
    fftwf_free(v);
    fftwf_free(f_x);
    fftwf_free(f_y);
    deinit_FFT();
}
