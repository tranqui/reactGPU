#include "reactgpu.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <stdexcept>
#include <iostream>


namespace kernel
{
    __constant__ Scalar dt, dxInv, dyInv;
    __constant__ int nrows, ncols;
    __constant__ Scalar Du, Dv, k;

    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    static constexpr int num_ghost = 1;

    __global__
    void reactor_integration(Scalar* u, Scalar* v)
    {
        __shared__ Scalar f[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];
        __shared__ Scalar g[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load tile into shared memory.

        f[i][j] = u[index];
        g[i][j] = v[index];

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            f[i - num_ghost][j] = u[col + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i + tile_rows][j] = u[col + ((row + tile_rows) % nrows) * ncols];

            g[i - num_ghost][j] = v[col + ((row - num_ghost + nrows) % nrows) * ncols];
            g[i + tile_rows][j] = v[col + ((row + tile_rows) % nrows) * ncols];
        }

        if (threadIdx.x < num_ghost)
        {
            f[i][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + row * ncols];
            f[i][j + tile_cols] = u[(col + tile_cols) % ncols         + row * ncols];

            g[i][j - num_ghost] = v[(col - num_ghost + ncols) % ncols + row * ncols];
            g[i][j + tile_cols] = v[(col + tile_cols) % ncols         + row * ncols];
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            f[i - num_ghost][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i - num_ghost][j + tile_cols] = u[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i + tile_rows][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            f[i + tile_rows][j + tile_cols] = u[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];

            g[i - num_ghost][j - num_ghost] = v[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            g[i - num_ghost][j + tile_cols] = v[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            g[i + tile_rows][j - num_ghost] = v[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            g[i + tile_rows][j + tile_cols] = v[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
        }

        __syncthreads();

        // Laplacian with nth order stencil.
        
        // Scalar f_rhs = Du * (f[i-1][j] + f[i+1][j] + f[i][j-1] + f[i][j+1] - 4*f[i][j]);
        // Scalar g_rhs = Dv * (g[i-1][j] + g[i+1][j] + g[i][j-1] + g[i][j+1] - 4*g[i][j]);
        Scalar f_rhs = Du * (f[i-1][j] + f[i+1][j] + f[i][j-1] + f[i][j+1] - 4*f[i][j]);
        Scalar g_rhs = Dv * (g[i-1][j] + g[i+1][j] + g[i][j-1] + g[i][j+1] - 4*g[i][j]);

        // Scalar f02 = (dyInv*dyInv*(2*f[si][-3 + sj] - 27*f[si][-2 + sj] + 270*f[si][-1 + sj] - 490*f[si][sj] + 270*f[si][1 + sj] - 27*f[si][2 + sj] + 2*f[si][3 + sj]))/180.;
        // Scalar f20 = (dxInv*dxInv*(2*f[-3 + si][sj] - 27*f[-2 + si][sj] + 270*f[-1 + si][sj] - 490*f[si][sj] + 270*f[1 + si][sj] - 27*f[2 + si][sj] + 2*f[3 + si][sj]))/180.;
        // Scalar f_rhs = f02 + f20;

        // Scalar g02 = (dyInv*dyInv*(2*g[si][-3 + sj] - 27*g[si][-2 + sj] + 270*g[si][-1 + sj] - 490*g[si][sj] + 270*g[si][1 + sj] - 27*g[si][2 + sj] + 2*g[si][3 + sj]))/180.;
        // Scalar g20 = (dxInv*dxInv*(2*g[-3 + si][sj] - 27*g[-2 + si][sj] + 270*g[-1 + si][sj] - 490*g[si][sj] + 270*g[1 + si][sj] - 27*g[2 + si][sj] + 2*g[3 + si][sj]))/180.;
        // Scalar g_rhs = 10 * (g02 + g20);

        // Add in chemical flux.

        // Scalar chem_flux = (k + (1-k) * square(f[i][j]) / (1 + square(f[i][j]))) * g[i][j] - f[i][j];
        Scalar chem_flux = (k + (1-k) * f[i][j]*f[i][j] / (1 + f[i][j]*f[i][j])) * g[i][j] - f[i][j];
        f_rhs += chem_flux;
        g_rhs -= chem_flux;

        // Update concentrations with diffusion effects only

        u[index] += f_rhs * dt;
        v[index] += g_rhs * dt;
        // if (i == 0) u[index] = 0;
        // if (i == nrows) u[index] = 0;
        // if (j == 0) u[index] = 0;
        // if (j == ncols) u[index] = 0;
    }

    __global__
    void rhs(Scalar* u, Scalar *rhs)
    {
        __shared__ Scalar f[tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load tile into shared memory.

        f[i][j] = u[index];

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            f[i - num_ghost][j] = u[col + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i + tile_rows][j] = u[col + ((row + tile_rows) % nrows) * ncols];
        }

        if (threadIdx.x < num_ghost)
        {
            f[i][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + row * ncols];
            f[i][j + tile_cols] = u[(col + tile_cols) % ncols         + row * ncols];
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            f[i - num_ghost][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i - num_ghost][j + tile_cols] = u[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
            f[i + tile_rows][j - num_ghost] = u[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
            f[i + tile_rows][j + tile_cols] = u[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
        }

        __syncthreads();
        
        rhs[index] = f[i+1][j] + f[i-1][j] + f[i][j+1] + f[i][j-1] - 4*f[i][j];
    }
}

Reactor::Reactor(const Eigen::Ref<const State>& initial_u,
                 const Eigen::Ref<const State>& initial_v,
                 Scalar dt, Scalar dx, Scalar dy,
                 Scalar Du, Scalar Dv, Scalar k,
                 size_t current_step)
    : dt(dt), dx(dx), dy(dy), dxInv(1/dx), dyInv(1/dy),
    nrows(initial_u.rows()), ncols(initial_u.cols()),
    pitch_width(initial_u.cols() * sizeof(Scalar)),
    mem_size(initial_u.rows() * initial_u.cols() * sizeof(Scalar)),
    Du(Du), Dv(Dv), k(k),
    current_step(current_step)
{
    if (initial_v.rows() != nrows or initial_v.cols() != ncols)
        throw std::runtime_error("fields u and v do not have the same dimensions!");

    // Initialize CUDA memory
    cudaMallocPitch(&u, &pitch_u, pitch_width, nrows);
    cudaMallocPitch(&v, &pitch_v, pitch_width, nrows);
    cudaMallocPitch(&rhs, &pitch_rhs, pitch_width, nrows);

    // Copy initial state to CUDA memory
    cudaMemcpy(u, initial_u.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(v, initial_v.data(), mem_size, cudaMemcpyHostToDevice);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
    }
}

Reactor::~Reactor()
{
    cudaFree(u);
    cudaFree(v);
    cudaFree(rhs);
}

void Reactor::run(const int nsteps)
{
    // Set parameters on device.
    cudaMemcpyToSymbol(kernel::dt, &dt, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::dxInv, &dxInv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::dyInv, &dyInv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::Du, &Du, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::Dv, &Dv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::k, &k, sizeof(Scalar), 0, cudaMemcpyHostToDevice);

    // Calculate new state on device.
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);

    for (int step = 0; step < nsteps; ++step)
    {
        kernel::reactor_integration<<<grid_size, block_dim>>>(u, v);
    }

    cudaDeviceSynchronize();

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
    }

    current_step += nsteps;
}

State Reactor::rhs_cpu() const
{
    auto f = get_u();
    auto out = State(nrows, ncols);
    out.setZero();
    // std::cout << f << std::endl;

    for (int i = 0; i < nrows; ++i)
    {
        int down = (i-1+nrows) % nrows;
        int up = (i+1) % nrows;
        for (int j = 0; j < ncols; ++j)
        {
            int left = (j-1+ncols) % ncols;
            int right = (j+1) % ncols;
            out(i,j) = f(i,left) + f(i,right) + f(up,j) + f(down,j) - 4*f(i,j);
            // out(i,j) = f(left,j) + f(right,j) - 2*f(i,j);
            // out(i,j) = f(i,up) + f(i,down) - 2*f(i,j);
            // out(i,j) = f(i,left);
        }
    }

    return out;
}

State Reactor::rhs_gpu() const
{
    // Set parameters on device.
    cudaMemcpyToSymbol(kernel::dt, &dt, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::dxInv, &dxInv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::dyInv, &dyInv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::Du, &Du, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::Dv, &Dv, sizeof(Scalar), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel::k, &k, sizeof(Scalar), 0, cudaMemcpyHostToDevice);

    // Calculate new state on device.
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);

    {
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Kernel Error1: " << cudaGetErrorString(error) << std::endl;
        }
    }

    kernel::rhs<<<grid_size, block_dim>>>(u, rhs);
    cudaDeviceSynchronize();

    {
        auto error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Kernel Error2: " << cudaGetErrorString(error) << std::endl;
        }
    }

    auto out = State(nrows, ncols);
    out.setZero();
    cudaMemcpy(out.data(), rhs, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

State Reactor::get_u() const
{
    auto out = State(nrows, ncols);
    cudaMemcpy(out.data(), u, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

State Reactor::get_v() const
{
    auto out = State(nrows, ncols);
    cudaMemcpy(out.data(), v, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

size_t Reactor::step() const
{
    return current_step;
}

Scalar Reactor::time() const
{
    return static_cast<Scalar>(current_step) * dt;
}
