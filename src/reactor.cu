#include "reactor.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <stdexcept>
#include <iostream>


namespace kernel
{
    void check_error()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::string message = "CUDA Kernel Error: " + std::string(cudaGetErrorString(error));
            throw CudaError(message);
        }
    }


    // A utility to count number of functor parameters.
    template <typename T>
    struct Arity;

    template <typename R, typename... Args>
    struct Arity<R(*)(Args...)>
    {
        static constexpr size_t value = sizeof...(Args);
    };

    static constexpr int MAX_PARAMETERS = 256;
    __constant__ Scalar chemical_flux_parameters[MAX_PARAMETERS];

    template <typename Implementation, size_t... I, typename... Fields>
    __device__ auto chemical_flux(std::index_sequence<I...>, Fields&&... fields)
    {
        return Implementation::chemical_flux(std::forward<Fields>(fields)...,
                                             chemical_flux_parameters[I]...);
    }

    template <typename Implementation, typename... Fields>
    __device__ auto chemical_flux(Fields&&... fields)
    {
        constexpr auto nparams = Arity<decltype(&Implementation::chemical_flux)>::value - sizeof...(fields);
        return chemical_flux<Implementation>(std::make_index_sequence<nparams>{},
                                             std::forward<Fields>(fields)...);
    }

    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    static constexpr int num_ghost = 1;

    __constant__ Scalar dt, dxInv, dyInv;
    __constant__ int nrows, ncols;
    __constant__ Scalar Du, Dv;


    // template <typename ChemicalFlux>
    __global__ void reactor_integration(Scalar* u, Scalar* v)
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

        // Evolve with chemical flux for reactions.
        auto [f_rhs, g_rhs] = chemical_flux<CellPolarisation>(f[i][j], g[i][j]);
  
        // Add in Laplacian (with 2nd order stencil) to make this reaction-diffusion.
        f_rhs += Du * (  dxInv*dxInv * (f[i+1][j] + f[i-1][j])
                       + dyInv*dyInv * (f[i][j+1] + f[i][j-1])
                       - 2*(dxInv*dxInv + dyInv*dyInv) * f[i][j]);
        g_rhs += Dv * (  dxInv*dxInv * (g[i+1][j] + g[i-1][j])
                       + dyInv*dyInv * (g[i][j+1] + g[i][j-1])
                       - 2*(dxInv*dxInv + dyInv*dyInv) * g[i][j]);

        u[index] += f_rhs * dt;
        v[index] += g_rhs * dt;
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

    // Copy initial state to CUDA memory
    cudaMemcpy(u, initial_u.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(v, initial_v.data(), mem_size, cudaMemcpyHostToDevice);

    kernel::check_error();
}

Reactor::~Reactor()
{
    cudaFree(u);
    cudaFree(v);
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

    // Extra system-dependent parameters for the chemical flux.
    Scalar parameters[kernel::MAX_PARAMETERS] = {k};
    cudaMemcpyToSymbol(kernel::chemical_flux_parameters, parameters, sizeof(parameters));

    // Calculate new state on device.
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);

    for (int step = 0; step < nsteps; ++step)
    {
        kernel::reactor_integration<<<grid_size, block_dim>>>(u, v);
    }

    cudaDeviceSynchronize();
    kernel::check_error();

    current_step += nsteps;
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
