#include "reactor.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>
#include <cmath>
#include <stdexcept>


/// Helper functions to navigate through a variadic number of fields at compile-time.

// Apply lambda to each element of a tuple (a compile-time ranged-based for loop).
template <std::size_t index = 0, typename Function, typename... T>
inline __host__ __device__ typename std::enable_if<index == sizeof...(T), void>::type
for_each(std::tuple<T...> &, Function) { }
template <std::size_t index = 0, typename Function, typename... T>
inline __host__ __device__ typename std::enable_if<index < sizeof...(T), void>::type
for_each(std::tuple<T...>& t, Function f)
{
    f(index, std::get<index>(t));
    for_each<index + 1, Function, T...>(t, f);
}

// Apply lambda function as a more conventional "for loop" (but at compile-time).
template <std::size_t... I, typename Function>
inline __host__ __device__ void for_each(std::index_sequence<I...>, Function func)
{
    (func(std::integral_constant<std::size_t, I>{}), ...);
}


/// Main execution on GPU device.

namespace kernel
{
    // Check CUDA for errors after GPU execution and throw them.
    __host__ void throw_errors()
    {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::string message = "CUDA Kernel Error: "
                                + std::string(cudaGetErrorString(error));
            throw CudaError(message);
        }
    }

    // Implementation is on a 2d grid with periodic boundary conditions.
    // GPU divided into an (tile_rows x tile_cols) tile (blocks) with
    // a CUDA thread for each tile sharing this memory. Varying the tile size
    // will potentially improve performance on different hardware - I found
    // 16x16 was close to optimum on my machine for simulations on a 1024x1024 grid.
    static constexpr int tile_rows = 16;
    static constexpr int tile_cols = 16;
    // We need ghost points for each tile so we can evaluate derivatives
    // (specifically the Laplacian for diffusion) at the tile borders.
    static constexpr int num_ghost = 1; // <- minimum for second-order finite-difference stencil.

    // Stencil parameters - 2d space (x, y), and time t.
    __constant__ Scalar dt, dxInv, dyInv; // size of each space-time point
    __constant__ int nrows, ncols;        // number of points in spatial grid

    // Diffusion coefficients for each species.
    static constexpr int MAX_FIELDS = 16;
    __constant__ Scalar D[MAX_FIELDS];


    /// Execution of chemical flux on CUDA device.

    // The chemical flux may take some number of constant parameters.
    // The implementation of the chemical flux is known at compile-time, so we can read
    // its function signature to determine the number via a bit of template metaprogramming.
    // This makes it trivial to implement new systems by just defining a new chemical
    // flux function. The implementation happens at compile-time, so should not have a
    // performance overhead.
    static constexpr int MAX_PARAMETERS = 256;
    __constant__ Scalar chemical_flux_parameters[MAX_PARAMETERS];

    // Evaluate chemical flux by unpacking the correct number of parameters to match
    // the implementation's signature.
    template <typename Implementation, std::size_t... I, typename... Fields>
    __device__ auto evaluate_chemical_flux(std::index_sequence<I...>, Fields&&... fields)
    {
        return Implementation::chemical_flux(std::forward<Fields>(fields)...,
                                             chemical_flux_parameters[I]...);
    }
    template <typename Implementation, typename... Fields>
    __device__ auto evaluate_chemical_flux(Fields&&... fields)
    {
        constexpr auto nparams = Implementation::nparams_device();
        return evaluate_chemical_flux<Implementation>(std::make_index_sequence<nparams>{},
                                                      std::forward<Fields>(fields)...);
    }

    // Extract the [i][j] element from each field in the tile.
    template<typename Tile, std::size_t... I>
    __device__ auto local_fields(Tile&& tile, std::index_sequence<I...>, int i, int j) {
        return std::make_tuple(tile[I][i][j]...);
    }


    /// The kernel itself.

    template <typename System, typename Fields>
    __global__ void reactor_integration(Fields fields)
    {
        constexpr size_t nfields = std::tuple_size<Fields>::value;
        static_assert(nfields == System::nfields(), "Number of fields passed incompatible with system!");

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        // Local indices.
        const int i = threadIdx.y + num_ghost;
        const int j = threadIdx.x + num_ghost;

        // Load tile into shared memory.
        __shared__ Scalar tile[nfields][tile_rows + 2*num_ghost][tile_cols + 2*num_ghost];

        {
            auto load = [&](auto m, auto field)
            {
                tile[m][i][j] = field[index];
            };
            for_each(fields, load);
        }

        // Fill in ghost points.

        if (threadIdx.y < num_ghost)
        {
            auto load = [&](auto m, auto field)
            {
                tile[m][i - num_ghost][j] = field[col + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[m][i + tile_rows][j] = field[col + ((row + tile_rows) % nrows) * ncols];
            };
            for_each(fields, load);
        }

        if (threadIdx.x < num_ghost)
        {
            auto load = [&](auto m, auto field)
            {
                tile[m][i][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + row * ncols];
                tile[m][i][j + tile_cols] = field[(col + tile_cols) % ncols         + row * ncols];
            };
            for_each(fields, load);
        }

        if (threadIdx.x < num_ghost and threadIdx.y < num_ghost)
        {
            auto load = [&](auto m, auto field)
            {
                tile[m][i - num_ghost][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[m][i - num_ghost][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row - num_ghost + nrows) % nrows) * ncols];
                tile[m][i + tile_rows][j - num_ghost] = field[(col - num_ghost + ncols) % ncols + ((row + tile_rows) % nrows) * ncols];
                tile[m][i + tile_rows][j + tile_cols] = field[(col + tile_cols) % ncols         + ((row + tile_rows) % nrows) * ncols];
            };
            for_each(fields, load);
        }

        __syncthreads();

        // Contributions to evolution equation from reactions.
        auto flux = [&](auto&&... args)
        {
            return evaluate_chemical_flux<System>(std::forward<decltype(args)>(args)...);
        };
        auto rhs = std::apply(flux, local_fields(tile, std::make_index_sequence<nfields>{}, i, j));

        // Contributions from Laplacian, making it into reaction-diffusion.
        // Implementation uses a basic second-order central finite difference stencil.
        auto diffusion = [&](auto m)
        {
            std::get<m>(rhs) += D[m] * (  dxInv*dxInv * (tile[m][i+1][j] + tile[m][i-1][j])
                                        + dyInv*dyInv * (tile[m][i][j+1] + tile[m][i][j-1])
                                        - 2*(dxInv*dxInv + dyInv*dyInv) * tile[m][i][j]);
        };
        for_each(std::make_index_sequence<nfields>{}, diffusion);

        // Integrate the original field with an Euler forward step and we're done.
        auto evolve = [&](auto m)
        {
            std::get<m>(fields)[index] += std::get<m>(rhs) * dt;
        };
        for_each(std::make_index_sequence<nfields>{}, evolve);
    }

    // Basic kernel to check for errors (e.g. if fields become nan or inf).
    template <typename System, typename Fields>
    __global__ void check_finite(Fields fields, bool* finite)
    {
        constexpr size_t nfields = std::tuple_size<Fields>::value;
        static_assert(nfields == System::nfields(), "Number of fields passed incompatible with system!");

        // Global indices.
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int index = col + row * ncols;

        auto check = [&](auto m, auto field)
        {
            if (not std::isfinite(field[index])) *finite = false;
        };
        for_each(fields, check);
    }
}


/// Host device definitions.

template <typename System>
Reactor<System>::Reactor(Reactor<System>&& other) noexcept
    : dt(other.dt), dx(other.dx), dy(other.dy), dxInv(other.dxInv), dyInv(other.dyInv),
    nrows(other.nrows), ncols(other.ncols),
    pitch_width(other.pitch_width), mem_size(other.mem_size),
    D(std::move(other.D)), flux_parameters(other.flux_parameters),
    current_step(other.current_step), pitch(std::move(other.pitch)),
    fields(std::move(other.fields))
{
    // Zero the pointers in the rvalue so that no device memory is freed at deconstruction.
    auto zero = [&](auto m)
    {
        std::get<m>(other.fields) = nullptr;
    };
    for_each(std::make_index_sequence<nfields>{}, zero);
}

template <typename System>
Reactor<System>::Reactor(const InitialState& initial_fields,
                         Scalar dt, Scalar dx, Scalar dy,
                         std::array<Scalar, nfields> D,
                         std::array<Scalar, nparams_host> params,
                         int current_step)
    : dt(dt), dx(dx), dy(dy), dxInv(1/dx), dyInv(1/dy),
    nrows(std::get<0>(initial_fields).rows()),
    ncols(std::get<0>(initial_fields).cols()),
    pitch_width(std::get<0>(initial_fields).cols() * sizeof(Scalar)),
    mem_size(std::get<0>(initial_fields).rows() * std::get<0>(initial_fields).cols() * sizeof(Scalar)),
    D(D), flux_parameters(params), current_step(current_step)
{
    auto malloc = [&](auto m)
    {
        if (std::get<m>(initial_fields).rows() != nrows or std::get<1>(initial_fields).cols() != ncols)
            throw std::runtime_error("fields do not have the same dimensions!");

        // Initialize device memory.
        cudaMallocPitch(&std::get<m>(fields), &pitch[m], pitch_width, nrows);
        set_field<m>(std::get<m>(initial_fields));
    };
    for_each(std::make_index_sequence<nfields>{}, malloc);

    kernel::throw_errors();
}

template <typename System>
Reactor<System>::~Reactor()
{
    auto free = [&](auto m)
    {
        cudaFree(std::get<m>(fields));
    };
    for_each(std::make_index_sequence<nfields>{}, free);
}

template <typename System>
void Reactor<System>::run(const int nsteps)
{
    // Set parameters on device.
    cudaMemcpyToSymbol(kernel::dt, &dt, sizeof(Scalar));
    cudaMemcpyToSymbol(kernel::dxInv, &dxInv, sizeof(Scalar));
    cudaMemcpyToSymbol(kernel::dyInv, &dyInv, sizeof(Scalar));
    cudaMemcpyToSymbol(kernel::nrows, &nrows, sizeof(int));
    cudaMemcpyToSymbol(kernel::ncols, &ncols, sizeof(int));

    // Diffusion coefficients for each species.
    cudaMemcpyToSymbol(kernel::D, &D, sizeof(D));

    // Extra system-dependent parameters for the chemical flux.
    auto params = device_parameters();
    cudaMemcpyToSymbol(kernel::chemical_flux_parameters, &params, sizeof(params));

    // Calculate new state on device.
    const dim3 block_dim(kernel::tile_cols, kernel::tile_rows);
    const dim3 grid_size((ncols + block_dim.x - 1) / block_dim.x,
                         (nrows + block_dim.y - 1) / block_dim.y);

    for (int step = 0; step < nsteps; ++step)
    {
        kernel::reactor_integration<System><<<grid_size, block_dim>>>(fields);
    }

    cudaDeviceSynchronize();
    kernel::throw_errors();

    // Numerical errors in integration often cause fields to diverge or go to nan, so we
    // need to check for these on the device and raise them up the stack.
    bool finite{true}, *device_finite;
    cudaMalloc(&device_finite, sizeof(bool));
    cudaMemcpy(device_finite, &finite, sizeof(bool), cudaMemcpyHostToDevice);
    kernel::check_finite<System><<<grid_size, block_dim>>>(fields, device_finite);
    cudaMemcpy(&finite, device_finite, sizeof(bool), cudaMemcpyDeviceToHost);

    if (not finite)
    {
        std::string message = "an unknown numerical error occurred during simulation";
        throw kernel::CudaError(message);
    }

    current_step += nsteps;
}

template <typename System>
State Reactor<System>::get_field(Scalar* field) const
{
    auto out = State(nrows, ncols);
    cudaMemcpy(out.data(), field, mem_size, cudaMemcpyDeviceToHost);
    return out;
}

template <typename System>
void Reactor<System>::set_field(const State& source, Scalar* destination)
{
    cudaMemcpy(destination, source.data(), mem_size, cudaMemcpyHostToDevice);
}

template <typename System>
int Reactor<System>::step() const
{
    return current_step;
}

template <typename System>
Scalar Reactor<System>::time() const
{
    return static_cast<Scalar>(current_step) * dt;
}


// Define the systems here so CUDA compiler (nvcc) knows to compile them.
template class Reactor<CellPolarisation>;
template class Reactor<ActiveModelB>;
template class Reactor<ToyModel>;