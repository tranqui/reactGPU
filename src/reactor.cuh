#pragma once
#include <Eigen/Eigen>
#include <tuple>
#include <string>


// A utility to count number of functor parameters.
template <typename T> struct Arity;
template <typename R, typename... Args>
struct Arity<R(*)(Args...)>
{
    static constexpr size_t value = sizeof...(Args);
};


namespace kernel
{
    // Generic python-safe exception to contain errors on GPU execution.
    // Note that the actual catching and throwing of errors has to handle on the host if
    // the CUDA kernel sets the error flag - this is handled in kernel::throw_errors().
    class CudaError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    // Check CUDA for errors after GPU execution and throw them.
    void throw_errors();
}


using Scalar = double;

using State = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Implementation>
struct ChemicalFlux
{
    template <typename... Args>
    inline auto operator()(Args&&... args) const
    {
        return static_cast<Implementation*>(this)->chemical_flux(std::forward<Args>(args)...);
    }

    static constexpr std::size_t nparams = Arity<decltype(&Implementation::chemical_flux)>::value - Implementation::nspecies;
};

struct CellPolarisation : public ChemicalFlux<CellPolarisation>
{
    static constexpr std::size_t nspecies = 2;

    inline static constexpr auto chemical_flux(Scalar u, Scalar v, Scalar k)
    {
        Scalar R = (k + (1-k) * u*u / (1 + u*u))* v - u;
        return std::make_tuple(R, -R);
    }
};


template <typename System>
class Reactor
{
protected:
    Scalar dt, dx, dy, dxInv, dyInv;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    static constexpr auto nspecies = System::nspecies;
    static constexpr auto nparams = System::nparams;
    std::array<Scalar, nspecies> D;
    std::array<Scalar, nparams> flux_parameters;
    size_t current_step;

    size_t pitch_u, pitch_v;
    Scalar* u;
    Scalar* v;

public:

    Reactor(const Eigen::Ref<const State>& initial_u,
            const Eigen::Ref<const State>& initial_v,
            Scalar dt, Scalar dx, Scalar dy,
            std::array<Scalar, nspecies> D, std::array<Scalar, nparams> params,
            size_t current_step=0);

    // Option to pass lists of Scalars rather than STL containers for the parameters.
    template<typename... Args>
    Reactor(const Eigen::Ref<const State>& initial_u,
            const Eigen::Ref<const State>& initial_v,
            Scalar dt, Scalar dx, Scalar dy, Args&&... args)
            : Reactor(initial_u, initial_v, dt, dx, dy,
                      unpack<0,nspecies>(std::forward<Args>(args)...),
                      unpack<nspecies,nparams>(std::forward<Args>(args)...),
                      unpack_remaining<nspecies+nparams>(std::forward<Args>(args)...))
    { }

    ~Reactor();

    void run(int nsteps);
    State get_u() const;
    State get_v() const;
    size_t step() const;
    Scalar time() const;

private:
    // Helper function to unpack N elements into one of the parameter arrays.
    // This allows for more flexible constructors that can take lists of Scalars rather
    // than having to cast them to an std::array.
    template <size_t Start, size_t N, typename... Args>
    static auto unpack(Args&&... args)
    {
        static_assert(sizeof...(Args) >= N + Start, "Not enough arguments provided.");
        return unpack_implementation<Start, N>(std::make_index_sequence<N>{},
                                               std::forward<Args>(args)...);
    }
    template<size_t Start, size_t N, size_t... Is, typename... Args>
    static std::array<Scalar, N>
    unpack_implementation(std::index_sequence<Is...>, Args&&... args)
    {
        return {{(std::get<Start + Is>(std::forward_as_tuple(args...)))...}};
    }

    // Helper to unpack current_step from the remaining arguments
    // Extract the remaining arguments into a tuple, if any
    template<size_t Start, typename... Args>
    static auto unpack_remaining(Args&&... args)
    {
        static_assert(sizeof...(Args) >= Start, "Not enough arguments for unpacking remaining.");
        if constexpr (sizeof...(Args) > Start)
        {
            return std::get<Start>(std::forward_as_tuple(args...));
        }
        else return std::make_tuple();
    }
};
