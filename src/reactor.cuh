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
using StateRef = Eigen::Ref<const State>;


/// System definitions

template <typename Implementation>
struct ChemicalFlux
{
    template <typename... Args>
    inline auto operator()(Args&&... args) const
    {
        return static_cast<Implementation*>(this)->chemical_flux(std::forward<Args>(args)...);
    }

    static constexpr std::size_t nparams = Arity<decltype(&Implementation::chemical_flux)>::value - Implementation::nfields;
};

struct CellPolarisation : public ChemicalFlux<CellPolarisation>
{
    static constexpr std::size_t nfields = 2;

    inline static constexpr auto chemical_flux(Scalar u, Scalar v, Scalar k)
    {
        Scalar R = (k + (1-k) * u*u / (1 + u*u))* v - u;
        return std::make_tuple(R, -R);
    }
};

struct ToyModel : public ChemicalFlux<ToyModel>
{
    static constexpr std::size_t nfields = 3;

    inline static constexpr auto chemical_flux(Scalar u, Scalar v, Scalar w, Scalar k)
    {
        Scalar R = (k + (1-k) * u*u / (1 + u*u))* v - u;
        return std::make_tuple(R, -R, 0);
    }
};


/// Simulation controller


namespace details
{
    template<typename T, typename... Ts>
    struct tuple_repeat;

    template<typename T, std::size_t... Is>
    struct tuple_repeat<T, std::index_sequence<Is...>>
    {
        // comma operator (Is, T) discards the index_sequence in favour of T, but makes
        // the sequence a part of the expression so we can expand with ... thereby creating
        // N copies. std::decay_t is needed otherwise the expression can become an rvalue. 
        using type = std::tuple<std::decay_t<decltype(Is, std::declval<T>())>...>;
    };
}

template <typename T, std::size_t N>
using tuple_repeat = typename details::tuple_repeat<T, std::make_index_sequence<N>>::type;


template <typename System>
class Reactor
{
public:
    static constexpr auto nfields = System::nfields;
    static constexpr auto nparams = System::nparams;

    using DeviceFields = tuple_repeat<Scalar*, nfields>;
    // using HostFields = tuple_repeat<State, nfields>;
    using InitialState = tuple_repeat<StateRef, nfields>;

protected:
    Scalar dt, dx, dy, dxInv, dyInv;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    std::array<Scalar, nfields> D;
    std::array<Scalar, nparams> flux_parameters;
    size_t current_step;

    std::array<size_t, nfields> pitch;
    DeviceFields fields;

public:
    Reactor(const InitialState& initial_fields,
            Scalar dt, Scalar dx, Scalar dy,
            std::array<Scalar, nfields> D,
            std::array<Scalar, nparams> params,
            size_t current_step=0);

    // Option to pass lists of Scalars rather than STL containers for the parameters.
    template<typename... Args>
    Reactor(const InitialState& initial_fields,
            Scalar dt, Scalar dx, Scalar dy, Args&&... args)
            : Reactor(initial_fields, dt, dx, dy,
                      unpack<0, nfields>(std::forward<Args>(args)...),
                      unpack<nfields, nparams>(std::forward<Args>(args)...),
                      unpack_remaining<nfields + nparams>(std::forward<Args>(args)...))
    { }

    ~Reactor();

    void run(int nsteps);
    State get_field(Scalar* field) const;

    template <std::size_t index> State get_field() const
    {
        return get_field(std::get<index>(fields));
    }

    template <std::size_t... Is>
    auto get_fields_implementation(std::index_sequence<Is...>) const
    {
        return std::make_tuple(get_field<Is>()...);
    }

    auto get_fields() const
    {
        return get_fields_implementation(std::make_index_sequence<std::tuple_size<decltype(fields)>::value>{});
    }

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
