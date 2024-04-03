#pragma once
#include "utilities.h"
#include <Eigen/Eigen>


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

/**
 *  Implementation must define two functions: device_parameters and chemical_flux.
 *  - chemical_flux: the reaction part of reaction-diffusion that defines the model.
 *      This takes the local fields as arguments, as well as a certain number of parameters
 *      and returns a py::tuple with the reaction rate for each field.
 *  - device_parameters: calculates the parameters passed ot the GPU device from a set
 *      of host parameters. This allows the host interface to use a convenient set of host
 *      parameters, which are processed before execution on the device to optimise kernel
 *      execution.
 */
template <typename Implementation>
struct ChemicalFlux
{
    template <typename... Args>
    inline auto operator()(Args&&... args) const
    {
        return static_cast<Implementation*>(this)->chemical_flux(std::forward<Args>(args)...);
    }

    // Number of parameters describing the top-level model.
    static constexpr std::size_t nparams_host()
    {
        using device_parameters_type = decltype(&Implementation::device_parameters);
        return Arity<device_parameters_type>::value;
    }

    // Number of (post-processed) parameters sent to the GPU device.
    static constexpr std::size_t nparams_device()
    {
        using device_parameters_type = decltype(&Implementation::device_parameters);
        using HostParameters = repeat_type<Scalar, nparams_host()>;
        return Cardinality<device_parameters_type, HostParameters>::value;
    }

    // Arguments to chemical_flux include the fields *and* the device parameters.
    static constexpr std::size_t nargs_flux()
    {
        using chemical_flux_type = decltype(&Implementation::chemical_flux);
        return Arity<chemical_flux_type>::value;
    }

    // We can infer the number of fields defined in the model using the other information above.
    static constexpr std::size_t nfields()
    {
        return nargs_flux() - nparams_device();
    }
};


/// Simulation controller


template <typename System>
class Reactor
{
public:
    static constexpr std::size_t nfields = System::nfields();
    static constexpr std::size_t nparams_host = System::nparams_host();
    static constexpr std::size_t nparams_device = System::nparams_device();

    using DeviceFields = repeat_type<Scalar*, nfields>;
    using HostFields = repeat_type<State, nfields>;
    using InitialState = repeat_type<StateRef, nfields>;

protected:
    Scalar dt, dx, dy, dxInv, dyInv;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    std::array<Scalar, nfields> D;
    std::array<Scalar, nparams_host> flux_parameters;
    int current_step;

    std::array<size_t, nfields> pitch;
    DeviceFields fields;

public:
    // Copy constructors are not safe because GPU device memory will not be copied.
    Reactor(const Reactor<System>&) = delete;
    Reactor<System>& operator=(const Reactor<System>&) = delete;
    // Move constructors are fine though.
    Reactor<System>& operator=(Reactor<System>&&) noexcept = default;
    Reactor(Reactor<System>&&) noexcept;

    Reactor(const InitialState& initial_fields,
            Scalar dt, Scalar dx, Scalar dy,
            std::array<Scalar, nfields> D,
            std::array<Scalar, nparams_host> params,
            int current_step=0);

    // Option to pass lists of Scalars rather than STL containers for the parameters.
    template<typename... Args>
    Reactor(const InitialState& initial_fields,
            Scalar dt, Scalar dx, Scalar dy, Args&&... args)
            : Reactor(initial_fields, dt, dx, dy,
                      unpack_as_array<Scalar, 0, nfields>(std::forward<Args>(args)...),
                      unpack_as_array<Scalar, nfields, nparams_host>(std::forward<Args>(args)...),
                      unpack_remaining<nfields + nparams_host>(std::forward<Args>(args)...))
    { }

    ~Reactor();

    void run(int nsteps);

    template <std::size_t index>
    State get_field() const
    {
        return get_field(std::get<index>(fields));
    }

    template <std::size_t... I>
    auto get_fields_implementation(std::index_sequence<I...>) const
    {
        return std::make_tuple(get_field<I>()...);
    }

    auto get_fields() const
    {
        return get_fields_implementation(std::make_index_sequence<std::tuple_size_v<decltype(fields)>>{});
    }

    int step() const;
    Scalar time() const;

    auto host_parameters() const
    {
        return flux_parameters;
    }

    std::array<Scalar, nparams_device> device_parameters() const
    {
        return std::apply(System::device_parameters, flux_parameters);
    }

protected:
    State get_field(Scalar* field) const;
    void set_field(const State& source, Scalar* destination);

    template <std::size_t index>
    void set_field(const State& source)
    {
        set_field(source, std::get<index>(fields));
    }
};

#include "models.h"