#pragma once
#include <Eigen/Eigen>
#include <tuple>
#include <string>

namespace kernel
{
    class CudaError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    void check_error();
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
};

struct CellPolarisation : public ChemicalFlux<CellPolarisation>
{
    inline static constexpr auto chemical_flux(Scalar u, Scalar v, Scalar k)
    {
        Scalar R = (k + (1-k) * u*u / (1 + u*u))* v - u;
        return std::make_tuple(R, -R);
    }
};


class Reactor
{
private:
    Scalar dt, dx, dy, dxInv, dyInv;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    Scalar Du, Dv; // <- change to std::array<Scalar, m>?
    Scalar k;
    size_t current_step;

    size_t pitch_u, pitch_v;
    Scalar* u;
    Scalar* v;

public:
    inline static constexpr
    auto chemical_flux(Scalar u, Scalar v, Scalar k)
    {
        return CellPolarisation::chemical_flux(u, v, k);
    }

    Reactor(const Eigen::Ref<const State>& initial_u,
            const Eigen::Ref<const State>& initial_v,
            Scalar dt, Scalar dx, Scalar dy,
            Scalar Du, Scalar Dv, Scalar k,
            size_t current_step=0);
    ~Reactor();

    void run(int nsteps);
    State get_u() const;
    State get_v() const;
    size_t step() const;
    Scalar time() const;
};
