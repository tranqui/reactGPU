#ifndef REACTGPU_CUDA_H
#define REACTGPU_CUDA_H

#include <Eigen/Eigen>

template <typename Scalar>
constexpr Scalar square(Scalar x)
{
    return x*x;
}

using Scalar = double;

using State = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// template <typename Self>
class Reactor
{
private:
    Scalar dt, dx, dy, dxInv, dyInv;
    int nrows, ncols;
    size_t pitch_width, mem_size;

    Scalar Du, Dv, k;
    size_t current_step;

    size_t pitch_u, pitch_v;
    Scalar* u;
    Scalar* v;
    size_t pitch_rhs;
    Scalar* rhs;

public:
    Reactor(const Eigen::Ref<const State>& initial_u,
            const Eigen::Ref<const State>& initial_v,
            Scalar dt, Scalar dx, Scalar dy,
            Scalar Du, Scalar Dv, Scalar k,
            size_t current_step=0);
    ~Reactor();

    void run(int nsteps);
    State rhs_cpu() const;
    State rhs_gpu() const;
    State get_u() const;
    State get_v() const;
    size_t step() const;
    Scalar time() const;
};


#endif
