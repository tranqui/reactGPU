#pragma once
#include "reactor.cuh"

/**
 * Two-component model for cell polarisation adapted from:
 *   Mori, Y., Jilkine, A., and Edelstein-Keshet, L. (2008).
 *   doi: 10.1529/biophysj.107.120824
 */
struct CellPolarisation : public ChemicalFlux<CellPolarisation>
{
    inline static constexpr std::array<Scalar,1> device_parameters(Scalar k)
    {
        return {k};
    }

    inline static constexpr auto chemical_flux(Scalar u, Scalar v, Scalar k)
    {
        Scalar R = (k + (1-k) * u*u / (1 + u*u))* v - u;
        return std::make_tuple(R, -R);
    }
};


/**
 * Uniformly conserved pattern-forming model inspired by
 *   Jacobs, B., Molenaar, J., and Deinum, E. E. (2019).
 *   doi: 10.1371/journal.pone.0213188
 *
 * We have modified the model to remove an unnecessary fourth component, and
 * to reuse the chemical flux of the Mori model for the conserved components.
 *
 * This model forms stable patterns when the diffusion coefficient of the third
 * non-conserved component is much larger (by ~an order of magnitude) than the
 * conserved components.
 */
struct JacobsModel : public ChemicalFlux<JacobsModel>
{
    static constexpr int n = 2; // Hill coefficient for chemical flux.

    inline static constexpr std::array<Scalar,4>
    device_parameters(Scalar k, Scalar c, Scalar d, Scalar psi)
    {
        return {k, c, d, psi};
    }

    inline static constexpr auto
    chemical_flux(Scalar u, Scalar v, Scalar w,
                  Scalar k, Scalar c, Scalar d, Scalar psi)
    {
        Scalar R1 = (k + (1-k) * std::pow(u, n) / (1 + std::pow(u, n))) * v - (1 + w)*u;
        Scalar R2 = c*u*(psi - w) - d*w;
        return std::make_tuple(R1, -R1, R2);
    }
};

/**
 * Globally conserved pattern-forming model inspired by
 *   Murray, S. M., and Sourjik V. (2017).
 *   doi: 10.1038/nphys4155
 *
 * This model forms patterns through a different mechanism from the JacobsModel,
 * that does not require strong heterogeneities in the diffusion coefficients.
 * Curiously, the conservation law is dispensed with at small length-scales,
 * only being restored globally..
 */
struct MurrayModel : public ChemicalFlux<MurrayModel>
{
    inline static constexpr std::array<Scalar,4>
    device_parameters(Scalar alpha, Scalar beta, Scalar gamma, Scalar delta)
    {
        return {alpha, beta, gamma, delta};
    }

    inline static constexpr auto
    chemical_flux(Scalar u, Scalar v, Scalar w,
                  Scalar alpha, Scalar beta, Scalar gamma, Scalar delta)
    {
        Scalar R1 = (-alpha - beta*v*v)*u + gamma*v;
        return std::make_tuple( R1 + delta*u, -R1 - delta*v, -delta*(u - v));
    }
};


/**
 *  A model with a finely-tuned chemical flux that, by construction, has an
 *  emergent Ginzburg-Landau bulk free energy, i.e.:
 *      $f(\phi) = -A\left( \frac{\phi^2}{2} - \frac{\phi^4}{4} \right)$
 *  We choose $A = 1/4$ for illustration purposes.
 *
 *  In addition, we fine-tune so that the surface term in the free energy,
 *      $\int d\vec{r} \, \kappa(\phi) \frac{|\nabla \phi|^2}{2}$
 *  has constant $\kappa(\phi) = \kappa$. We also tune the non-integrable
 *  square-gradient contribution to the emergent chemical potential with
 *  constant $\lambda(\phi) = \lambda$. Finely, we set the eigenvalues of
 *  the gradient (in chemical space) of the linear chemical flux
 *  $L = \nabla_C R$ by setting its trace and determinant to constants
 *  $\Tr{L} = X$ and $\det{L} = Y$. 
 *
 *  Note that to achieve the fine-tuning we require the diffusion coefficients,
 *    as the field coefficients emerges from both reaction and diffusion.
 */
struct ActiveModelB : public ChemicalFlux<ActiveModelB>
{
    static constexpr Scalar A = 0.25;

    inline static constexpr std::array<Scalar, 16>
    device_parameters(Scalar Du, Scalar Dv, Scalar Dw,
                      Scalar kappa, Scalar lambda, Scalar X, Scalar Y)
    {
        Scalar num0[2][2] = {
            {48*A*A*Du*Du*Du*Du*Dv*Dv
                + 12*A*Du*Du*Du*Dv*(2*Dv*lambda + A*(lambda - 12*kappa))*X
                + A*Du*Du*(-48*A*Dv*Dv*Dv*Dv - 24*Dv*Dv*Dv*(-4*A*Dw + lambda*X)
                           + 12*Dv*Dv*(-12*A*Dw*Dw + A*(12*kappa - lambda)*X
                           + Dw*lambda*X) - A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y
                           + 2*Dv*(12*kappa - lambda)*(-6*A*Dw*X + lambda*Y))
                + Dv*Dw*Dw*(2*A*lambda*(lambda - 12*kappa)*Y
                            + 3*Dv*(4*A*Dv*(-4*A*Dv + 8*A*Dw + lambda*X)
                            + lambda*lambda*Y))
                - 2*Du*Dw*(A*A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y
                           + 3*Dv*(A*lambda*(lambda - 12*kappa)*Y
                           + Dv*(2*A*A*(lambda - 12*kappa)*X
                                + 2*A*(8*A*Dv*Dv + Dv*(-16*A*Dw + lambda*X)
                                       + Dw*(8*A*Dw + lambda*X))
                                + lambda*lambda*Y))),
            std::sqrt(3)*(Dv*Dv*(Dw - 2*Du)*(Dw - 2*Du)*lambda*lambda*Y
                          + 2*A*Dv*(2*Du - Dw)*lambda*(2*Dv*(-Du + Dv)*Dw*X - 4*Dv*Dw*Dw*X
                                                       + Du*(2*Dv*(Du + Dv)*X + (lambda - 12*kappa)*Y))
                                     + A*A*(64*(Du - Dv)*Dv*Dv*Dw*Dw*Dw + 64*Dv*Dv*Dw*Dw*Dw*Dw
                                             - 4*Du*(Du - Dv)*Dv*Dw*(8*Dv*(Du + Dv) + (lambda - 12*kappa)*X)
                                             + 8*Dv*Dw*Dw*(-6*Du*Du*Dv + 2*Dv*Dv*Dv + Du*(-12*Dv*Dv + 12*kappa*X - lambda*X))
                                             + Du*Du*(4*Dv*(Du + Dv)*(4*Dv*(Du + Dv) + (lambda - 12*kappa)*X)
                                             + (lambda - 12*kappa)*(lambda - 12*kappa)*Y)))},
            {(-144*A*A*Du*Du*Du*Du*Dv*Dv + 12*A*A*Du*Du*Du*Dv*(24*Dv*(Dv - Dw) + (lambda - 12*kappa)*X)
                + A*Du*Du*(-12*Dv*(12*A*Dv*Dw*Dw + A*Dv*(12*Dv*Dv + (lambda - 12*kappa)*X)
                           - 3*Dw*(16*A*Dv*Dv + Dv*lambda*X + A*(lambda - 12*kappa)*X))
                           - A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y)
                - 2*A*Du*Dw*(2*A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y
                             + 3*Dv*(48*A*Dv*Dv*Dv - 6*Dv*(12*A*kappa - A*lambda + Dw*lambda)*X
                                     + 6*Dv*Dv*(-8*A*Dw + lambda*X) + (lambda - 12*kappa)*(-4*A*Dw*X + lambda*Y)))
                + Dw*Dw*(-4*A*A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y
                         - 3*Dv*(4*A*lambda*(lambda - 12*kappa)*Y
                         + Dv*(8*A*A*(lambda - 12*kappa)*X + 12*A*Dv*(4*A*Dv + lambda*X) + 3*lambda*lambda*Y)))) / std::sqrt(3),
             -48*A*A*Du*Du*Du*Du*Dv*Dv + 4*A*A*Du*Du*Du*Dv*(lambda - 12*kappa)*X
                + A*Du*Du*(48*A*Dv*Dv*Dv*Dv - 96*A*Dv*Dv*Dv*Dw
                           + 4*Dv*Dv*(36*A*Dw*Dw + 3*Dw*lambda*X + A*(lambda - 12*kappa)*X)
                           + A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y - 2*Dv*(12*kappa - lambda)*(2*A*Dw*X + lambda*Y))
                + 2*Du*Dw*(48*A*A*Dv*Dv*Dv*Dv + 6*A*Dv*Dv*Dv*(-16*A*Dw + lambda*X)
                           + A*A*(lambda - 12*kappa)*(lambda - 12*kappa)*Y
                           + A*Dv*(12*kappa - lambda)*(8*A*Dw*X - 3*lambda*Y)
                           + 3*Dv*Dv*(2*A*A*(lambda - 12*kappa)*X + 2*A*Dw*(8*A*Dw - lambda*X) + lambda*lambda*Y))
                + Dv*Dw*Dw*(48*A*A*Dv*Dv*Dv + 12*A*Dv*Dv*(-8*A*Dw + lambda*X)
                            + 2*A*(12*kappa - lambda)*(8*A*Dw*X + lambda*Y)
                            - Dv*(A*A*(96*kappa - 8*lambda)*X + 24*A*Dw*lambda*X + 3*lambda*lambda*Y))}};

        Scalar num2[2][2] = {
            {6*A*lambda*(-6*A*Du*Du*Du*Dv*X - Dv*Dw*Dw*lambda*Y
                         + Du*Dw*(6*A*Dv*Dv*X + 3*Dv*lambda*Y + 2*A*(lambda - 12*kappa)*Y)
                         + Du*Du*(A*(lambda - 12*kappa)*Y + Dv*(6*A*(Dv - Dw)*X + lambda*Y))),
            6*std::sqrt(3)*A*Du*lambda*(-2*A*Du*Du*Dv*X + Dv*Dw*(-2*A*Dv*X + 4*A*Dw*X + lambda*Y)
                                        - Du*(A*(lambda - 12*kappa)*Y + 2*Dv*(A*(Dv - Dw)*X + lambda*Y)))},
            {(6*A*(Du + 2*Dw)*lambda*(-6*A*Du*Du*Dv*X
                                      + A*Du*(6*Dv*(Dv - Dw)*X + (lambda - 12*kappa)*Y)
                                      + Dw*(6*A*Dv*Dv*X + 3*Dv*lambda*Y + 2*A*(lambda - 12*kappa)*Y))) / std::sqrt(3),
            6*A*lambda*(-2*A*Du*Du*Du*Dv*X + Dv*Dw*Dw*(-4*A*Dv*X + 8*A*Dw*X + lambda*Y)
                        + Du*Dw*(2*A*(12*kappa - lambda)*Y + Dv*(-6*A*Dv*X + 8*A*Dw*X - 3*lambda*Y))
                        - Du*Du*(A*(lambda - 12*kappa)*Y + Dv*(2*A*(Dv + Dw)*X + lambda*Y)))}};

        Scalar num4[2][2] = {
            {-9*A*A*Du*(Du + 2*Dw)*lambda*lambda*Y,
              9*std::sqrt(3)*A*A*Du*Du*lambda*lambda*Y},
            {(-9*A*A*(Du + 2*Dw)*(Du + 2*Dw)*lambda*lambda*Y) / std::sqrt(3),
              9*A*A*Du*(Du + 2*Dw)*lambda*lambda*Y}};

        Scalar den0 = 12*A*(Dv - 2*Dw - 2*Du) * kappa + (3*Dv * (Du - Dv + Dw) + A*(2*Du - Dv + 2*Dw)) * lambda;
        Scalar den2 = 3*A*(Dv - 2*Dw - 2*Du) * lambda;
        den0 *= 8*A*Dv*(Du*Du - Dw*Dw);
        den2 *= 8*A*Dv*(Du*Du - Dw*Dw);
        // Scalar den = 8*A*Dv*(Du*Du - Dw*Dw) * (den0 + den2 * phi*phi);

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                num0[i][j] /= den0;
                num2[i][j] /= den0;
                num4[i][j] /= den0;
            }
        }
        den2 /= den0;

        // return {{{(num0[0][0] + num2[0][0] * phi*phi + num4[0][0] * phi*phi*phi*phi) / den,
        //           (num0[0][1] + num2[0][1] * phi*phi + num4[0][1] * phi*phi*phi*phi) / den},
        //          {(num0[1][0] + num2[1][0] * phi*phi + num4[1][0] * phi*phi*phi*phi) / den,
        //           (num0[1][1] + num2[1][1] * phi*phi + num4[1][1] * phi*phi*phi*phi) / den}}};

        return {Du, Dv, Dw, den2,
                num0[0][0], num0[0][1], num0[1][0], num0[1][1],
                num2[0][0], num2[0][1], num2[1][0], num2[1][1],
                num4[0][0], num4[0][1], num4[1][0], num4[1][1]};
    }

    inline static constexpr
    std::array<std::array<Scalar, 2>, 2>
    linear_flux(Scalar phi, Scalar den2,
                Scalar num0_00, Scalar num0_01, Scalar num0_10, Scalar num0_11,
                Scalar num2_00, Scalar num2_01, Scalar num2_10, Scalar num2_11,
                Scalar num4_00, Scalar num4_01, Scalar num4_10, Scalar num4_11)
    {
        Scalar den = 1 + den2 * phi*phi;

        return {{{(num0_00 + num2_00 * phi*phi + num4_00 * phi*phi*phi*phi) / den,
                  (num0_01 + num2_01 * phi*phi + num4_01 * phi*phi*phi*phi) / den},
                 {(num0_10 + num2_10 * phi*phi + num4_10 * phi*phi*phi*phi) / den,
                  (num0_11 + num2_11 * phi*phi + num4_11 * phi*phi*phi*phi) / den}}};
    }

    inline static constexpr auto
    chemical_flux(Scalar u, Scalar v, Scalar w,
                  Scalar Du, Scalar Dv, Scalar Dw, Scalar den2,
                  Scalar num0_00, Scalar num0_01, Scalar num0_10, Scalar num0_11,
                  Scalar num2_00, Scalar num2_01, Scalar num2_10, Scalar num2_11,
                  Scalar num4_00, Scalar num4_01, Scalar num4_10, Scalar num4_11)
    {
        Scalar phi = u + v + w; // conserved order parameter (the density).

        // Nullcline position at this phi:
        Scalar u_eq = phi * (A*(phi*phi - 1) - Dv) / (Du - Dw);
        Scalar v_eq = phi;
        Scalar w_eq = phi - v_eq - u_eq;

        // Perturbation from nullcline:
        Scalar drho[3] = {u - u_eq, v - v_eq, w - w_eq};

        // Coordinates within reactive subspace (i.e. within level sets of phi):
        Scalar x[2] = {(drho[1] - drho[0]) / std::sqrt(2),
                       (2*drho[2] - drho[1] - drho[0]) / std::sqrt(6)};

        // Chemical flux is linear in deviations from nullcline.
        auto L = linear_flux(phi, den2, num0_00, num0_01, num0_10, num0_11,
                                        num2_00, num2_01, num2_10, num2_11,
                                        num4_00, num4_01, num4_10, num4_11);
        Scalar R[2] = {L[0][0] * x[0] + L[0][1] * x[1],
                       L[1][0] * x[0] + L[1][1] * x[1]}; // matrix multiplication

        // Convert back to (u,v,w) coordinates:
        R[0] /= std::sqrt(2);
        R[1] /= std::sqrt(6);
        return std::make_tuple(-R[0] - R[1], R[0] - R[1], 2*R[1]);
    }
};