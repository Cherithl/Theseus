// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "NavierStokesFlux.hpp"
#include "theseus_kernels.hpp"

namespace Theseus
{

  namespace LaxFriedrichsFlux
  {
  
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasT>
    MFEM_HOST_DEVICE
    inline static void ComputeVolumeFluxKernel(const GasT &gas,
                                                 const mfem::real_t* q1,
                                                 const mfem::real_t* q2,
                                                 const mfem::real_t* met1,
                                                 const mfem::real_t* met2,
                                                 mfem::real_t* F_tilde)
    {
      const int dim = gas.dim();
      const int neq = gas.num_equations();

      // mean metric row
      mfem::real_t met[3] = {0,0,0};
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      mfem::real_t inv_flux_1[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t inv_flux_2[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t inv_flux_bar[Theseus::MAXEQ];

      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q1, inv_flux_1);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q2, inv_flux_2);
      for(int ieq=0;ieq < neq;ieq++){
        inv_flux_bar[ieq] = 0;
        for(int idim = 0;idim < dim;idim++){
          inv_flux_bar[ieq] += 0.5*(inv_flux_1[ieq][idim] + inv_flux_2[ieq][idim])*met[idim];
        }
      }

      for(int ieq = 0;ieq < neq;ieq++){
        F_tilde[ieq] = inv_flux_bar[ieq];
      }

    }
  
    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static void
    ComputeFaceFluxKernel(const GasModelT &gasModel,
                          const mfem::real_t *state1,
                          const mfem::real_t *state2,
                          const mfem::real_t *nor,
                          mfem::real_t *flux)
    {
      const int neq = gasModel.num_equations();

      const auto left = NavierStokesFlux::MakeInviscidNormalFluxState(gasModel, state1);
      const auto right = NavierStokesFlux::MakeInviscidNormalFluxState(gasModel, state2);
      const mfem::real_t lambda_max = NavierStokesFlux::MaximumNormalWaveSpeed(
        gasModel, state1, state2, nor);
      for (int equation = 0; equation < neq; ++equation)
        {
          const mfem::real_t normal_left =
            NavierStokesFlux::ComputeInviscidNormalFluxEquation(gasModel, left, nor, equation);
          const mfem::real_t normal_right =
            NavierStokesFlux::ComputeInviscidNormalFluxEquation(gasModel, right, nor, equation);
          flux[equation] = 0.5 * (normal_left + normal_right) -
            0.5 * lambda_max * (state2[equation] - state1[equation]);
        }
    }

    struct InviscidFlux {
      template<typename GasModelT>
      MFEM_HOST_DEVICE inline void ComputeVolumeFlux(const GasModelT &gasModel,
                                                       const mfem::real_t *q1, const mfem::real_t *q2,
                                                       const mfem::real_t *met1, const mfem::real_t *met2,
                                                       mfem::real_t *F_tilde) const{
        ComputeVolumeFluxKernel(gasModel, q1, q2, met1, met2, F_tilde);
      }
      template<typename GasModelT>
      MFEM_HOST_DEVICE inline void ComputeFaceFlux(const GasModelT &gasModel,const mfem::real_t *qminus,
                                                     const mfem::real_t *qplus, const mfem::real_t *nor,
                                                     mfem::real_t *flux) const {
        ComputeFaceFluxKernel(gasModel, qminus, qplus, nor, flux);
      }
    };
  };
}
