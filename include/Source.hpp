// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "Utilities.hpp"

namespace Theseus
{
  namespace Source
  {
    template<typename DeviceContext>
    MFEM_HOST_DEVICE
    inline void TargetStateSource(const DeviceContext &dc, const mfem::real_t *Ue, mfem::real_t *dUe)
    {
      const int dof = dc.ndof_scalar_el;
      const int neq = dc.num_equations;
      auto gas = dc.gas;
      const int dim = gas.dim();

      const mfem::real_t *volume_avg_state = dc.volume_avg_state_d;
      const mfem::real_t *target_state = dc.target_state_d;

      mfem::real_t state[Theseus::MAXEQ];
      mfem::real_t source[Theseus::MAXEQ];

      source[0] = (target_state[0] - volume_avg_state[0])*dc.tau_inv;
      source[1] = (target_state[1] - volume_avg_state[1])*dc.tau_inv;
      source[dim+1] = (target_state[dim+1] - volume_avg_state[dim+1])*dc.tau_inv;

      for(int point =0; point < dof; point++)
      {
        Kernels::el_gather_state(Ue, dof, neq, point, state);
        PointStateView S{state};
        source[dim+1] += gas.velocity(S,0) * source[1];
        Kernels::el_scatter_add(source, dof, neq, point, 1.0, dUe);
      }
    };

    template<typename DeviceContext, typename OperatorContext>
    inline void PreProcessSourceTerms(DeviceContext &dc, OperatorContext &op_cache, const mfem::real_t* Ue_d, MPI_Comm comm)
    {
      if(dc.use_target_state_source)
      {
        Utilities::ComputeVolumeAverages(dc, op_cache, Ue_d, comm);
        const mfem::real_t alpha = 0.3; // CL ALERT : This is a tuning parameter for the target state source term
        dc.tau_inv = 1.0 / (alpha *op_cache.current_dt);
      }
    };

    template<typename DeviceContext>
    MFEM_HOST_DEVICE
    inline void AddSourceTerms(const DeviceContext &dc, const mfem::real_t *Ue, mfem::real_t *dUe)
    {
      if(dc.use_target_state_source && dc.tau_inv > 0.0)
      {
        Source::TargetStateSource(dc, Ue, dUe);
      }
    };
  }
}