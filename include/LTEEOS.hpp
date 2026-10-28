// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"
#include "LTETable.hpp"
#include "LTETemperatureRecovery.hpp"

using namespace Theseus::LTETable;

namespace Theseus
{

  // ============================================================================
  // LTEGasEOS: Gas-mixture in local thermodynamic equilibrium
  // ============================================================================
  struct LTEGasEOS
  {

    MFEM_HOST_DEVICE
    mfem::real_t isothermal_wall_beta(const PhysicsConstants &,
                                     mfem::real_t wall_temperature) const
    {
      return 1.0 / wall_temperature;
    }

    // ---- helpers on conservative state --------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t R_gas(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S, const LTETables &lteTables) const
    {
      printf("LTEGasEOS::R_gas is not implemented for LTEGasEOS\n");
#ifdef __CUDA_ARCH__
      asm("trap;");
#else
      MFEM_ABORT("LTEGasEOS::R_gas is not implemented for LTEGasEOS");
#endif
      return pressure(phys, L, S, lteTables) / (density(phys, L, S, lteTables) * temperature(phys, L, S, lteTables));
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t density(const PhysicsConstants &phys, const StateLayout &L,
                                const StateView &S, const LTETables &lteTables) const
    {
      return S.mass(L); // this is "rho" (mass density)
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t rhoE(const PhysicsConstants &phys, const StateLayout &L,
                             const StateView &S) const
    {
      return S.energy(L);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t momentum_sq(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S) const
    {
      const int dim = L.dim;   // uses state layout
      mfem::real_t m2 = 0;
      for (int d = 0; d < dim; ++d)
        {
          const mfem::real_t m = S.momentum(L,d);
          m2 += m * m;
        }
      return m2;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t kinetic_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                               const StateView &S, const LTETables &lteTables) const
    {
      // 0.5 * rho * |u|^2 = 0.5 * |rho*u|^2 / rho
      const mfem::real_t rho  = density(phys, L, S, lteTables);
      const mfem::real_t m2   = momentum_sq(phys, L, S);
      return 0.5 * m2 / rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_density(const PhysicsConstants &phys, const StateLayout &L,
                                                const StateView &S, const LTETables &lteTables) const
    {
      // rho*e = rho*E - 0.5*rho*|u|^2
      return rhoE(phys, L, S) - kinetic_energy_density(phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_pressure(const PhysicsConstants &phys, const StateLayout &L,
                                                      const StateView &S, mfem::real_t pressure_target,
                                                      const LTETables &lteTables) const
    {
//       mfem::real_t U[Theseus::MAXEQ];
//       PointStateViewRW S_dummy(U);
//       S_dummy.set_mass(L, S.mass(L));

//       for(int idim = 0; idim < L.dim; idim++)
//         {
//           S_dummy.set_momentum(L, idim, S.momentum(L, idim));
//         }
//       S_dummy.set_energy(L, S.energy(L));

//       // Secant Method to find internal energy that matches a target pressure
//       mfem::real_t tol   = 1e-12;
//       mfem::real_t denom = 0.0;
//       mfem::real_t ie_old = pressure_target * 3; // initial guess for internal energy (CL NOTE : make sure this is inside the table range)
//       mfem::real_t ie_new = ie_old*1.01 + tol;

//       mfem::real_t ke = kinetic_energy_density(phys, L, S, lteTables);

//       mfem::real_t f_old, f_new, ie_update;

//       for(int iter = 0; iter < 100; iter++)
//         {
//           S_dummy.set_energy(L, ie_old+ke);
//           f_old = property_lookup(lteTables.L.P_idx, phys, L, S_dummy, lteTables) - pressure_target;

//           S_dummy.set_energy(L, ie_new+ke);
//           f_new = property_lookup(lteTables.L.P_idx, phys, L, S_dummy, lteTables) - pressure_target;

//           denom = f_new - f_old;

//           if( Theseus::Kernels::rabs(f_new) < tol || Theseus::Kernels::rabs(denom) < 1e-12)
//             {
//               break;
//             }

//           ie_update = ie_new - f_new * (ie_new - ie_old) / denom;

//           ie_old = ie_new;
//           ie_new = ie_update;
//           if(iter == 99){
// #ifdef __CUDA_ARCH__
//               printf("Secant method did not converge in internal_energy_from_pressure");
//               asm("trap;");
// #else
//             MFEM_ABORT("Secant method did not converge in internal_energy_from_pressure");
// #endif
//           }
//         }
#ifdef __CUDA_ARCH__
      printf("internal_energy_from_pressure is not implemented for LTEGasEOS\n");
      asm("trap;");
#else
      MFEM_ABORT("internal_energy_from_pressure is not implemented for LTEGasEOS");
#endif
      mfem::real_t ie_new = 0.0; // placeholder to avoid compiler warning
      return ie_new;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t specific_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                                 const StateView &S, const LTETables &lteTables) const
    {
      // e = (rho*e) / rho
      const mfem::real_t rho  = density(phys, L, S, lteTables);
      const mfem::real_t rhoe = internal_energy_density(phys, L, S, lteTables);
      return rhoe / rho;
    }

    // ---- primary EOS interface ----------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_helmholtz(const PhysicsConstants &phys, const StateLayout &L,
                                               const StateView &S1, const LTETables &lteTables,
                                               const mfem::real_t *temp = nullptr) const
    {
      // e = H - T * H_T
      mfem::real_t coeffs[16];
      mfem::real_t xi, eta, drho, dT;

      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S1, lteTables);

      GetHermiteCoeffs(phys, L, S1, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 0);
      mfem::real_t H_T = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 1);
      return H - T * H_T;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t pressure(const PhysicsConstants &phys, const StateLayout &L,
                                 const StateView &S, const LTETables &lteTables,
                                 const mfem::real_t *temp = nullptr,
                                 mfem::real_t *pres_der = nullptr) const
    {
      // p = rho^2 * dH/drho
      mfem::real_t coeffs[16];
      mfem::real_t xi, eta, drho, dT;

      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);

      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t dH_drho = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 0);
      mfem::real_t rho = density(phys, L, S, lteTables);

      if(pres_der != nullptr)
        {
          mfem::real_t H_rr = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 2, 0);
          mfem::real_t H_rT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 1);
          pres_der[0] = 2 * rho * dH_drho + rho * rho * H_rr;
          pres_der[1] = rho * rho * H_rT;
        }
      return rho*rho*dH_drho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gamma(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S, const LTETables &lteTables,
                              const mfem::real_t *temp = nullptr) const
    {
      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);
      return cp(phys, L, S, lteTables, &T) / cv_from_helmholtz(phys, L, S, lteTables, &T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temperature(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S, const LTETables &lteTables) const
    {
      return temp_from_internal_energy(phys, L, S, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const PhysicsConstants &phys, const StateLayout  &L,
                                 const StateView &S, const mfem::real_t *grad_rho,
                                 const mfem::real_t *grad_p, mfem::real_t *grad_t,
                                 const LTETables &lteTables) const
    {
      for(int i = 0; i < L.dim; i++){
        grad_t[i] = grad_p[i]; // CL NOTE : we store T_xi in contiguous gradient array for LTE (W=[rho, u, v, w , T])
      }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t sound_speed(const PhysicsConstants &phys, const StateLayout &L,
                                    const StateView &S, const LTETables &lteTables,
                                    const mfem::real_t *temp = nullptr) const
    {
      // c^2 = 2rho H_rho + rho^2 H_rr - rho^2 H_rT^2 / H_TT
      mfem::real_t coeffs[16];
      mfem::real_t xi , eta, drho, dT;

      mfem::real_t rho = density(phys, L, S, lteTables);
      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);

      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H_r = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 0);
      mfem::real_t H_rr = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 2, 0);
      mfem::real_t H_rT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 1);
      mfem::real_t H_TT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 2);

      mfem::real_t c2 = 2.0*rho*H_r + rho*rho*H_rr - rho*rho*H_rT*H_rT/H_TT;
      return Kernels::rsqrt(c2);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cv_from_helmholtz(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S, const LTETables &lteTables,
                           const mfem::real_t *temp = nullptr) const
    {
      // cv = -T H_TT
      mfem::real_t coeffs[16];
      mfem::real_t xi , eta, drho, dT;

      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);

      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H_TT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 2);
      return -T * H_TT;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cp(const PhysicsConstants &phys, const StateLayout &L,
                           const StateView &S, const LTETables &lteTables,
                           const mfem::real_t *temp = nullptr) const
    {
      mfem::real_t coeffs[16];
      mfem::real_t xi , eta, drho, dT;

      mfem::real_t rho = density(phys, L, S, lteTables);
      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);

      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H_TT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 2);
      mfem::real_t H_rT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 1);
      mfem::real_t P_rho = 0.0; // CL ALERT : Have to implement function P_rho
#ifdef __CUDA_ARCH__
      asm("trap;");
#else
      MFEM_ABORT("LTEGasEOS::cp is not implemented for LTEGasEOS");
#endif
      return -T * H_TT + T * rho * rho * H_rT * H_rT / P_rho;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t entropy(const PhysicsConstants &phys, const StateLayout &L,
                                const StateView &S, const LTETables &lteTables,
                                const mfem::real_t *temp = nullptr) const
    {
      mfem::real_t coeffs[16];
      mfem::real_t xi , eta, drho, dT;

      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);
      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H_T = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 1);
      return -H_T;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gibbs(const PhysicsConstants &phys, const StateLayout &L,
                              const StateView &S, const LTETables &lteTables,
                              const mfem::real_t *temp,
                              mfem::real_t *gibbs_der = nullptr) const
    {
      mfem::real_t coeffs[16];
      mfem::real_t xi , eta, drho, dT;

      mfem::real_t rho = density(phys, L, S, lteTables);
      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);

      GetHermiteCoeffs(phys, L, S, lteTables, T, xi, eta, drho, dT, coeffs);
      mfem::real_t H = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 0);

      if(gibbs_der != nullptr)
      {
        mfem::real_t H_r = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 0);
        mfem::real_t H_T = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 0, 1);
        mfem::real_t H_rr = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 2, 0);
        mfem::real_t H_rT = Evaluate_Hermite_Polynomial(coeffs, xi, eta, drho, dT, 1, 1);
        gibbs_der[0] = 2*H_r + rho * H_rr;
        gibbs_der[1] = H_T + rho * H_rT;
      }

      return H + pressure(phys, L, S, lteTables, &T)/rho;
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const PhysicsConstants &phys, const StateLayout &L,
                              const InStateView &S, OutStateView &E,
                              const LTETables &lteTables,
                              const mfem::real_t *temp = nullptr) const
    {
      const mfem::real_t T = (temp != nullptr) ? *temp : temperature(phys, L, S, lteTables);
      const mfem::real_t beta = 1/T;
      const mfem::real_t g = gibbs(phys, L, S, lteTables, &T);
      const mfem::real_t ke = kinetic_energy_density(phys, L, S, lteTables)/S.mass(L);
      const mfem::real_t ent_1 = beta*(g - ke);

      E.set_mass(L, ent_1);
      int dim = L.dim;
      for(int idim = 0;idim < dim;idim++){
        E.set_momentum(L, idim, beta * S.velocity(L, idim));
      }
      E.set_energy(L, -beta);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const PhysicsConstants &phys, const StateLayout &L,
                                          const InStateView &S, const InStateView &dE,
                                          OutStateView &dPrim, const LTETables &lteTables) const
    {

      const mfem::real_t T = temperature(phys, L, S, lteTables);

      dPrim.set_mass(L, 0.0); // CL NOTE : won't be using density gradient (if W = [rho, u, v, w , T])
      int dim = L.dim;
      for(int i=0; i < dim; i++)
        {
          dPrim.set_momentum(L, i, dE.momentum(L, i)*T + T*S.velocity(L, i)*dE.energy(L));
        }
      dPrim.set_energy(L, T*T*dE.energy(L));
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                     const InStateView &Se, OutStateView &Sc,
                                     const LTETables &lteTables) const
    {
      std::cerr<< " CL ALERT : Not functional in LTE yet "<<std::endl;
    }

    template<typename InStateView, typename OutStateView>
    inline void primitive_to_conserved(const PhysicsConstants &phys, const StateLayout &L,
                                       const InStateView &prim, OutStateView &cons,
                                       const LTETables &lteTables) const
    {
      const mfem::real_t rho = prim.mass(L);
      const int dim    = L.dim;
      mfem::real_t v2        = 0.0;

      cons.set_mass(L, rho);
      for(int d = 0; d < dim; d++)
        {
          cons.set_momentum(L, d, rho*prim.velocity(L, d));
          v2 += prim.velocity(L,d)*prim.velocity(L,d);
        }
      mfem::real_t rhoe = internal_energy_from_pressure(phys, L, cons, prim.pressure(L), lteTables);
      cons.set_energy(L, rhoe + 0.5 * rho * v2);
    }

    // TODO: Consider whether this is needed/convenient
    // It *can be* nice to have here, but kind of out-of-place
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void velocity(const PhysicsConstants &phys, const StateLayout &L,
                         const StateView &S, mfem::real_t u[3], const LTETables &lteTables) const
    {
      const int dim = L.dim;
      for (int d = 0; d < dim; ++d)
        {
          u[d] = S.velocity(L, d);
        }
      for (int d = dim; d < 3; ++d)
        {
          u[d] = mfem::real_t(0);
        }
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t property_lookup(int property_idx, const PhysicsConstants &phys,
                                        const StateLayout &L, const StateView &S,
                                        const LTETables &lteTables) const
    {
      mfem::real_t T = temp_from_internal_energy(phys, L, S, lteTables);
      return biinterp_lte_table(property_idx, phys, L, S, T, lteTables);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temp_from_internal_energy(const PhysicsConstants &phys, const StateLayout &L,
                                                  const StateView &S, const LTETables &lteTables) const
    {
      const auto e = specific_internal_energy(phys, L, S, lteTables);
      detail::LTETemperatureRecovery recovery(biinterp_inverse_table(phys, L, S, lteTables));
      while (recovery.NeedsIteration())
        {
          const auto e_guess = internal_energy_from_helmholtz(phys, L, S, lteTables, &recovery.temperature);
          const auto cv = cv_from_helmholtz(phys, L, S, lteTables, &recovery.temperature);
          recovery.Update(e, e_guess, cv);
          if (recovery.ExceededIterations())
            {
#ifdef __CUDA_ARCH__
              printf("Newton method did not converge in temp_from_internal_energy");
              asm("trap;");
#else
              MFEM_ABORT("Newton method did not converge in temp_from_internal_energy");
#endif
            }
        }

      return recovery.temperature;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t biinterp_inverse_table(const PhysicsConstants &phys, const StateLayout &L,
                                               const StateView &S, const LTETables &lteTables) const
    {
      /*
       * Q01--------Q11
       *  |          |
       *  |          |
       *  |          |
       * Q00--------Q10
       */

      // Point rho and e values
      mfem::real_t rho  = density(phys, L, S, lteTables);
      mfem::real_t e = specific_internal_energy(phys, L, S, lteTables);

      // Get the lower and upper x and y indices of the cell
      int l_x = hunt(lteTables.tables.rho_grid, lteTables.L.nx, rho, 0);
      int l_y = hunt(lteTables.tables.e_grid, lteTables.L.ny, e, 0);
      int u_x = l_x + 1  , u_y = l_y + 1;

      if(l_x < 0 || u_x >= lteTables.L.nx || l_y < 0 || u_y >= lteTables.L.ny)
        {
#ifdef __CUDA_ARCH__
          printf(" CL ALERT : Out of bounds in LTE table lookup! \n");
          printf("l_x : %d l_y : %d \n", l_x, l_y);
          printf("rho : %e < %e < %e \n", lteTables.tables.rho_grid[0],
                 rho, lteTables.tables.rho_grid[lteTables.L.nx-1]);
          printf("e : %e < %e < %e \n", lteTables.tables.e_grid[0],
                 e, lteTables.tables.e_grid[lteTables.L.ny-1]);
          asm("trap;");
#else
          std::cout << " CL ALERT : Out of bounds in LTE table lookup! "<<std::endl;
          std::cout << "l_x : " << l_x << "l_y : " << l_y << std::endl;
          std::cout << "rho : " << lteTables.tables.rho_grid[0] << " < " << rho
                    << " < " << lteTables.tables.rho_grid[lteTables.L.nx-1] << std::endl;
          std::cout << "e : " << lteTables.tables.e_grid[0] << " < "
                    << e << " < " << lteTables.tables.e_grid[lteTables.L.ny-1] << std::endl;
          std::exit(1);
#endif
        }

      // Get the lower and upper x and y coordinates of the cell
      mfem::real_t rho_l  = lteTables.tables.rho_grid[l_x] , rho_u = lteTables.tables.rho_grid[u_x];
      mfem::real_t e_l    = lteTables.tables.e_grid[l_y]   , e_u   = lteTables.tables.e_grid[u_y];

      // Get the corner property values
      mfem::real_t Q00 = lteTables.tables.inv_table[lteTables.L.property_index(0, l_x, l_y)];
      mfem::real_t Q01 = lteTables.tables.inv_table[lteTables.L.property_index(0, l_x, u_y)];
      mfem::real_t Q10 = lteTables.tables.inv_table[lteTables.L.property_index(0, u_x, l_y)];
      mfem::real_t Q11 = lteTables.tables.inv_table[lteTables.L.property_index(0, u_x, u_y)];


      mfem::real_t wx = (rho  - rho_l)  / (rho_u - rho_l);
      mfem::real_t wy = (e - e_l) / (e_u - e_l);


      // Clamp to [0, 1]
      wx = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wx));
      wy = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wy));


      return Q00 * ((1 - wx) * (1 - wy)) + Q01 * ((1 - wx) * wy) +
        Q10 * (wx * (1 - wy)) + Q11 * (wx * wy);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t biinterp_lte_table(int property_idx, const PhysicsConstants &phys,
                                           const StateLayout &L, const StateView &S,
                                           const mfem::real_t T,
                                           const LTETables &lteTables) const
    {
      /*
       * Q01--------Q11
       *  |          |
       *  |          |
       *  |          |
       * Q00--------Q10
       */
      mfem::real_t rho  = density(phys, L, S, lteTables);

      // Get the lower and upper x and y indices of the cell
      int l_x = hunt(lteTables.tables.rho_grid, lteTables.L.nx, rho, 0);
      int l_y = hunt(lteTables.tables.T_grid, lteTables.L.ny, T, 0);
      int u_x = l_x + 1  , u_y = l_y + 1;

      // Get the lower and upper x and y coordinates of the cell
      mfem::real_t rho_l  = lteTables.tables.rho_grid[l_x] , rho_u = lteTables.tables.rho_grid[u_x];
      mfem::real_t T_l    = lteTables.tables.T_grid[l_y]   , T_u   = lteTables.tables.T_grid[u_y];

      // Get the corner property values
      mfem::real_t Q00 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, l_x, l_y)];
      mfem::real_t Q01 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, l_x, u_y)];
      mfem::real_t Q10 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, u_x, l_y)];
      mfem::real_t Q11 = lteTables.tables.lte_table[lteTables.L.property_index(property_idx, u_x, u_y)];

      mfem::real_t wx, wy;
      const mfem::real_t one_over_dx = 1.0 / (rho_u - rho_l);
      const mfem::real_t one_over_dy = 1.0 / (T_u - T_l);

      wx = (rho - rho_l) * one_over_dx;
      wy = (T - T_l)     * one_over_dy;

      // Clamp to [0, 1]
      wx = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wx));
      wy = Theseus::Kernels::rmax(mfem::real_t(0), Theseus::Kernels::rmin(mfem::real_t(1), wy));


      return Q00 * ((1 - wx) * (1 - wy)) + Q01 * ((1 - wx) * wy) +
        Q10 * (wx * (1 - wy)) + Q11 * (wx * wy);
    }

    template<typename StateView, typename Evaluate>
    MFEM_HOST_DEVICE
    inline void symmetrized_Itoh_Abe_discrete_gradient(Evaluate thermo,
                                                       const PhysicsConstants &phys, const StateLayout &L,
                                                       const StateView &S1, const StateView &S2,
                                                       mfem::real_t &grad_fbeta_rho, mfem::real_t &grad_fbeta_beta,
                                                       const LTETables &lteTables) const
    {
      mfem::real_t rho1 = density(phys, L, S1, lteTables);
      mfem::real_t rho2 = density(phys, L, S2, lteTables);
      mfem::real_t T1   = temperature(phys, L, S1, lteTables);
      mfem::real_t T2   = temperature(phys, L, S2, lteTables);
      mfem::real_t inv_T1 = 1.0 / T1;
      mfem::real_t inv_T2 = 1.0 / T2;

      mfem::real_t f1_der[2], f2_der[2];
      mfem::real_t f11 = thermo(S1, &T1, f1_der) * inv_T1;
      mfem::real_t f12 = thermo(S1, &T2) * inv_T2;
      mfem::real_t f21 = thermo(S2, &T1) * inv_T1;
      mfem::real_t f22 = thermo(S2, &T2, f2_der) * inv_T2;

      mfem::real_t drho = rho2 - rho1;
      mfem::real_t dbeta   = inv_T2 - inv_T1;

      mfem::real_t tol_rho = 2.22e-15 + 1.49e-8*Kernels::rmax(rho1, rho2);
      mfem::real_t tol_beta = 2.22e-15 + 1.49e-8*Kernels::rmax(inv_T1, inv_T2);

      bool use_diff_x = Kernels::rabs(drho) >= tol_rho;
      bool use_diff_y = Kernels::rabs(dbeta) >= tol_beta;

      mfem::real_t mx = static_cast<mfem::real_t>(use_diff_x);
      mfem::real_t my = static_cast<mfem::real_t>(use_diff_y);

      mfem::real_t drho_safe = drho * mx + (1.0 - mx);
      mfem::real_t dbeta_safe = dbeta * my + (1.0 - my);

      mfem::real_t num_rho = 0.5*((f21 - f11) + (f22 - f12));
      mfem::real_t num_beta = 0.5*((f12 - f11) + (f22 - f21));

      mfem::real_t out_rho = num_rho / drho_safe;
      mfem::real_t out_beta = num_beta / dbeta_safe;

      // If the difference is too small, use the derivative instead
      mfem::real_t out_rho_grad = 0.5 * (f1_der[0] * inv_T1 + f2_der[0] * inv_T2);
      mfem::real_t out_beta_grad = 0.5*( (f11*T1 - T1 * f1_der[1]) + (f22*T2 - T2 * f2_der[1]) );

      grad_fbeta_rho = out_rho * mx + out_rho_grad * (1.0 - mx);
      grad_fbeta_beta = out_beta * my + out_beta_grad * (1.0 - my);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void rho_ie_avg_discrete_gradient(const PhysicsConstants &phys, const StateLayout &L,
                                             const StateView &S1, const StateView &S2,
                                             mfem::real_t &rho_avg, mfem::real_t &e_avg,
                                             const LTETables &lteTables) const
    {
      auto eval_gibbs = [=] MFEM_HOST_DEVICE (const StateView &S, mfem::real_t *T,
                                              mfem::real_t *gibbs_der = nullptr) {
          return gibbs(phys, L, S, lteTables, T, gibbs_der);
      };

      auto eval_pressure = [=] MFEM_HOST_DEVICE (const StateView &S, mfem::real_t *T,
                                                 mfem::real_t *pres_der = nullptr) {
          return pressure(phys, L, S, lteTables, T, pres_der);
      };
      mfem::real_t grad_Pbeta_rho, grad_Pbeta_beta;
      mfem::real_t grad_gbeta_rho, grad_gbeta_beta;
      symmetrized_Itoh_Abe_discrete_gradient(eval_pressure, phys, L, S1, S2, grad_Pbeta_rho, grad_Pbeta_beta, lteTables);
      symmetrized_Itoh_Abe_discrete_gradient(eval_gibbs, phys, L, S1, S2, grad_gbeta_rho, grad_gbeta_beta, lteTables);
      rho_avg = grad_Pbeta_rho / grad_gbeta_rho;
      e_avg = grad_gbeta_beta - grad_Pbeta_beta / rho_avg;
    }

    MFEM_HOST_DEVICE
    inline mfem::real_t Hermite_Basis(int basis, int derivative, mfem::real_t t) const
    {
      const mfem::real_t t2 = t * t;
      const mfem::real_t t3 = t2 * t;
      
      if(derivative == 0)
      {
        switch(basis)
        {
          case 0: return 2*t3 - 3*t2 + 1;
          case 1: return -2*t3 + 3*t2;
          case 2: return t3 - 2*t2 + t;
          case 3: return t3 - t2;
        }
      }
      else if(derivative == 1)
      {
        switch(basis)
        {
          case 0: return 6*t2 - 6*t;
          case 1: return -6*t2 + 6*t;
          case 2: return 3*t2 - 4*t + 1;
          case 3: return 3*t2 - 2*t;
        }
      }
      else if(derivative == 2)
      {
        switch(basis)
        {
          case 0: return 12*t - 6;
          case 1: return -12*t + 6;
          case 2: return 6*t - 4;
          case 3: return 6*t - 2;
        }
      }
      else
      {
        printf("Error: Invalid derivative order %d for Hermite basis function.\n", derivative);
#ifdef __CUDA_ARCH__
        asm("trap;");
#else
        MFEM_ABORT("Error: Invalid derivative order for Hermite basis function.");
#endif
      }
      return 0.0;
    }

    MFEM_HOST_DEVICE
    inline mfem::real_t Evaluate_Hermite_Polynomial(const mfem::real_t *coeffs, mfem::real_t xi, mfem::real_t eta,
                                            mfem::real_t drho, mfem::real_t dT,
                                            int derivative_xi, int derivative_eta) const
    {
      mfem::real_t result = 0.0;
      for(int i = 0; i < 4; i++)
      {
        for(int j = 0; j < 4; j++)
        {
          mfem::real_t basis_xi = Hermite_Basis(i, derivative_xi, xi);
          mfem::real_t basis_eta = Hermite_Basis(j, derivative_eta, eta);
          result += coeffs[i * 4 + j] * basis_xi * basis_eta;
        }
      }
      mfem::real_t fac = 1.0;
      while(derivative_xi > 0)
      {
        fac *= drho;
        derivative_xi--;
      }
      while(derivative_eta > 0)
      {
        fac *= dT;
        derivative_eta--;
      }
      return result / fac;
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void GetHermiteCoeffs(const PhysicsConstants &phys, const StateLayout &L,
                          const StateView &S, const LTETables &lteTables, 
                          const mfem::real_t &T, mfem::real_t &xi, mfem::real_t &eta,
                          mfem::real_t &drho, mfem::real_t &dT, mfem::real_t *coeffs) const
    {
    mfem::real_t rho = density(phys, L, S, lteTables);

    // Get the lower and upper x and y indices of the cell
    int l_x = hunt(lteTables.tables.rho_grid, lteTables.L.nx, rho, 0);
    int l_y = hunt(lteTables.tables.T_grid, lteTables.L.ny, T, 0);
    int u_x = l_x + 1  , u_y = l_y + 1;

    mfem::real_t rho_l  = lteTables.tables.rho_grid[l_x] , rho_u = lteTables.tables.rho_grid[u_x];
    mfem::real_t T_l    = lteTables.tables.T_grid[l_y]   , T_u   = lteTables.tables.T_grid[u_y];

    xi = (rho - rho_l) / (rho_u - rho_l);
    eta = (T - T_l) / (T_u - T_l);
    drho = rho_u - rho_l;
    dT = T_u - T_l;

    // Get the coefficients for the Hermite polynomial
    for(int i = 0; i < 2; i++)
      {
        for(int j = 0; j < 2; j++)
        {
          coeffs[i*4 + j] = lteTables.tables.lte_table[lteTables.L.property_index(lteTables.L.H_idx, l_x + i, l_y + j)];
          coeffs[i*4 + j + 2] = lteTables.tables.lte_table[lteTables.L.property_index(lteTables.L.H_T_idx, l_x + i, l_y + j)] * dT;
          coeffs[(i + 2)*4 + j] = lteTables.tables.lte_table[lteTables.L.property_index(lteTables.L.H_r_idx, l_x + i, l_y + j)] * drho;
          coeffs[(i + 2)*4 + j + 2] = lteTables.tables.lte_table[lteTables.L.property_index(lteTables.L.H_r_T_idx, l_x + i, l_y + j)] * dT * drho;
        }
      }
    }
  };
}
