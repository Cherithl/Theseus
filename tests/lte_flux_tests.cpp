// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#include "unit_test.hpp"
#include "test_helpers.hpp"
#include "plato_helpers.hpp"
#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTETable.hpp"
#include "LTEEOS.hpp"
#include "LTEGasModel.hpp"
#include "LTEECFlux.hpp"
#include "PirozzoliFlux.hpp"
#include "TheseusConfig.hpp"
#include "json.hpp"

using real_t = mfem::real_t;

using namespace Theseus;
using namespace Theseus::LTETable;

int num_properties = 11;
int nx = 1001;
int ny = 1001;
real_t rho_min  = 1e-4  , rho_max  = 1.1 ;
real_t T_min = 250.0, T_max = 10000.0;
mfem::Vector lte_table( (num_properties) * (nx*ny) ), inv_table( nx*ny );
mfem::Vector rho_grid, T_grid, e_grid;

const int dim = 3, ndofs = 1;
StateLayout L(dim, ndofs);
Theseus::LTETable::LTETables lteT(nx, ny);
std::shared_ptr<PhysicsConstants> phys = std::make_shared<PhysicsConstants>(1.4, 0.72, 287.05, 0.02);
LTEGasEOS eos;

TEST(plato_construct_table_test)
{
    // CL NOTE : To make sure the database path is correctly set and avoiding the initialization overhead in the subsequent tests.
    std::string solver  = "LTE_table_rhoT_(air5)";
    std::string mixture = "air5";
    std::string path(Theseus::BuildConfig::PlatoDBPath);
    if(Theseus::LTETable::check_plato_database_path(path)){
      std::cerr << "Failed to find PLATO database: " << path << std::endl;
      return 1;
    }
    //"/home/cherith2/Workspace/SOURCE_CODES/database";
    std::string empty_str = "empty";
    std::cout << "Calling PLATO Initialize with: " << std::endl
	      << "Solver: " << solver << std::endl
	      << "Mixture: " << mixture << std::endl
	      << "Path: " << path << std::endl;

    plato_initialize(solver.c_str(), mixture.c_str(), empty_str.c_str(), empty_str.c_str(), path.c_str());
    log_grid(nx, rho_min, rho_max, rho_grid);
    uniform_grid(ny, T_min, T_max, T_grid);
    real_t e_min, e_max;
    fill_table(lteT.L, rho_grid.GetData(), T_grid.GetData(),
         lte_table.GetData(), e_min, e_max);
    uniform_grid(ny, e_min, e_max, e_grid);
    fill_inv_table(lteT.L, rho_grid.GetData(), e_grid.GetData(), T_grid.GetData(), inv_table.GetData());

    lteT.tables = {
      lte_table.HostRead(), inv_table.HostRead(),
      rho_grid.HostRead(), T_grid.HostRead(),
      e_grid.HostRead()
    };
    return 0;
}

static void check_symmetric_itoh_abe_gradient(const real_t rho1, const real_t T1,
                                  const real_t rho2, const real_t T2)
{
   PlatoMixture mix1, mix2;

   const int num_eq = L.eq_energy + 1;

   const real_t u1[3] = {10.0, -3.0, 5.0};
   const real_t u2[3] = {-10.0, 3.0, -5.0};

   mfem::Vector U1(num_eq * ndofs), U2(num_eq * ndofs);

   plato_set_state(rho1, T1, mix1);
   plato_set_state(rho2, T2, mix2);

   fill_single_dof_state(L, U1, dim, rho1, u1, rho1 * mix1.e);
   fill_single_dof_state(L, U2, dim, rho2, u2, rho2 * mix2.e);

   PointStateView S1(U1.HostRead()), S2(U2.HostRead());

   const real_t T1_eos = eos.temperature(*phys, L, S1, lteT);
   const real_t T2_eos = eos.temperature(*phys, L, S2, lteT);

   const real_t g1 = eos.gibbs(*phys, L, S1, lteT, &T1_eos);
   const real_t g2 = eos.gibbs(*phys, L, S2, lteT, &T2_eos);

   const real_t P1 = eos.pressure(*phys, L, S1, lteT, &T1_eos);
   const real_t P2 = eos.pressure(*phys, L, S2, lteT, &T2_eos);

   real_t grad_gbeta_rho, grad_gbeta_beta;
   real_t grad_Pbeta_rho, grad_Pbeta_beta;

   auto eval_pressure = [&](const auto &S, mfem::real_t *T, mfem::real_t *pres_der = nullptr) {
     return eos.pressure(*phys, L, S, lteT, T, pres_der);
   };

   auto eval_gibbs = [&](const auto &S, mfem::real_t *T, mfem::real_t *gibbs_der = nullptr) {
     return eos.gibbs(*phys, L, S, lteT, T, gibbs_der);
   };

   eos.symmetrized_Itoh_Abe_discrete_gradient(eval_gibbs,
      *phys, L, S1, S2, grad_gbeta_rho, grad_gbeta_beta, lteT);

   eos.symmetrized_Itoh_Abe_discrete_gradient(eval_pressure,
      *phys, L, S1, S2, grad_Pbeta_rho, grad_Pbeta_beta, lteT);

   const real_t beta1 = 1.0 / T1_eos;
   const real_t beta2 = 1.0 / T2_eos;

   const real_t gbeta_jump = g2 * beta2 - g1 * beta1;
   const real_t Pbeta_jump = P2 * beta2 - P1 * beta1;

   const real_t gbeta_jump_approx = grad_gbeta_rho * (rho2 - rho1) + grad_gbeta_beta * (beta2 - beta1);
   const real_t Pbeta_jump_approx = grad_Pbeta_rho * (rho2 - rho1) + grad_Pbeta_beta * (beta2 - beta1);

   const real_t abs_tol = 5e-10;
   const real_t rel_tol = 1e-13;

   const real_t tol_g = abs_tol + rel_tol * std::max(std::abs(gbeta_jump), std::abs(gbeta_jump_approx));
   const real_t tol_P = abs_tol + rel_tol * std::max(std::abs(Pbeta_jump), std::abs(Pbeta_jump_approx));

   EXPECT_SMALL(gbeta_jump - gbeta_jump_approx, tol_g);
   EXPECT_SMALL(Pbeta_jump - Pbeta_jump_approx, tol_P);

  //  std::cout << "\n";
  //  std::cout << "del_rho = " << rho2 - rho1 << ", del_T = " << T2 - T1 << std::endl;

  //  std::cout << "\terr(gbeta_jump(table) - gbeta_jump_dg) = " << (gbeta_jump - gbeta_jump_approx) << "\n";
  //  std::cout << "\terr(Pbeta_jump(table) - Pbeta_jump_dg) = " << (Pbeta_jump - Pbeta_jump_approx) << "\n";
  //  std::cout << "\trel(gbeta_jump(table) - gbeta_jump_dg) = " << (gbeta_jump - gbeta_jump_approx) / gbeta_jump << "\n";
  //  std::cout << "\trel(Pbeta_jump(table) - Pbeta_jump_dg) = " << (Pbeta_jump - Pbeta_jump_approx) / Pbeta_jump << "\n";

}

template <typename FluxT>
static void run_vol_flux_tadmor_condition(const real_t rho1, const real_t T1,
                                          const real_t rho2, const real_t T2,
                                          const FluxT &num_flux)
{
  mfem::Vector met1(3), met2(3), flux(5);
  mfem::Vector wL(5), wR(5);
  Theseus::LTEGas gas(*phys, L, lteT);
  PlatoMixture mix1, mix2;
  const int num_eq = L.eq_energy + 1;

  const real_t u1[3] = {12.0, 3.0, 1.0};
  const real_t u2[3] = {23.0, 2.0, 2.0};

  mfem::Vector qL(num_eq * ndofs), qR(num_eq * ndofs);

  plato_set_state(rho1, T1, mix1);
  plato_set_state(rho2, T2, mix2);

  fill_single_dof_state(L, qL, dim, rho1, u1, rho1 * mix1.e);
  fill_single_dof_state(L, qR, dim, rho2, u2, rho2 * mix2.e);
  Theseus::PointStateView SL(qL.HostRead()), SR(qR.HostRead());
  Theseus::PointStateViewRW WL(wL.HostWrite()), WR(wR.HostWrite());
  gas.entropy_state(SL, WL);
  gas.entropy_state(SR, WR);

  // Take back the Temperarue from the States
  real_t T1_eos = eos.temperature(*phys, L, SL, lteT);
  real_t T2_eos = eos.temperature(*phys, L, SR, lteT);

  // std::cout << "\n";
  // std::cout << "del_rho = " << rho2 - rho1 << ", del_T = " << T2 - T1 << std::endl;
  for(int d=0; d < 3; d++)
  {
    met1 = 0.0;
    met2 = 0.0;
    met1(d) = 1.0;
    met2(d) = 1.0;
    num_flux.ComputeVolumeFlux(gas, qL.HostRead(), qR.HostRead(), met1.HostRead(), met2.HostRead(), flux.HostWrite());

    mfem::real_t psi_L = gas.pressure(SL) * gas.velocity(SL, d) / T1_eos;
    mfem::real_t psi_R = gas.pressure(SR) * gas.velocity(SR, d) / T2_eos;
    mfem::real_t RHS = psi_R - psi_L;
    mfem::real_t LHS = 0.0;
    for(int eq=0; eq < num_eq; eq++)
    {
      LHS += flux(eq) * (wR(eq) - wL(eq));
    }
    // std::cout << "d = " << d << " LHS - RHS = " << LHS - RHS << ", rel(LHS - RHS) = " << (LHS - RHS)/RHS << std::endl;
    EXPECT_CLOSE(LHS, RHS, 1e-9);
  }
}

static void compute_physical_normal_flux(const mfem::Vector &q,
                                            const mfem::Vector &nor,
                                            LTEGas &gas,
                                            mfem::Vector &flux)
{
    flux.SetSize(5);

    if(q.Size() != 5 || nor.Size() != 3)
    {
      std::cerr << "Error: q must be of size 5 and nor must be of size 3." << std::endl;
      return;
    }
    const real_t rho = q(0);
    const real_t u   = q(1) / rho;
    const real_t v   = q(2) / rho;
    const real_t w   = q(3) / rho;
    const real_t E   = q(4);

    Theseus::PointStateView S(q.HostRead());
    const real_t p = gas.pressure(S);

    const real_t nx = nor(0);
    const real_t ny = nor(1);
    const real_t nz = nor(2);
    const real_t un = u*nx + v*ny + w*nz;

    flux(0) = rho * un;
    flux(1) = rho * u * un + p * nx;
    flux(2) = rho * v * un + p * ny;
    flux(3) = rho * w * un + p * nz;
    flux(4) = (E + p) * un;
}

template <typename FluxT>
static void run_face_flux_consistency(const FluxT &num_flux)
{
  mfem::Vector nor(3), flux(5), expected_flux(5);
  PlatoMixture mix;
  Theseus::LTEGas gas(*phys, L, lteT);
  const int num_eq = L.eq_energy + 1;

  for(real_t rho = 1e-3; rho < 1.0; rho *= 1.5)
  {
    for(real_t T = 300.0; T < 10000.0; T += 500.0)
    {
      const real_t u[3] = {10.0, -3.0, 5.0};
      mfem::Vector q(num_eq * ndofs);
      plato_set_state(rho, T, mix);
      fill_single_dof_state(L, q, dim, rho, u, rho * mix.e);
      Theseus::PointStateView S(q.HostRead());

      nor(0) = 0.6;
      nor(1) = 0.8;
      nor(2) = 0.0;
      num_flux.ComputeFaceFlux(gas, q.HostRead(), q.HostRead(), nor.HostRead(), flux.HostWrite());
      compute_physical_normal_flux(q, nor, gas, expected_flux);

      for(int eq=0; eq < num_eq; eq++)
      {
        // mfem::out.precision(17);
        // mfem::out << "eq = " << eq << " err = " << flux(eq) - expected_flux(eq) << " rel = " << (flux(eq) - expected_flux(eq))/expected_flux(eq) << "\n";
        EXPECT_CLOSE(flux(eq)/expected_flux(eq), 1.0, 1e-9);
      }
    }
  }
  // const real_t rho = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
  // const real_t T   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];
}

TEST(gibbs)
{
  PlatoMixture mix;

  real_t rho = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
  real_t T0   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];
  plato_set_state(rho, T0, mix);

  const int num_eq = L.eq_energy + 1;
  const real_t u1[3] = {10.0, -3.0, 5.0};
  mfem::Vector U(num_eq * ndofs);
  fill_single_dof_state(L, U, dim, rho, u1, rho*mix.e);
  PointStateView S(U.HostRead());

  real_t g_plato = mix.g;
  std::vector<real_t> g_der(2), p_der(2);
  real_t T_new = eos.temperature(*phys, L, S, lteT);
  real_t g_eos = eos.gibbs(*phys, L, S, lteT, &T_new, g_der.data());
  real_t p_eos = eos.pressure(*phys, L, S, lteT, &T_new, p_der.data());

  real_t dg_dT_plato = mix.dg_dT;
  real_t dg_drho_plato = mix.dg_drho;
  real_t dp_dT_plato = mix.dp_dT;
  real_t dp_drho_plato = mix.dp_drho;

  std::cout << "\n";
  std::cout << "nx = " << nx << ", ny = " << ny << std::endl;
  std::cout << "g_rel   = " << (g_plato - g_eos)/g_plato << std::endl;
  std::cout << "dg_dT_rel = " << (dg_dT_plato - g_der[1])/dg_dT_plato << std::endl;
  std::cout << "dg_drho_rel = " << (dg_drho_plato - g_der[0])/dg_drho_plato << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "p_rel   = " << (mix.P - p_eos)/mix.P << std::endl;
  std::cout << "dp_dT_rel = " << (dp_dT_plato - p_der[1])/dp_dT_plato << std::endl;
  std::cout << "dp_drho_rel = " << (dp_drho_plato - p_der[0])/dp_drho_plato << std::endl;

  return 0;
}

TEST(symmetric_itoh_abe_gradient_test)
{
   // rho1 - rho2 > tol and T1 - T2 > tol
   real_t rho1 = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
   real_t T1   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];

   real_t rho2 = 0.766475*rho_grid[nx/4] + 0.233525*rho_grid[5*nx/6];
   real_t T2   = 0.231744*T_grid[ny/12] + 0.768256*T_grid[4*ny/5];

  for(real_t rho1 = 1e-2; rho1 < 0.5; rho1 *= 1.1)
  {
    for(real_t T1 = 300.0; T1 < 4000.0; T1 += 500.0)
    {
      rho2 = 2*rho1; T2 = 1.5*T1;
      check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

      rho2 = rho1 + 1e-8; T2 = 1.5*T1;
      check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

      rho2 = 2*rho1; T2 = T1 + 1e-8;
      check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

      rho2 = rho1 + 1e-8; T2 = T1 + 1e-8;
      check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);
    }
  }
  return 0;
}

TEST(ECFlux_Tadmor)
{
  const int dim=2;
  const int ndofs = 1;
  Theseus::LTEECFlux::InviscidFlux lteec_flux;
  Theseus::PirozzoliFlux::InviscidFlux piro_flux;

  for(real_t rho1 = 1e-2; rho1 < 0.5; rho1 *= 1.1)
  {
    for(real_t T1 = 300.0; T1 < 4000.0; T1 += 500.0)
    {
      real_t rho2 = 2*rho1; real_t T2 = 1.5*T1;
      run_vol_flux_tadmor_condition(rho1, T1, rho2, T2, lteec_flux);

      rho2 = rho1 + 1e-15;
      run_vol_flux_tadmor_condition(rho1, T1, rho2, T2, lteec_flux);

      rho2 = 2*rho1; T2 = T1 + 1e-15;
      run_vol_flux_tadmor_condition(rho1, T1, rho2, T2, lteec_flux);

      rho2 = rho1 + 1e-15; T2 = T1 + 1e-15;
      run_vol_flux_tadmor_condition(rho1, T1, rho2, T2, lteec_flux);
    }
  }
  return 0;
}

TEST(flux_consistency_test)
{
  Theseus::LTEECFlux::InviscidFlux lteec_flux;
  run_face_flux_consistency(lteec_flux);

  Theseus::PirozzoliFlux::InviscidFlux piro_flux;
  run_face_flux_consistency(piro_flux);

  return 0;
}

// TEST(flux_symmetry_consistency_test)
// {

// }