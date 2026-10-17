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
    log_grid(ny, T_min, T_max, T_grid);
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

TEST(gibbs)
{
  PlatoMixture mix;

  real_t rho = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
  real_t T0   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];
  plato_set_state(rho, T0, mix);

  const int num_eq = L.eq_energy + 1;
  const real_t u1[3] = {10.0, -3.0, 5.0};
  std::vector<real_t> U(num_eq * ndofs);
  fill_single_dof_state(L, U, dim, rho, u1, rho*mix.e);
  PointStateView S(U.data());

  real_t g_plato = mix.g;
  real_t dg_rrho_eos, dg_dT_eos, dp_drho_eos, dp_dT_eos;
  real_t T_new = eos.temperature(*phys, L, S, lteT);
  real_t g_eos = eos.biinterp_lte_table(lteT.L.g_idx, *phys, L, S, T_new, lteT, &dg_rrho_eos, &dg_dT_eos);
  real_t p_eos = eos.biinterp_lte_table(lteT.L.P_idx, *phys, L, S, T_new, lteT, &dp_drho_eos, &dp_dT_eos);

  real_t dg_dT_plato = mix.dg_dT;
  real_t dg_drho_plato = mix.dg_drho;
  real_t dp_dT_plato = mix.dp_dT;
  real_t dp_drho_plato = mix.dp_drho;

  std::cout << "\n";
  std::cout << "nx = " << nx << ", ny = " << ny << std::endl;
  std::cout << "g_rel   = " << (g_plato - g_eos)/g_plato << std::endl;
  std::cout << "dg_dT_rel = " << (dg_dT_plato - dg_dT_eos)/dg_dT_plato << std::endl;
  std::cout << "dg_drho_rel = " << (dg_drho_plato - dg_rrho_eos)/dg_drho_plato << std::endl;
  std::cout << "===============================" << std::endl;
  std::cout << "p_rel   = " << (mix.P - p_eos)/mix.P << std::endl;
  std::cout << "dp_dT_rel = " << (dp_dT_plato - dp_dT_eos)/dp_dT_plato << std::endl;
  std::cout << "dp_drho_rel = " << (dp_drho_plato - dp_drho_eos)/dp_drho_plato << std::endl;

  return 0;
}

static void
check_symmetric_itoh_abe_gradient(const real_t rho1, const real_t T1,
                                  const real_t rho2, const real_t T2)
{
   PlatoMixture mix1, mix2;

   const int num_eq = L.eq_energy + 1;

   const real_t u1[3] = {10.0, -3.0, 5.0};
   const real_t u2[3] = {-10.0, 3.0, -5.0};

   std::vector<real_t> U1(num_eq * ndofs), U2(num_eq * ndofs);

   plato_set_state(rho1, T1, mix1);
   plato_set_state(rho2, T2, mix2);

   fill_single_dof_state(L, U1, dim, rho1, u1, rho1 * mix1.e);
   fill_single_dof_state(L, U2, dim, rho2, u2, rho2 * mix2.e);

   PointStateView S1(U1.data()), S2(U2.data());

   const real_t T1_eos = eos.temperature(*phys, L, S1, lteT);
   const real_t T2_eos = eos.temperature(*phys, L, S2, lteT);

   const real_t g1 = eos.biinterp_lte_table(lteT.L.g_idx, *phys, L, S1, T1_eos, lteT);
   const real_t g2 = eos.biinterp_lte_table(lteT.L.g_idx, *phys, L, S2, T2_eos, lteT);

   const real_t P1 = eos.biinterp_lte_table(lteT.L.P_idx, *phys, L, S1, T1_eos, lteT);
   const real_t P2 = eos.biinterp_lte_table(lteT.L.P_idx, *phys, L, S2, T2_eos, lteT);

   real_t grad_gbeta_rho, grad_gbeta_beta;
   real_t grad_Pbeta_rho, grad_Pbeta_beta;

   eos.symmetrized_Itoh_Abe_discrete_gradient(lteT.L.g_idx,
      *phys, L, S1, S2, grad_gbeta_rho, grad_gbeta_beta, lteT);

   eos.symmetrized_Itoh_Abe_discrete_gradient(lteT.L.P_idx,
      *phys, L, S1, S2, grad_Pbeta_rho, grad_Pbeta_beta, lteT);

   const real_t beta1 = 1.0 / T1_eos;
   const real_t beta2 = 1.0 / T2_eos;

   const real_t gbeta_exact = mix2.g * (1/T2) - mix1.g * (1/T1);
   const real_t Pbeta_exact = mix2.P * (1/T2) - mix1.P * (1/T1);

   const real_t gbeta_jump = g2 * beta2 - g1 * beta1;
   const real_t Pbeta_jump = P2 * beta2 - P1 * beta1;

   const real_t gbeta_jump_approx = grad_gbeta_rho * (rho2 - rho1) + grad_gbeta_beta * (beta2 - beta1);
   const real_t Pbeta_jump_approx = grad_Pbeta_rho * (rho2 - rho1) + grad_Pbeta_beta * (beta2 - beta1);

   std::cout << "\n";
   std::cout << "del_rho = " << rho2 - rho1 << ", del_T = " << T2 - T1 << std::endl;

  //  std::cout << "[[g/T]]        = " << gbeta_jump << "\n";
  //  std::cout << "[[g/T]]_approx = " << gbeta_jump_approx << "\n";

  //  std::cout << "[[P/T]]        = " << Pbeta_jump << "\n";
  //  std::cout << "[[P/T]]_approx = " << Pbeta_jump_approx << "\n";

   std::cout << "\trel(gbeta_jump(table) - gbeta_jump_dg) = " << (gbeta_jump - gbeta_jump_approx) / gbeta_jump << "\n";
   std::cout << "\trel(Pbeta_jump(table) - Pbeta_jump_dg) = " << (Pbeta_jump - Pbeta_jump_approx) / Pbeta_jump << "\n";

   std::cout << "\trel(gbeta_jump(table) - gbeta_jump(exact)) = " << (gbeta_jump - gbeta_exact) / gbeta_jump << "\n";
   std::cout << "\trel(Pbeta_jump(table) - Pbeta_jump(exact)) = " << (Pbeta_jump - Pbeta_exact) / Pbeta_jump << "\n";

   std::cout << "\trel(gbeta_jump(exact) - gbeta_jump_dg) = " << (gbeta_exact - gbeta_jump_approx) / gbeta_exact << "\n";
   std::cout << "\trel(Pbeta_jump(exact) - Pbeta_jump_dg) = " << (Pbeta_exact - Pbeta_jump_approx) / Pbeta_exact << "\n";
}

TEST(symmetric_itoh_abe_gradient_test)
{
   // rho1 - rho2 > tol and T1 - T2 > tol
   real_t rho1 = 0.233525*rho_grid[nx/4] + 0.766475*rho_grid[5*nx/6];
   real_t T1   = 0.768256*T_grid[ny/12] + 0.231744*T_grid[4*ny/5];

   real_t rho2 = 0.766475*rho_grid[nx/4] + 0.233525*rho_grid[5*nx/6];
   real_t T2   = 0.231744*T_grid[ny/12] + 0.768256*T_grid[4*ny/5];

   check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

   // rho1 - rho2 < tol and T1 - T2 > tol
   rho2 = rho1 + 1e-10;
   check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

   // rho1 - rho2 > tol and T1 - T2 < tol
   rho2 = 0.766475*rho_grid[nx/4] + 0.233525*rho_grid[5*nx/6];
   T2   = T1 + 1e-10;
   check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);

   // rho1 - rho2 < tol and T1 - T2 < tol
    rho2 = rho1 + 1e-10;
    T2   = T1 + 1e-10;
    check_symmetric_itoh_abe_gradient(rho1, T1, rho2, T2);
   return 0;
}

// TEST(flux_symmetry_consistency_test)
// {

// }