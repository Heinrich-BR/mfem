#include "ex41p.hpp"

int main(int argc, char *argv[]) {
  // Initialise MPI and device
  Mpi::Init();
  int myid = Mpi::WorldRank();
  Hypre::Init();
  Device device("cpu");
  if (myid == 0) {
    device.Print();
  }

  // Simulation parameters
  int order = 1;
  int ref_levels = 0;
  real_t xL = 1.0; // Lengths measured in units of 100*rho_s0
  real_t yL = 1.0;

  int nstep = 8000;
  real_t max_t = 12;

  // Create the Ricci2D problem and obtain initial values
  Ricci2D ricci(order, ref_levels, xL, yL, max_t, nstep);
  ricci.formMKB();
  ricci.updateDataCollection(0);
  ricci.Save();

  // Create the time evolution operator
  FE_Evolution t_op(ricci._M_Kdt, ricci._B, ricci._block_offsets,
                    ricci._pmesh->GetComm());

  // Set up the ODE solver
  std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(21);
  ode_solver->Init(t_op);

  real_t t = 0.0;

  // Time evolution
  for (int s = 0; s < ricci._nstep; ++s) {

    if (s != 0) {
      ricci.formMKB();
      t_op.updateMKB(ricci._M_Kdt, ricci._B);
    }

    ode_solver->Step(*ricci._var_blocks, t, ricci._dt);
    ricci.updateVars();
    std::cout << "--------------------------" << std::endl;
    std::cout << "t = " << t << " | dt = " << ricci._dt << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "T Norml2     = " << ricci._vars[T_IDX]->Norml2() << std::endl;
    if (NUM_VARS > OMEGA_IDX)
      std::cout << "Omega Norml2 = " << ricci._vars[OMEGA_IDX]->Norml2() << std::endl;
    if (NUM_VARS > N_IDX)
      std::cout << "n Norml2     = " << ricci._vars[N_IDX]->Norml2() << std::endl;
    std::cout << "Phi Norml2   = " << ricci._phi->Norml2() << std::endl;
    std::cout << "---------------------------" << std::endl;

    ricci.updateDataCollection(s + 1);
    ricci.Save();
  }

  return 0;
}

// Ricci2D methods

Ricci2D::Ricci2D(int order, int ref_levels, real_t xL, real_t yL, real_t max_t,
                 int nstep)
    : _max_t(max_t), _nstep(nstep), _order(order), _ref_levels(ref_levels),
      _xL(xL), _yL(yL), _Binv(40.0), _Lambda(3.0), _S_0n(0.03),
      _rotmat({{0, -1}, {1, 0}}),
      _grad_rotate{std::make_unique<MatrixConstantCoefficient>(_rotmat)} {
  _dt = _max_t / _nstep;
  _dt_coef = std::make_unique<ConstantCoefficient>(_dt);
  buildMesh(xL, yL);
  buildGridFunctions();
  makeRadialCoefficient();
  makeSn();
  setInitialConditions();
  setOutput();

  HYPRE_BigInt size = _h1_fes->GlobalTrueVSize();
  if (Mpi::WorldRank() == 0) {
    std::cout << "Number of finite element unknowns: " << size << std::endl;
  }
}

void Ricci2D::makeRadialCoefficient() {
  // Make a radial coefficient whose origin is at the centre of the mesh
  _r.reset(new TransformedCoefficient(
      new CartesianXCoefficient, new CartesianYCoefficient,
      [this](real_t x, real_t y) {
        return sqrt(pow((x - _xL / 2), 2) + pow((y - _yL / 2), 2));
      }));

  _r_test_gf = std::make_unique<ParGridFunction>(_h1_fes.get());
  _r_test_gf->ProjectCoefficient(*_r);
}

void Ricci2D::makeSn() {
  // Source term S_n, which in the normalised equations is equal to S_T

  real_t r_unit = 1e-2; // 100 rho_s0 in normalised units

  _Sn.reset(new TransformedCoefficient(_r.get(), [r_unit, this](real_t r) {
    real_t r_norm = r / r_unit;
    real_t r_factor = (r_norm-20)/0.5;
    real_t b_att = 5.0; // Border attenuation factor
    return _S_0n * (1 - tanh(r_factor/b_att));
  }));

  _Sn_test_gf->ProjectCoefficient(*_Sn);
}

void Ricci2D::Save() { _dc->Save(); }

void Ricci2D::updateVd() {
  _grad_phi.reset(new GradientGridFunctionCoefficient(_phi.get()));
  _vd.reset(new MatrixVectorProductCoefficient(*_grad_rotate, *_grad_phi));

  _norm_vd.reset(new NormalizedVectorCoefficient(*_vd, _eps2));
  _scaled_norm_vd.reset(
      new ScalarVectorProductCoefficient(_h / 2.0, *_norm_vd));
  _div_vd.reset(div(*_vd));
  _div_vd_scaled.reset(new ProductCoefficient(-_Binv, *_div_vd));
  _vd_test_gf->ProjectCoefficient(*_vd);
}

void Ricci2D::updatePhi() {

  if (NUM_VARS < OMEGA_IDX + 1)
    return; // No need to update ϕ if ω is not evolved in time
  
  // Boundary conditions
  Array<int> ess_tdof_list,
      ess_bdr(_phi->ParFESpace()->GetParMesh()->bdr_attributes.Max());
  ess_bdr = 1;
  _h1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  // Assemble the equation system
  ParBilinearForm a(_h1_fes.get());
  ConstantCoefficient one(1.0);
  a.AddDomainIntegrator(new DiffusionIntegrator(one));
  a.Assemble();

  ParLinearForm b(_h1_fes.get());
  GridFunctionCoefficient _omega_coef(_vars[OMEGA_IDX].get());
  ProductCoefficient _neg_omega_coef(-1.0, _omega_coef);
  b.AddDomainIntegrator(new DomainLFIntegrator(_neg_omega_coef));
  b.Assemble();
  OperatorPtr _A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, *_phi, b, _A, X, B);

  // Solve
  HypreBoomerAMG _prec;
  CGSolver cg(MPI_COMM_WORLD);
  cg.SetRelTol(1e-12);
  cg.SetMaxIter(2000);
  cg.SetPrintLevel(0);
  if (_prec) {
    cg.SetPreconditioner(_prec);
  }
  cg.SetOperator(*_A);
  cg.Mult(B, X);
  a.RecoverFEMSolution(X, b, *_phi);
}

void Ricci2D::updateGFCoefs() {
  _var_gfcoefs.clear();
  _var_gfcoefs.resize(NUM_VARS);
  for (int i = 0; i < NUM_VARS; ++i) {
    _var_gfcoefs[i] = std::make_unique<GridFunctionCoefficient>(_vars[i].get());
  }
  _phi_gfcoef.reset(new GridFunctionCoefficient(_phi.get()));
}

void Ricci2D::updateBLFCoefs() {

  // Update the streamline upwinding coefficient
  _SUW_matcoef.reset(new OuterProductCoefficient(*_vd, *_scaled_norm_vd));

  // Update the thermal exponential coefficient to simplify some expressions
  _thermal_exp.reset(new TransformedCoefficient(
      _var_gfcoefs[T_IDX].get(), _phi_gfcoef.get(),
      [this](real_t T, real_t phi) {
        return exp(_Lambda - phi / sqrt(T * T + _eps2));
      }));

  _DTFT.reset(new TransformedCoefficient(
      _var_gfcoefs[T_IDX].get(), _phi_gfcoef.get(),
      [this](real_t T, real_t phi) {
        return (_dt / 36.) *
                   (1.71 * exp(_Lambda - phi / sqrt(T * T + _eps2)) - 0.71) +
               _dt * 0.0475 * exp(_Lambda - phi / sqrt(T * T + _eps2)) * phi /
                   sqrt(T * T + _eps2);
      }));

  // Update the Gateaux derivative coefficients
  if (NUM_VARS > T_IDX)
    _DTFw.reset(new TransformedCoefficient(
        _var_gfcoefs[T_IDX].get(), _phi_gfcoef.get(),
        [this](real_t T, real_t phi) {
          return (_dt / 24.) * exp(_Lambda - phi / sqrt(T * T + _eps2)) * phi /
                 (T * T + _eps2);
        }));

  if (NUM_VARS > N_IDX)
  {
    _DnFn.reset(
        new TransformedCoefficient(_thermal_exp.get(), [this](real_t exp_term) {
          return (_dt / 24.) * exp_term;
        }));

    _DTFn.reset(new TransformedCoefficient(
        _var_gfcoefs[N_IDX].get(), _DTFw.get(),
        [this](real_t n, real_t DTFw) { return DTFw * n; }));
  }

}

void Ricci2D::updateLFCoefs() {

  _T_DTFT.reset(new ProductCoefficient(*_var_gfcoefs[T_IDX], *_DTFT));
  if (NUM_VARS > OMEGA_IDX)
    _T_DTFw.reset(new ProductCoefficient(*_var_gfcoefs[T_IDX], *_DTFw));
  
  if (NUM_VARS > N_IDX)
  {
    _n_DnFn.reset(new ProductCoefficient(*_var_gfcoefs[N_IDX], *_DnFn));
    _T_DTFn.reset(new ProductCoefficient(*_var_gfcoefs[T_IDX], *_DTFn));
  }
  
  _F.clear();
  _F.resize(NUM_VARS);

  // F_T = (T/36)(1.71 e^(Λ−ϕ/T) − 0.71) − S_T
  _FT_no_source.reset(new TransformedCoefficient(
      _var_gfcoefs[T_IDX].get(), _thermal_exp.get(),
      [](real_t T, real_t exp_term) {
        return (T / 36.) * (1.71 * exp_term - 0.71);
      }));
  _F[T_IDX].reset(new TransformedCoefficient(
      _FT_no_source.get(), _Sn.get(),
      [](real_t FT_no_source, real_t Sn) { return FT_no_source - Sn; }));

  // F_ω = (1/24)(e^(Λ−ϕ/T) − 1)
  if (NUM_VARS > OMEGA_IDX)
    _F[OMEGA_IDX].reset(new TransformedCoefficient(
        _thermal_exp.get(),
        [](real_t exp_term) { return (1. / 24.) * (exp_term - 1.); }));

  // F_n = (n/24) e^(Λ−ϕ/T) − S_n
  if (NUM_VARS > N_IDX)
  {
    _Fn_no_source.reset(new TransformedCoefficient(
        _var_gfcoefs[N_IDX].get(), _thermal_exp.get(),
        [](real_t n, real_t exp_term) { return (n / 24.) * exp_term; }));
    _F[N_IDX].reset(new TransformedCoefficient(
        _Fn_no_source.get(), _Sn.get(),
      [](real_t Fn_no_source, real_t Sn) { return Fn_no_source - Sn; }));
  }
}

void Ricci2D::updateVars() {
  for (int i = 0; i < NUM_VARS; ++i) {
    *_vars[i] = _var_blocks->GetBlock(i);
  }
}

Coefficient *Ricci2D::div(VectorCoefficient &vc) {
  _hdiv_gf->ProjectCoefficient(vc);
  return new DivergenceGridFunctionCoefficient(_hdiv_gf.get());
}

void Ricci2D::formMKB() {
  updatePhi();
  updateVd();
  updateGFCoefs();
  updateBLFCoefs();
  updateLFCoefs();

  DeleteAllBlocks();
  _h_blocks.SetSize(NUM_VARS, NUM_VARS);
  ClearAllBlocks();

  // Make system blocks (block ordering: T, ω, n — see VarIdx).
  _block_offsets.SetSize(NUM_VARS + 1);
  _block_offsets[0] = 0;
  for (int i = 0; i < NUM_VARS; ++i)
    _block_offsets[i + 1] = _h1_fes->GetVSize();
  
  _block_offsets.PartialSum();

  _B.reset(new BlockVector(_block_offsets));
  _var_blocks.reset(new BlockVector(_block_offsets));
  for (int i = 0; i < NUM_VARS; ++i)
    _var_blocks->GetBlock(i) = *_vars[i];

  // Default every block to nullptr; the diagonal loop and the two off-diagonal
  // blocks below overwrite the entries that are non-zero. HypreParMatrixFromBlocks
  // treats nullptr as a zero block.
  for (int i = 0; i < NUM_VARS; ++i) {
    for (int j = 0; j < NUM_VARS; ++j) {
      _h_blocks(i, j) = nullptr;
    }
  }

  // Diagonal terms
  for (int i = 0; i < NUM_VARS; ++i)
  {
      ParBilinearForm m(_h1_fes.get());
      ParBilinearForm k(_h1_fes.get());
      ParLinearForm b(_h1_fes.get());

      // Form K matrix

      SumIntegrator *k_int = new SumIntegrator;
      k_int->AddIntegrator(new ConvectionIntegrator(*_vd, -_Binv)); // Poisson bracket first term
      //k_int->AddIntegrator(new MassIntegrator(*_div_vd_scaled)); // Poisson bracket second term (should = 0)
      k_int->AddIntegrator(new DiffusionIntegrator(*_SUW_matcoef)); // SUW term
      // ScaleIntegrator does NOT take ownership: k owns k_int exclusively.
      ScaleIntegrator *k_dt_int = new ScaleIntegrator(k_int, _dt, false);

      k.AddDomainIntegrator(k_int);
      k.Assemble();
      k.Finalize();

      // Form M + K*dt matrix

      SumIntegrator *m_k_dt_int = new SumIntegrator;
      m_k_dt_int->AddIntegrator(new MassIntegrator); // Time derivative term

      if (i == T_IDX)
          m_k_dt_int->AddIntegrator(new MassIntegrator(*_DTFT)); // Δt D_T F_T
      else if (i == N_IDX)
          m_k_dt_int->AddIntegrator(new MassIntegrator(*_DnFn)); // Δt D_n F_n
      // OMEGA_IDX: identity only — no diagonal Gateaux term in M.

      m_k_dt_int->AddIntegrator(k_dt_int); // K matrix, scaled by dt

      m.AddDomainIntegrator(m_k_dt_int);
      m.Assemble();
      m.Finalize();

      // Form RHS = -K x - b   (b = (F_i, test))
      b.AddDomainIntegrator(new DomainLFIntegrator(*_F[i]));
      b.Assemble();
      b.Neg();
      k.AddMult(_var_blocks->GetBlock(i), b, -1.0);
      _B->GetBlock(i) = b;

      _h_blocks(i, i) = m.ParallelAssemble();
  }

  if (NUM_VARS > OMEGA_IDX)
  {
    ParBilinearForm m_om_T(_h1_fes.get());
    m_om_T.AddDomainIntegrator(new MassIntegrator(*_DTFw));
    m_om_T.Assemble();
    m_om_T.Finalize();
    _h_blocks(OMEGA_IDX, T_IDX) = m_om_T.ParallelAssemble();
  }

  if (NUM_VARS > N_IDX)
  {
    ParBilinearForm m_n_T(_h1_fes.get());
    m_n_T.AddDomainIntegrator(new MassIntegrator(*_DTFn));
    m_n_T.Assemble();
    m_n_T.Finalize();
    _h_blocks(N_IDX, T_IDX) = m_n_T.ParallelAssemble();
  }

  _M_Kdt.reset(HypreParMatrixFromBlocks(_h_blocks));

}

void Ricci2D::buildMesh(real_t xL, real_t yL) {
  Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true, xL,
                                    yL, false);
  for (int l = 0; l < _ref_levels; l++) {
    mesh.UniformRefinement();
  }

  _pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
  _h = _pmesh->GetElementSize(_pmesh->GetTypicalElementTransformation());
}

void Ricci2D::buildGridFunctions() {
  _dim = _pmesh->Dimension();
  _h1_fes = std::make_unique<ParFiniteElementSpace>(
      _pmesh.get(), new H1_FECollection(_order, _dim));
  _hdiv_fes = std::make_unique<ParFiniteElementSpace>(
      _pmesh.get(), new RT_FECollection(_order, _dim));

  _phi = std::make_unique<ParGridFunction>(_h1_fes.get());

  _vars.resize(NUM_VARS);
  for (int i = 0; i < NUM_VARS; ++i) {
    _vars[i] = std::make_unique<ParGridFunction>(_h1_fes.get());
  }

  _vd_test_gf = std::make_unique<ParGridFunction>(_hdiv_fes.get());
  _Sn_test_gf = std::make_unique<ParGridFunction>(_h1_fes.get());

  // Intermediate gf for computing the divergence of v_d
  _hdiv_gf = std::make_unique<ParGridFunction>(_hdiv_fes.get());

  // List is empty for pure Neumann BCs
  _ess_tdof_lists.resize(2);
}

void Ricci2D::setInitialConditions() {
  *_phi = 0.03;
  *_vars[T_IDX] = 1.0e-4;
  if (NUM_VARS > OMEGA_IDX)
    *_vars[OMEGA_IDX] = 0.0;
  if (NUM_VARS > N_IDX)
    *_vars[N_IDX] = 1.0e-4;
}

void Ricci2D::setOutput() {
  _dc = std::make_unique<ParaViewDataCollection>("Ricci2D/Step", _pmesh.get());
  _dc->RegisterField("phi", _phi.get());
  _dc->RegisterField("T", _vars[T_IDX].get());
  if (NUM_VARS > OMEGA_IDX)
  _dc->RegisterField("omega", _vars[OMEGA_IDX].get());
  if (NUM_VARS > N_IDX)
  _dc->RegisterField("n", _vars[N_IDX].get());

  //_dc->RegisterField("Vd", _vd_test_gf.get());
  _dc->RegisterField("Sn", _Sn_test_gf.get());
  //_dc->RegisterField("r", _r_test_gf.get());
}

void Ricci2D::updateDataCollection(int step) {
  _dc->SetCycle(step);
  _dc->SetTime(step * _dt);
}

void Ricci2D::DeleteAllBlocks() {
  for (int i = 0; i < _h_blocks.NumRows(); ++i)
    for (int j = 0; j < _h_blocks.NumCols(); ++j)
      delete _h_blocks(i, j);
  _h_blocks.DeleteAll();
}

void Ricci2D::ClearAllBlocks() {
  for (int i = 0; i < _h_blocks.NumRows(); ++i)
    for (int j = 0; j < _h_blocks.NumCols(); ++j)
      _h_blocks(i,j) = nullptr;
}

// FE_Evolution methods

FE_Evolution::FE_Evolution(std::shared_ptr<HypreParMatrix> M_Kdt, std::shared_ptr<BlockVector> b,
                           Array<int> block_offsets, MPI_Comm comm)
    : TimeDependentOperator(M_Kdt->Height(), M_Kdt->Width()), _z(block_offsets),
      _block_offsets(block_offsets), _M_solver(comm)
{
  updateMKB(M_Kdt, b);
}

void FE_Evolution::updateMKB(std::shared_ptr<HypreParMatrix> M_Kdt, std::shared_ptr<BlockVector> b)
{
  _M_Kdt = M_Kdt;
  _b = b;
  // _M_Kdt is rebuilt every outer time step; the factorization must track it.
  _SA.reset(new SuperLURowLocMatrix(*_M_Kdt));
  _M_solver.SetOperator(*_SA);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const {
  // y = (M+K*dt)^{-1} (-K x - b)
  std::cout << "We shouldn't be here!" << std::endl;
}

void FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &y) {
  // RHS of the equation: -K*x-b
  _dt = dt;
  _M_solver.Mult(*_b, y);
}

void FE_Evolution::SetTimeStep(real_t dt) { _dt = dt; }
