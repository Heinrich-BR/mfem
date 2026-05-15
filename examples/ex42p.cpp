#include "ex42p.hpp"

#include <cmath>
#include <utility>

void RogersRicciNLFIntegrator::AssembleElementVector(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector*> &elfun,
   const Array<Vector*> &elvec)
{
   MFEM_VERIFY(el.Size() == NUM_VARS,
               "RogersRicciNLFIntegrator expects exactly "
               << NUM_VARS << " FE spaces.");
   const int dof = el[0]->GetDof();

   Vector shape(dof);
   for (int s = 0; s < NUM_VARS; ++s)
   {
      elvec[s]->SetSize(dof);
      *elvec[s] = 0.0;
   }

   // Single H1 space across blocks → same geometry/order on every block.
   // 2*p + 3 is the order used by IncompressibleNeoHookeanIntegrator and is
   // generous enough for the exponential coupling here.
   const IntegrationRule &ir =
      IntRules.Get(el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 3);

   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);
      el[0]->CalcShape(ip, shape);

      // omega does not appear in any F_x (PDF eq. 7), so we never read
      // elfun[OMEGA_IDX] here — F_omega depends on T and phi only.
      const real_t T = shape * (*elfun[T_IDX]);
      const real_t n = shape * (*elfun[N_IDX]);

      const real_t phi = _phi->Eval(Tr, ip);
      const real_t Sn  = _Sn->Eval(Tr, ip);

      // Regularisation: replace 1/T by 1/Treg with Treg = sqrt(T^2 + eps2).
      // The Jacobian below uses the chain-rule derivative consistent with
      // this regularisation, so Newton convergence is unaffected.
      const real_t Treg = std::sqrt(T*T + _eps2);
      const real_t A    = _Lambda - phi / Treg;
      const real_t eA   = std::exp(A);

      const real_t F_T = (T / 36.0) * (1.71 * eA - 0.71) - Sn;
      const real_t F_w = (eA - 1.0) / 24.0;
      const real_t F_n = (n / 24.0) * eA - Sn;

      const real_t w = ip.weight * Tr.Weight();

      // Always evolve T; omega and n only if their rows are active.
      // Frozen rows stay at zero residual contribution → Newton sees only
      // M*k_s, which drives k_s to 0 and leaves the predictor state intact.
      elvec[T_IDX]->Add(w * F_T, shape);
      if (_num_active >= 2) { elvec[OMEGA_IDX]->Add(w * F_w, shape); }
      if (_num_active >= 3) { elvec[N_IDX]    ->Add(w * F_n, shape); }
   }
}

void RogersRicciNLFIntegrator::AssembleElementGrad(
   const Array<const FiniteElement*> &el,
   ElementTransformation &Tr,
   const Array<const Vector*> &elfun,
   const Array2D<DenseMatrix*> &elmats)
{
   MFEM_VERIFY(el.Size() == NUM_VARS,
               "RogersRicciNLFIntegrator expects exactly "
               << NUM_VARS << " FE spaces.");
   const int dof = el[0]->GetDof();

   Vector shape(dof);
   for (int s = 0; s < NUM_VARS; ++s)
   {
      for (int t = 0; t < NUM_VARS; ++t)
      {
         elmats(s, t)->SetSize(dof, dof);
         *elmats(s, t) = 0.0;
      }
   }

   const IntegrationRule &ir =
      IntRules.Get(el[0]->GetGeomType(), 2 * el[0]->GetOrder() + 3);

   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr.SetIntPoint(&ip);
      el[0]->CalcShape(ip, shape);

      const real_t T = shape * (*elfun[T_IDX]);
      const real_t n = shape * (*elfun[N_IDX]);

      const real_t phi  = _phi->Eval(Tr, ip);
      const real_t Treg = std::sqrt(T*T + _eps2);
      const real_t A    = _Lambda - phi / Treg;
      const real_t eA   = std::exp(A);

      // d(1/Treg)/dT = -T/Treg^3   ⇒   dA/dT = phi*T/Treg^3.
      // For T >> sqrt(eps2) this collapses to PDF eq. 8's phi/T^2 form;
      // near T == 0 it stays bounded, unlike the bare PDF formula.
      const real_t Treg3  = Treg * Treg * Treg;
      const real_t deA_dT = eA * phi * T / Treg3;

      // dF_T/dT = (1.71 eA - 0.71)/36 + (T/36)·1.71·dE/dT
      const real_t dFT_dT = (1.71 * eA - 0.71) / 36.0
                            + (T * 1.71 / 36.0) * deA_dT;
      const real_t dFw_dT = deA_dT / 24.0;
      const real_t dFn_dT = n * deA_dT / 24.0;
      const real_t dFn_dn = eA / 24.0;

      const real_t w = ip.weight * Tr.Weight();

      // Same gating as the residual: only contribute to active-row blocks.
      AddMult_a_VVt(w * dFT_dT, shape, *elmats(T_IDX, T_IDX));
      if (_num_active >= 2)
      {
         AddMult_a_VVt(w * dFw_dT, shape, *elmats(OMEGA_IDX, T_IDX));
      }
      if (_num_active >= 3)
      {
         AddMult_a_VVt(w * dFn_dT, shape, *elmats(N_IDX, T_IDX));
         AddMult_a_VVt(w * dFn_dn, shape, *elmats(N_IDX, N_IDX));
      }
   }
}

// ===========================================================================
//                       Ricci2DNonlinear implementation
// ===========================================================================

Ricci2DNonlinear::Ricci2DNonlinear(MPI_Comm comm, int order, int ref_levels,
                                   real_t xL, real_t yL, int num_active)
   : _xL(xL), _yL(yL), _order(order), _ref_levels(ref_levels),
     _num_active(num_active),
     _rotmat({{0.0, -1.0}, {1.0, 0.0}}),
     _comm(comm)
{
   MFEM_VERIFY(num_active >= 1 && num_active <= NUM_VARS,
               "Ricci2DNonlinear: num_active must be 1, 2, or 3.");
}

void Ricci2DNonlinear::Setup()
{
   buildMesh();
   buildSpaces();
   buildState();
   buildCoefficients();
   buildQuadratureSamples();
   buildMassMatrix();
   setInitialConditions();
   syncBlocksFromGridFuncs(*_var_blocks);

   // Sn is geometry-only — project it into its quadrature-function buffer
   // once and the integrator will look it up with O(1) cost forever after.
   _Sn->Project(*_Sn_qf);

   // phi starts uniform (constant 0.03) — seed the quadrature sample so the
   // first F_form action sees the initial condition, not zeros.
   projectPhiToQuadrature();

   buildNonlinearForm();
   setOutputCollection();

   // Build the phi-Poisson solver once: the Laplacian operator and the
   // boundary topology don't change with time, so we can amortise the AMG
   // setup across every implicit stage.
   buildPhiSolver();

   // Build the K (advection + SUW) bilinear form once with the integrators
   // wired to live coefficients; reassembleK() will Update/Assemble in place.
   buildKForm();

   // Initial K with zero v_d (phi is uniform → grad phi = 0).
   refreshDriftCoefs();
   reassembleK();
}

void Ricci2DNonlinear::buildMesh()
{
   // Match ex41p: 64x64 quads on [0,xL] x [0,yL].
   Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true,
                                     _xL, _yL, false);
   for (int l = 0; l < _ref_levels; ++l) { mesh.UniformRefinement(); }

   _pmesh = std::make_unique<ParMesh>(_comm, mesh);
   _h = _pmesh->GetElementSize(_pmesh->GetTypicalElementTransformation());
}

void Ricci2DNonlinear::buildSpaces()
{
   _h1_fes = std::make_unique<ParFiniteElementSpace>(
                _pmesh.get(), new H1_FECollection(_order, _dim));
}

void Ricci2DNonlinear::buildState()
{
   _vars.resize(NUM_VARS);
   for (int i = 0; i < NUM_VARS; ++i)
   {
      _vars[i] = std::make_unique<ParGridFunction>(_h1_fes.get());
   }
   _phi = std::make_unique<ParGridFunction>(_h1_fes.get());

   _block_trueOffsets.SetSize(NUM_VARS + 1);
   _block_trueOffsets[0] = 0;
   for (int i = 0; i < NUM_VARS; ++i)
   {
      _block_trueOffsets[i + 1] = _h1_fes->GetTrueVSize();
   }
   _block_trueOffsets.PartialSum();

   _var_blocks = std::make_unique<BlockVector>(_block_trueOffsets);
}

void Ricci2DNonlinear::buildCoefficients()
{
   // Radial distance from mesh centre (used by S_n, lifted verbatim from ex41p).
   _r.reset(new TransformedCoefficient(
               new CartesianXCoefficient, new CartesianYCoefficient,
               [this](real_t x, real_t y) {
      return std::sqrt(std::pow(x - _xL/2.0, 2) + std::pow(y - _yL/2.0, 2));
   }));

   // Firedrake reference form (rr.py:rr_src_ufl):
   //   S(r) = (S_0n / 2) * (1 - tanh((r - rs) / Ls))
   // with rs = 20*rho_s0, Ls = 0.5*rho_s0.  Since the mesh is now in rho_s0
   // units, rs and Ls are taken as 20 and 0.5 in mesh units directly.
   const real_t rs = 20.0;
   const real_t Ls = 0.5;
   _Sn.reset(new TransformedCoefficient(
                _r.get(), [this, rs, Ls](real_t r) {
      return 0.5 * _S_0n * (1.0 - std::tanh((r - rs) / Ls));
   }));

   // 90° rotation matrix, used to build v_d from grad(phi).
   _grad_rotate = std::make_unique<MatrixConstantCoefficient>(_rotmat);

   // phi-dependent coefficients.  v_d = R · grad(phi) with R = [[0,-1],[1,0]],
   // so v_d = (-d_y phi, d_x phi) — the ExB-drift direction in normalised units.
   _phi_gfcoef = std::make_unique<GridFunctionCoefficient>(_phi.get());
   _grad_phi   = std::make_unique<GradientGridFunctionCoefficient>(_phi.get());
   _vd         = std::make_unique<MatrixVectorProductCoefficient>(*_grad_rotate,
                                                                  *_grad_phi);

   // True drift velocity v_E = (1/B) * v_d.  This is what appears in both the
   // convection term (-(1/B) pb(u, phi) = -v_E · grad(u)) and in the
   // streamline-upwinding stabilisation in the Firedrake reference.
   _v_E = std::make_unique<ScalarVectorProductCoefficient>(_Binv, *_vd);

   // SUW matching Firedrake's SU_term (rr.py:54-63):
   //     0.5 * h * (v_E · grad u)(v_E · grad v) / sqrt(|v_E|^2 + eps^2)
   // As a DiffusionIntegrator(M) bilinear form with M = c(x) * v_E ⊗ v_E,
   // where c(x) = h / (2 sqrt(|v_E|^2 + eps^2)) — a *smooth* regularisation,
   // contrasting with MFEM's NormalizedVectorCoefficient which is a
   // threshold (returns zero below tol).  The smooth form caps the SUW
   // coefficient at h/(2*eps) when v_E -> 0, which keeps the Jacobian
   // well-conditioned in regions where the drift velocity vanishes.
   _v_E_sq     = std::make_unique<InnerProductCoefficient>(*_v_E, *_v_E);
   _suw_scalar = std::make_unique<TransformedCoefficient>(
                    _v_E_sq.get(),
                    [h = _h, eps2 = _eps2](real_t v_sq)
   {
      return h / (2.0 * std::sqrt(v_sq + eps2));
   });
   _v_E_outer   = std::make_unique<OuterProductCoefficient>(*_v_E, *_v_E);
   _SUW_matcoef = std::make_unique<ScalarMatrixProductCoefficient>(
                     *_suw_scalar, *_v_E_outer);
}

void Ricci2DNonlinear::buildQuadratureSamples()
{
   // Same order as RogersRicciNLFIntegrator's integration rule (2p+3), so
   // QuadratureFunctionCoefficient::Eval indexes into the same point set
   // the integrator iterates over.
   const int qorder = 2 * _order + 3;
   _qspace  = std::make_unique<QuadratureSpace>(_pmesh.get(), qorder);
   _phi_qf  = std::make_unique<QuadratureFunction>(*_qspace);
   _Sn_qf   = std::make_unique<QuadratureFunction>(*_qspace);
   _phi_qfc = std::make_unique<QuadratureFunctionCoefficient>(*_phi_qf);
   _Sn_qfc  = std::make_unique<QuadratureFunctionCoefficient>(*_Sn_qf);
}

void Ricci2DNonlinear::projectPhiToQuadrature()
{
   _phi_qf->ProjectGridFunction(*_phi);
}

void Ricci2DNonlinear::buildMassMatrix()
{
   _M_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _M_form->AddDomainIntegrator(new MassIntegrator);
   _M_form->Assemble();
   _M_form->Finalize();
   _M_mat.reset(_M_form->ParallelAssemble());
}

void Ricci2DNonlinear::buildKForm()
{
   // K = -(1/B) (v_d · grad u, v) + (c v_E ⊗ v_E : grad u ⊗ grad v).
   // The integrators reference the live coefficients owned by Ricci2DNonlinear,
   // so updates to phi (and hence v_d, v_E) flow through to the next Assemble
   // without rewiring.
   _K_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _K_form->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv));
   _K_form->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef));
}

void Ricci2DNonlinear::buildPhiSolver()
{
   // Essential dofs: Dirichlet on every boundary attribute (values held at
   // whatever _phi was set to in setInitialConditions, i.e. the constant
   // 0.03 used as the symmetry-breaking inlet potential).
   Array<int> ess_bdr(_pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   _h1_fes->GetEssentialTrueDofs(ess_bdr, _ess_tdof_list_phi);

   // Bilinear form: ∇·∇  (assembled once; the operator is static).
   _one_phi = std::make_unique<ConstantCoefficient>(1.0);
   _a_phi   = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _a_phi->AddDomainIntegrator(new DiffusionIntegrator(*_one_phi));
   _a_phi->Assemble();
   _a_phi->Finalize();

   // Long-lived linear form: omega coefficient is a live view into the
   // _vars[OMEGA_IDX] grid function, so re-Assembling each stage picks up
   // the latest predictor state.
   _omega_coef     = std::make_unique<GridFunctionCoefficient>(
                        _vars[OMEGA_IDX].get());
   _neg_omega_coef = std::make_unique<ProductCoefficient>(-1.0, *_omega_coef);
   _b_phi          = std::make_unique<ParLinearForm>(_h1_fes.get());
   _b_phi->AddDomainIntegrator(new DomainLFIntegrator(*_neg_omega_coef));

   // Trigger the BC-elimination path once so _A_phi points to the
   // already-eliminated p_mat.  Subsequent FormLinearSystem calls in
   // updatePhi will reuse this matrix (see ParBilinearForm::FormSystemMatrix
   // — after the first call, `mat` is NULL and only the BC application on B
   // is repeated, with no re-assembly cost).
   _b_phi->Assemble();
   Vector X_tmp, B_tmp;
   _a_phi->FormLinearSystem(_ess_tdof_list_phi, *_phi, *_b_phi,
                            _A_phi, X_tmp, B_tmp);

   // AMG hierarchy built once against the eliminated parallel matrix.
   _amg_phi = std::make_unique<HypreBoomerAMG>();
   _amg_phi->SetPrintLevel(0);
   _amg_phi->SetOperator(*_A_phi);

   _cg_phi = std::make_unique<CGSolver>(_comm);
   _cg_phi->SetRelTol(1e-12);
   _cg_phi->SetMaxIter(2000);
   _cg_phi->SetPrintLevel(0);
   _cg_phi->SetPreconditioner(*_amg_phi);
   _cg_phi->SetOperator(*_A_phi);
}

void Ricci2DNonlinear::buildNonlinearForm()
{
   Array<ParFiniteElementSpace*> fes(NUM_VARS);
   for (int i = 0; i < NUM_VARS; ++i) { fes[i] = _h1_fes.get(); }

   _F_form = std::make_unique<ParBlockNonlinearForm>(fes);
   _F_form->AddDomainIntegrator(
      new RogersRicciNLFIntegrator(*_phi_qfc, *_Sn_qfc, _num_active,
                                 _Lambda, _eps2));
}

void Ricci2DNonlinear::setInitialConditions()
{
   *_phi              = 0.03;
   *_vars[T_IDX]      = 1.0e-4;
   *_vars[OMEGA_IDX]  = 0.0;
   *_vars[N_IDX]      = 1.0e-4;
}

void Ricci2DNonlinear::setOutputCollection()
{
   _dc = std::make_unique<ParaViewDataCollection>("Ricci2DNonlinearOpt/Step",
                                                  _pmesh.get());
   _dc->RegisterField("phi",   _phi.get());
   _dc->RegisterField("T",     _vars[T_IDX].get());
   _dc->RegisterField("omega", _vars[OMEGA_IDX].get());
   _dc->RegisterField("n",     _vars[N_IDX].get());
}

void Ricci2DNonlinear::syncGridFuncsFromBlocks(const BlockVector &u_blk)
{
   for (int i = 0; i < NUM_VARS; ++i)
   {
      _vars[i]->SetFromTrueDofs(u_blk.GetBlock(i));
   }
}

void Ricci2DNonlinear::syncBlocksFromGridFuncs(BlockVector &u_blk) const
{
   for (int i = 0; i < NUM_VARS; ++i)
   {
      _vars[i]->ParallelProject(u_blk.GetBlock(i));
   }
}

void Ricci2DNonlinear::pullOmegaFromBlocks(const BlockVector &u_blk)
{
   _vars[OMEGA_IDX]->SetFromTrueDofs(u_blk.GetBlock(OMEGA_IDX));
}

void Ricci2DNonlinear::updatePhi()
{
   // RHS is the only thing that changes between stages: the omega coefficient
   // is a live view into _vars[OMEGA_IDX] (refreshed by pullOmegaFromBlocks
   // just before this call), so re-Assembling _b_phi picks up the predictor.
   _b_phi->Assemble();

   // FormLinearSystem after the first call returns the same eliminated p_mat
   // (no re-assembly), applies BC adjustment to B, and packs X with current
   // _phi true-dof values.  The cached AMG and CG operators were bound to
   // this same matrix in buildPhiSolver().
   OperatorPtr A;
   Vector X, B;
   _a_phi->FormLinearSystem(_ess_tdof_list_phi, *_phi, *_b_phi, A, X, B);

   _cg_phi->Mult(B, X);
   _a_phi->RecoverFEMSolution(X, *_b_phi, *_phi);
}

void Ricci2DNonlinear::refreshDriftCoefs()
{
   // GradientGridFunctionCoefficient / MatrixVectorProductCoefficient /
   // NormalizedVectorCoefficient / OuterProductCoefficient all evaluate
   // lazily from the underlying state — there is no cached data to
   // invalidate.  This hook exists for clarity and as a place to hang
   // diagnostics later.
}

void Ricci2DNonlinear::reassembleK()
{
   // Update() drops the previous SparseMatrix and the eliminated parallel
   // matrix without touching the attached integrators or coefficients, so
   // the next Assemble re-uses the same integrator chain and just picks up
   // the new v_d / v_E values that depend on the freshly-updated phi.
   _K_form->Update();
   _K_form->Assemble();
   _K_form->Finalize();
   _K_mat.reset(_K_form->ParallelAssemble());
}

void Ricci2DNonlinear::updateDataCollection(int step, real_t t)
{
   _dc->SetCycle(step);
   _dc->SetTime(t);
}

void Ricci2DNonlinear::Save()
{
   _dc->Save();
}

RicciImplicitStageOp::RicciImplicitStageOp(Ricci2DNonlinear &ricci)
   : Operator(ricci.BlockTrueOffsets().Last()),
     _ricci(ricci),
     _gamma(0.0),
     _u_pred(nullptr),
     _z(ricci.BlockTrueOffsets()),
     _tmp_block(ricci.H1FES()->GetTrueVSize()),
     _Jac_block(ricci.BlockTrueOffsets())
{
   // Workspace vectors participate in HypreParMatrix SpMV operations that go
   // to device when Hypre is built with GPU support; mark them so the memory
   // class follows the active Device().
   _z.UseDevice(true);
   _tmp_block.UseDevice(true);
}

void RicciImplicitStageOp::SetParameters(real_t gamma, const BlockVector &u_pred)
{
   _gamma = gamma;
   _u_pred = &u_pred;

   // The M + gamma*K block is shared across every Newton iter of this stage
   // and across every variable row.  Build it once here so GetGradient only
   // has to fold in the reaction contribution F'_ss.
   HypreParMatrix *M = _ricci.MassMat();
   HypreParMatrix *K = _ricci.KMat();
   _M_plus_gK.reset(Add(1.0, *M, _gamma, *K));
}

void RicciImplicitStageOp::Mult(const Vector &k_vec, Vector &R_vec) const
{
   MFEM_VERIFY(_u_pred != nullptr,
               "RicciImplicitStageOp: SetParameters() not called before Mult().");

   // z = u_pred + gamma * k.  (`_z` is a BlockVector so GetBlock(s) works.)
   add(*_u_pred, _gamma, k_vec, _z);

   // R = F(z)   (each block s receives (F_s(z), test_s) from the integrator).
   _ricci.FForm()->Mult(_z, R_vec);

   // R += M·k + K·z, applied block-by-block.  M and K are single-block H1
   // operators replicated on the diagonal of the (otherwise zero-coupled in
   // the linear part) 3-variable system.
   const Array<int> &offs = _ricci.BlockTrueOffsets();
   BlockVector k_blk(const_cast<Vector&>(k_vec), offs);
   BlockVector R_blk(R_vec, offs);

   HypreParMatrix *M = _ricci.MassMat();
   HypreParMatrix *K = _ricci.KMat();
   for (int s = 0; s < NUM_VARS; ++s)
   {
      M->Mult(k_blk.GetBlock(s), _tmp_block);
      R_blk.GetBlock(s) += _tmp_block;
      K->Mult(_z.GetBlock(s), _tmp_block);
      R_blk.GetBlock(s) += _tmp_block;
   }
}

Operator &RicciImplicitStageOp::GetGradient(const Vector &k_vec) const
{
   MFEM_VERIFY(_u_pred != nullptr,
               "RicciImplicitStageOp: SetParameters() not called before "
               "GetGradient().");

   // z = u_pred + gamma * k.
   add(*_u_pred, _gamma, k_vec, _z);

   // F's 3x3 block gradient at z.  The integrator gates this on num_active
   // — frozen rows produce no contribution.  Non-zero blocks (fully active)
   // are (T,T), (omega,T), (n,T), (n,n).
   BlockOperator &Fgrad = _ricci.FForm()->GetGradient(_z);
   auto Fblock = [&](int i, int j) -> HypreParMatrix &
   {
      return dynamic_cast<HypreParMatrix &>(Fgrad.GetBlock(i, j));
   };

   const int       n_active = _ricci.NumActive();
   HypreParMatrix *M = _ricci.MassMat();

   // T row is always active.  M + gamma*K is precomputed in SetParameters.
   _diag_T.reset(Add(1.0, *_M_plus_gK, _gamma, Fblock(T_IDX, T_IDX)));
   _Jac_block.SetDiagonalBlock(T_IDX, _diag_T.get());

   // omega row: M + gamma*K if active (F'_omega,omega = 0), bare M if frozen
   // so Newton drives k_omega to 0.
   _Jac_block.SetDiagonalBlock(OMEGA_IDX,
                               (n_active >= 2) ? _M_plus_gK.get() : M);

   // n row: M + gamma*K + gamma*F'_{n,n} if active, bare M if frozen.
   if (n_active >= 3)
   {
      _diag_n.reset(Add(1.0, *_M_plus_gK, _gamma, Fblock(N_IDX, N_IDX)));
      _Jac_block.SetDiagonalBlock(N_IDX, _diag_n.get());
   }
   else
   {
      _diag_n.reset();
      _Jac_block.SetDiagonalBlock(N_IDX, M);
   }

   // Off-diagonals: gamma * F'_{omega,T} and gamma * F'_{n,T}.
   // Add(0, X, gamma, X) yields a fresh gamma*X HypreParMatrix.
   if (n_active >= 2)
   {
      _gFwT.reset(Add(0.0, Fblock(OMEGA_IDX, T_IDX),
                      _gamma, Fblock(OMEGA_IDX, T_IDX)));
      _Jac_block.SetBlock(OMEGA_IDX, T_IDX, _gFwT.get());
   }
   if (n_active >= 3)
   {
      _gFnT.reset(Add(0.0, Fblock(N_IDX, T_IDX),
                      _gamma, Fblock(N_IDX, T_IDX)));
      _Jac_block.SetBlock(N_IDX, T_IDX, _gFnT.get());
   }

   return _Jac_block;
}

// ===========================================================================
//             SuperLU adapter + RicciTimeOperator implementation
// ===========================================================================

namespace
{
// Backtracking Armijo line search on top of MFEM's NewtonSolver.  Mirrors
// PETSc's "snes_linesearch_type: l2" (which Firedrake's Rogers-Ricci config
// uses, see 2Drogers-ricci.py:67-75).  Without this, a Newton step in the
// stiff/swinging-phi regime can move the iterate to a state where the
// reaction exp(Lambda - phi/T) is many orders of magnitude away from the
// previous iterate, causing the residual to grow rather than shrink.
class BacktrackingNewtonSolver : public NewtonSolver
{
public:
#ifdef MFEM_USE_MPI
   explicit BacktrackingNewtonSolver(MPI_Comm comm) : NewtonSolver(comm) {}
#endif

   // Armijo backtrack: find alpha in (0, 1] such that
   //     ||F(x - alpha c) - b|| <= (1 - sigma * alpha) ||F(x) - b||
   // halving alpha each failed trial, up to max_tries.
   real_t ComputeScalingFactor(const Vector &x, const Vector &b) const override
   {
      const real_t norm0 = Norm(r);    // r is the residual at x, stored by Mult
      if (norm0 == 0.0) { return 1.0; }

      const real_t sigma = 1e-4;
      const int max_tries = 20;
      Vector x_trial(x.Size()), r_trial(r.Size());
      real_t alpha = 1.0;

      for (int i = 0; i < max_tries; ++i)
      {
         add(x, -alpha, c, x_trial);
         oper->Mult(x_trial, r_trial);
         if (b.Size() == r_trial.Size()) { r_trial -= b; }
         const real_t norm_trial = Norm(r_trial);
         if (norm_trial <= (1.0 - sigma * alpha) * norm0) { return alpha; }
         alpha *= 0.5;
      }
      // Couldn't satisfy Armijo; return the smallest tried so Newton keeps
      // making progress rather than aborting.  If this happens often, the
      // outer time step is the real culprit.
      return alpha;
   }
};

// Block-aware Krylov solver consumed by NewtonSolver as its linear solver.
//
// NewtonSolver hands us the Jacobian via SetOperator(op) and expects us to
// solve J s = r on Mult(r, s).  We:
//   - cast the operator down to BlockOperator (built by RicciImplicitStageOp),
//   - extract its three diagonal HypreParMatrix blocks,
//   - rebuild the per-variable AMG hierarchies that sit on the diagonal of a
//     BlockDiagonalPreconditioner, and
//   - call GMRES with the BlockOperator as the system operator and the
//     block-diag preconditioner.
//
// Inherits from IterativeSolver so that NewtonSolver::SetAdaptiveLinRtol
// (Eisenstat-Walker forcing) can call our SetRelTol — we override Mult to
// forward the inherited rel_tol/abs_tol/max_iter to the internal GMRES.
class BlockNewtonLinearSolver : public IterativeSolver
{
public:
   BlockNewtonLinearSolver(MPI_Comm comm,
                           real_t rtol = 1e-10,
                           int max_it = 200,
                           int kdim = 50)
      : IterativeSolver(comm),
        _gmres(comm)
   {
      SetRelTol(rtol);
      SetAbsTol(0.0);
      SetMaxIter(max_it);
      _gmres.SetKDim(kdim);
      _gmres.SetPrintLevel(0);
      _amgs.resize(NUM_VARS);
      for (int s = 0; s < NUM_VARS; ++s)
      {
         _amgs[s] = std::make_unique<HypreBoomerAMG>();
         _amgs[s]->SetPrintLevel(0);
      }
   }

   void SetOperator(const Operator &op) override
   {
      const BlockOperator *bop = dynamic_cast<const BlockOperator*>(&op);
      MFEM_VERIFY(bop != nullptr,
                  "BlockNewtonLinearSolver: expected a BlockOperator from "
                  "RicciImplicitStageOp::GetGradient().");

      // (Re)build each AMG on the matching diagonal HypreParMatrix first, so
      // its height/width are valid before the block-diag preconditioner runs
      // its dimension check in SetDiagonalBlock.  GetBlock is non-const on
      // BlockOperator, so cast away const — we don't mutate.
      BlockOperator &bopnc = const_cast<BlockOperator&>(*bop);
      for (int s = 0; s < NUM_VARS; ++s)
      {
         HypreParMatrix &diag = dynamic_cast<HypreParMatrix&>(
                                   bopnc.GetBlock(s, s));
         _amgs[s]->SetOperator(diag);
      }

      // Build the BDP lazily so we don't need the offsets in the ctor.
      // The AMG pointers remain valid across Newton iters, so a single BDP
      // suffices for the whole stage (and run).
      if (!_bdp)
      {
         _bdp = std::make_unique<BlockDiagonalPreconditioner>(bop->RowOffsets());
         for (int s = 0; s < NUM_VARS; ++s)
         {
            _bdp->SetDiagonalBlock(s, _amgs[s].get());
         }
         _gmres.SetPreconditioner(*_bdp);
      }

      _gmres.SetOperator(*bop);
      height = bop->Height();
      width  = bop->Width();
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      // Propagate any Eisenstat-Walker rtol set by NewtonSolver since the
      // last call.  rel_tol / max_iter live on the IterativeSolver base; the
      // internal GMRES has its own copies.
      _gmres.SetRelTol(rel_tol);
      _gmres.SetAbsTol(abs_tol);
      _gmres.SetMaxIter(max_iter);
      _gmres.Mult(b, x);
   }

private:
   mutable GMRESSolver                          _gmres;
   std::unique_ptr<BlockDiagonalPreconditioner> _bdp;
   std::vector<std::unique_ptr<HypreBoomerAMG>> _amgs;
};
} // anonymous namespace

RicciTimeOperator::RicciTimeOperator(MPI_Comm comm, Ricci2DNonlinear &ricci,
                                     real_t newton_rtol, int newton_max_iter,
                                     real_t lin_rtol, int lin_max_iter,
                                     int kdim)
   : TimeDependentOperator(ricci.BlockTrueOffsets().Last(),
                           /*t=*/0.0,
                           TimeDependentOperator::IMPLICIT),
     _ricci(ricci),
     _stage_op(ricci),
     _newton(new BacktrackingNewtonSolver(comm)),
     _lin_solver(new BlockNewtonLinearSolver(comm, lin_rtol,
                                             lin_max_iter, kdim))
{
   _newton->iterative_mode = false;       // initial guess for k is zero
   _newton->SetSolver(*_lin_solver);
   _newton->SetOperator(_stage_op);
   _newton->SetRelTol(newton_rtol);
   _newton->SetAbsTol(0.0);
   _newton->SetMaxIter(newton_max_iter);
   _newton->SetPrintLevel(1);             // print Newton iterations per stage

   // Eisenstat-Walker forcing is opt-in via EnableEisenstatWalker() (CLI: -ew).
   // In this regime Newton usually converges in 1-2 iters per stage, so a
   // tight fixed linear rtol (the default constructed above) is typically the
   // right choice; EW pays off mostly when Newton chains get long.
}

void RicciTimeOperator::EnableEisenstatWalker(real_t rtol0, real_t rtol_max)
{
   _newton->SetAdaptiveLinRtol(/*type=*/2, rtol0, rtol_max);
}

void RicciTimeOperator::ImplicitSolve(real_t gamma, const Vector &u_pred,
                                      Vector &k)
{
   // The ODESolver hands us a flat true-dof Vector; view it block-wise so we
   // can reach into the omega component for the phi solve and so the stage
   // operator can do block-by-block arithmetic.
   BlockVector u_pred_blk(const_cast<Vector&>(u_pred),
                          _ricci.BlockTrueOffsets());

   // (1) omega-from-predictor → (2) phi from Poisson → (3) refresh drift →
   // (4) re-assemble K with the new vd → (5) resample phi at quadrature
   // points for the reaction integrator's hot loop.
   _ricci.pullOmegaFromBlocks(u_pred_blk);
   _ricci.updatePhi();
   _ricci.refreshDriftCoefs();
   _ricci.reassembleK();
   _ricci.projectPhiToQuadrature();

   // (5) Wire up the stage equation R(k) = M k + K(u + γk) + F(u + γk).
   _stage_op.SetParameters(gamma, u_pred_blk);

   // (6) Newton drives R(k) = 0.  Empty `b` is interpreted as zero RHS.
   k = 0.0;
   Vector zero;
   _newton->Mult(zero, k);
   MFEM_VERIFY(_newton->GetConverged(),
               "RicciTimeOperator::ImplicitSolve: Newton failed to converge "
               "(gamma = " << gamma << ").");
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   const int myid = Mpi::WorldRank();
   Hypre::Init();

   // ---- options ----------------------------------------------------------
   const char *device_config = "cpu";   // "cuda", "hip", "occa-cpu", ...
                                        // (requires MFEM/HYPRE built with the
                                        //  matching backend; on GPU also
                                        //  needs HYPRE_USING_GPU).
   int    order             = 1;
   int    ser_ref_levels    = 0;
   int    par_ref_levels    = 0;
   int    ode_solver_type   = 21;   // 21 = Backward Euler; 23 = SDIRK23
   real_t xL                = 100.0;   // domain = [0, 100]² in rho_s0 units
   real_t yL                = 100.0;
   real_t t_final           = 6.0;
   real_t dt                = 2.4e-3;
   real_t newton_rtol       = 1e-8;
   int    newton_max_iter   = 15;
   real_t lin_rtol          = 1e-10;
   int    lin_max_iter      = 200;
   int    kdim              = 50;
   bool   enable_ew         = false;
   real_t ew_rtol0          = 0.5;
   real_t ew_rtol_max       = 0.9;
   int    num_active        = NUM_VARS;
   int    vis_steps         = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure() "
                  "(e.g. 'cpu', 'cuda', 'hip').");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the H1 finite element space.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Serial uniform refinements before partitioning.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Parallel uniform refinements after partitioning.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&xL, "-xL", "--x-length", "Domain length in x.");
   args.AddOption(&yL, "-yL", "--y-length", "Domain length in y.");
   args.AddOption(&t_final, "-tf", "--t-final", "Final time.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&newton_rtol, "-rtol", "--newton-rtol",
                  "Relative tolerance for the Newton solver.");
   args.AddOption(&newton_max_iter, "-nmax", "--newton-max-iter",
                  "Maximum Newton iterations per implicit stage.");
   args.AddOption(&lin_rtol, "-ltol", "--lin-rtol",
                  "Relative tolerance for the inner GMRES.");
   args.AddOption(&lin_max_iter, "-lmax", "--lin-max-iter",
                  "Maximum GMRES iterations per Newton step.");
   args.AddOption(&kdim, "-kdim", "--krylov-dim",
                  "Krylov subspace dimension for GMRES.");
   args.AddOption(&enable_ew, "-ew", "--eisenstat-walker",
                  "-no-ew", "--no-eisenstat-walker",
                  "Enable Eisenstat-Walker adaptive linear tolerance.");
   args.AddOption(&ew_rtol0, "-ew-rtol0", "--ew-rtol-initial",
                  "Initial Eisenstat-Walker rtol.");
   args.AddOption(&ew_rtol_max, "-ew-rtol-max", "--ew-rtol-max",
                  "Maximum (loosest) Eisenstat-Walker rtol.");
   args.AddOption(&num_active, "-nv", "--num-vars",
                  "How many variables to evolve in time: 1 = T only, "
                  "2 = T and omega, 3 = T, omega, and n (full system).");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Save ParaView output every N steps.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(std::cout); }

   // Configure the Device AFTER parsing so -d cuda/hip can be honoured.
   // Done before any FE space / vector allocation so memory classes are set
   // correctly on construction.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   Ricci2DNonlinear ricci(MPI_COMM_WORLD, order, ser_ref_levels, xL, yL,
                          num_active);
   ricci.Setup();

   // Initial save.
   ricci.syncGridFuncsFromBlocks(*ricci.StateBlocks());
   ricci.updateDataCollection(0, 0.0);
   ricci.Save();

   if (myid == 0)
   {
      std::cout << "Number of true H1 unknowns per variable: "
                << ricci.H1FES()->GlobalTrueVSize() << std::endl;
      std::cout << "Total true unknowns: "
                << NUM_VARS * ricci.H1FES()->GlobalTrueVSize() << std::endl;
      std::cout << "Active variables: " << num_active
                << " (T" << (num_active >= 2 ? ", omega" : "")
                       << (num_active >= 3 ? ", n"     : "") << ")"
                << std::endl;
   }

   RicciTimeOperator time_op(MPI_COMM_WORLD, ricci,
                             newton_rtol, newton_max_iter,
                             lin_rtol, lin_max_iter, kdim);
   if (enable_ew) { time_op.EnableEisenstatWalker(ew_rtol0, ew_rtol_max); }

   std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);
   ode_solver->Init(time_op);

   real_t t = 0.0;
   int step = 0;
   while (t < t_final)
   {
      real_t dt_step = std::min(dt, t_final - t);
      ode_solver->Step(*ricci.StateBlocks(), t, dt_step);
      ++step;

      // Mirror the new state into the per-variable ParGridFunctions for I/O.
      ricci.syncGridFuncsFromBlocks(*ricci.StateBlocks());

      if (myid == 0)
      {
         std::cout << "step " << step
                   << "  t = " << t
                   << "  dt = " << dt_step
                   << "  ||T||_2 = " << ricci.StateBlocks()->GetBlock(T_IDX).Norml2()
                   << "  ||w||_2 = " << ricci.StateBlocks()->GetBlock(OMEGA_IDX).Norml2()
                   << "  ||n||_2 = " << ricci.StateBlocks()->GetBlock(N_IDX).Norml2()
                   << "  ||phi||_2 = " << ricci.Phi()->Norml2()
                   << std::endl;
      }

      if (step % vis_steps == 0)
      {
         ricci.updateDataCollection(step, t);
         ricci.Save();
      }
   }

   return 0;
}
