#include "ex42p.hpp"

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

Ricci2DNonlinear::Ricci2DNonlinear(MPI_Comm comm, int order, int ref_levels,
                                   real_t xL, real_t yL, int num_active,
                                   const std::string &save_dir)
   : _xL(xL), _yL(yL), _order(order), _ref_levels(ref_levels),
     _num_active(num_active),
     _rotmat({{0.0, -1.0}, {1.0, 0.0}}),
     _comm(comm),
     _save_dir(save_dir)
{
   MFEM_VERIFY(num_active >= 1 && num_active <= NUM_VARS,
               "Ricci2DNonlinear: num_active must be 1, 2, or 3.");
   MFEM_VERIFY(!save_dir.empty(),
               "Ricci2DNonlinear: save_dir must be non-empty.");
}

void Ricci2DNonlinear::Setup(const std::string &restart_dir)
{
   _restart_dir = restart_dir;

   buildMesh();
   buildSpaces();
   buildState();
   buildCoefficients();
   buildQuadratureSamples();
   buildMassMatrix();

   if (_restart_dir.empty())
      setInitialConditions();
   else
      loadFieldsFromCheckpoint();

   syncBlocksFromGridFuncs(*_var_blocks);

   _Sn->Project(*_Sn_qf);

   projectPhiToQuadrature();

   buildNonlinearForm();
   setOutput();
   buildPhiSolver();
   buildKForm();
   reassembleK();

   _restart_dir.clear();
}

void Ricci2DNonlinear::buildMesh()
{
   if (_restart_dir.empty())
   {
      Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true,
                                        _xL, _yL, false);
      _pmesh = std::make_unique<ParMesh>(_comm, mesh);
      for (int l = 0; l < _ref_levels; ++l) { _pmesh->UniformRefinement(); }
   }
   else
   {
      // Load the partitioned mesh from previous run
      const int myid = Mpi::WorldRank();
      const std::string fname =
         MakeParFilename(_restart_dir + "/mesh.", myid);
      std::ifstream ifs(fname);
      MFEM_VERIFY(ifs.good(),
                  "Ricci2DNonlinear: cannot open checkpoint mesh '" << fname
                  << "'. Was the previous run launched with the same number "
                  "of MPI ranks?");
      _pmesh = std::make_unique<ParMesh>(_comm, ifs);
   }
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
      _vars[i] = std::make_unique<ParGridFunction>(_h1_fes.get());

   _phi = std::make_unique<ParGridFunction>(_h1_fes.get());

   _block_trueOffsets.SetSize(NUM_VARS + 1);
   _block_trueOffsets[0] = 0;

   for (int i = 0; i < NUM_VARS; ++i)
      _block_trueOffsets[i + 1] = _h1_fes->GetTrueVSize();
   
   _block_trueOffsets.PartialSum();

   _var_blocks = std::make_unique<BlockVector>(_block_trueOffsets);
}

void Ricci2DNonlinear::buildCoefficients()
{
   // Radial distance from mesh centre
   _r.reset(new TransformedCoefficient(
               new CartesianXCoefficient, new CartesianYCoefficient,
               [this](real_t x, real_t y) {
      return std::sqrt(std::pow(x - _xL/2.0, 2) + std::pow(y - _yL/2.0, 2));
   }));

   // Source term
   const real_t rs = 20.0;
   const real_t Ls = 0.5;
   _Sn.reset(new TransformedCoefficient(
                _r.get(), [this, rs, Ls](real_t r) {
      return 0.5 * _S_0n * (1.0 - std::tanh((r - rs) / Ls));
   }));

   _grad_rotate = std::make_unique<MatrixConstantCoefficient>(_rotmat);
   _grad_phi = std::make_unique<GradientGridFunctionCoefficient>(_phi.get());
   _vd = std::make_unique<MatrixVectorProductCoefficient>(*_grad_rotate, *_grad_phi);

   // Drift velocity:
   // v_E = (1/B) * v_d
   _v_E = std::make_unique<ScalarVectorProductCoefficient>(_Binv, *_vd);

   // SUW term:
   // 0.5 * h * (v_E · grad u)(v_E · grad v) / sqrt(|v_E|^2 + eps^2)
   _v_E_sq = std::make_unique<InnerProductCoefficient>(*_v_E, *_v_E);
   _suw_scalar = std::make_unique<TransformedCoefficient>(
                    _v_E_sq.get(),
                    [h = _h, eps2 = _eps2](real_t v_sq)
   {
      return h / (2.0 * std::sqrt(v_sq + eps2));
   });
   _v_E_outer = std::make_unique<OuterProductCoefficient>(*_v_E, *_v_E);
   _SUW_matcoef = std::make_unique<ScalarMatrixProductCoefficient>(
                     *_suw_scalar, *_v_E_outer);
}

void Ricci2DNonlinear::buildQuadratureSamples()
{
   // This is to accelerate coefficient evaluations of Sn and phi
   const int qorder = 2 * _order + 3;
   _qspace = std::make_unique<QuadratureSpace>(_pmesh.get(), qorder);
   _phi_qf = std::make_unique<QuadratureFunction>(*_qspace);
   _Sn_qf = std::make_unique<QuadratureFunction>(*_qspace);
   _phi_qfc = std::make_unique<QuadratureFunctionCoefficient>(*_phi_qf);
   _Sn_qfc = std::make_unique<QuadratureFunctionCoefficient>(*_Sn_qf);
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
   // Poisson + SUW terms
   // K = -(1/B) (v_d · grad u, v) + (c v_E ⊗ v_E : grad u ⊗ grad v).
   // The integrators reference the live coefficients owned by Ricci2DNonlinear
   _K_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _K_form->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv));
   _K_form->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef));
}

void Ricci2DNonlinear::buildPhiSolver()
{
   Array<int> ess_bdr(_pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   _h1_fes->GetEssentialTrueDofs(ess_bdr, _ess_tdof_list_phi);

   _a_phi = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _a_phi->AddDomainIntegrator(new DiffusionIntegrator());
   _a_phi->Assemble();
   _a_phi->Finalize();

   _omega_coef = std::make_unique<GridFunctionCoefficient>(
                        _vars[OMEGA_IDX].get());
   _neg_omega_coef = std::make_unique<ProductCoefficient>(-1.0, *_omega_coef);
   _b_phi = std::make_unique<ParLinearForm>(_h1_fes.get());
   _b_phi->AddDomainIntegrator(new DomainLFIntegrator(*_neg_omega_coef));

   _a_phi->FormSystemMatrix(_ess_tdof_list_phi, _A_phi);

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

   for (int i = 0; i < NUM_VARS; ++i) 
      fes[i] = _h1_fes.get();

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

void Ricci2DNonlinear::setOutput()
{
   _dc = std::make_unique<ParaViewDataCollection>(_save_dir + "/Step",
                                                  _pmesh.get());
   _dc->RegisterField("phi", _phi.get());
   _dc->RegisterField("T", _vars[T_IDX].get());
   _dc->RegisterField("omega", _vars[OMEGA_IDX].get());
   _dc->RegisterField("n", _vars[N_IDX].get());
   _dc->UseRestartMode(true);
}

namespace
{
// Variable name in the per-rank checkpoint file for each block index.
const char *checkpoint_var_name(int idx)
{
   switch (idx)
   {
      case T_IDX:     return "T";
      case OMEGA_IDX: return "omega";
      case N_IDX:     return "n";
      default:        MFEM_ABORT("unknown VarIdx"); return "";
   }
}
} // namespace

void Ricci2DNonlinear::loadFieldsFromCheckpoint()
{
   const int myid = Mpi::WorldRank();

   for (int i = 0; i < NUM_VARS; ++i)
   {
      const std::string fname = MakeParFilename(
         _restart_dir + "/" + checkpoint_var_name(i) + ".", myid);
      std::ifstream ifs(fname);
      MFEM_VERIFY(ifs.good(),
                  "Ricci2DNonlinear: cannot open checkpoint field '"
                  << fname << "'.");
      ParGridFunction tmp(_pmesh.get(), ifs);
      MFEM_VERIFY(tmp.Size() == _vars[i]->Size(),
                  "Ricci2DNonlinear: size mismatch loading '" << fname
                  << "' (got " << tmp.Size() << ", expected "
                  << _vars[i]->Size() << ").");
      *_vars[i] = tmp;
   }

   const std::string phi_fname =
      MakeParFilename(_restart_dir + "/phi.", myid);
   std::ifstream ifs_phi(phi_fname);
   MFEM_VERIFY(ifs_phi.good(),
               "Ricci2DNonlinear: cannot open checkpoint field '"
               << phi_fname << "'.");
   ParGridFunction tmp_phi(_pmesh.get(), ifs_phi);
   MFEM_VERIFY(tmp_phi.Size() == _phi->Size(),
               "Ricci2DNonlinear: size mismatch loading phi.");
   *_phi = tmp_phi;
}

void Ricci2DNonlinear::SaveCheckpoint(const std::string &dir, real_t t,
                                      int step) const
{
   const int myid = Mpi::WorldRank();

   // Rank 0 creates the leaf directory (its parent <save_dir>/ was already
   // created recursively by ParaViewDataCollection::Save just before us) and
   // writes meta.txt.  create_directory returns false if the directory
   // already exists (which is fine in single-slot mode); it only sets ec on
   // hard filesystem errors.
   if (myid == 0)
   {
      std::error_code ec;
      std::filesystem::create_directory(dir, ec);
      MFEM_VERIFY(!ec,
                  "Ricci2DNonlinear::SaveCheckpoint: cannot create '"
                  << dir << "': " << ec.message() << ".");

      std::ofstream meta(dir + "/meta.txt", std::ios::trunc);
      MFEM_VERIFY(meta.good(),
                  "Ricci2DNonlinear::SaveCheckpoint: cannot open meta.txt "
                  "for writing.");
      meta.precision(17);
      meta << "t " << t << "\n";
      meta << "step " << step << "\n";
   }
   // Wait until the directory exists before any other rank writes into it.
   MPI_Barrier(_comm);

   // Per-rank mesh.
   {
      std::ofstream ofs(MakeParFilename(dir + "/mesh.", myid));
      MFEM_VERIFY(ofs.good(),
                  "Ricci2DNonlinear::SaveCheckpoint: cannot open mesh file "
                  "for writing.");
      ofs.precision(17);
      _pmesh->ParPrint(ofs);
   }

   // Per-rank fields.
   auto save_gf = [&](const std::string &name, const ParGridFunction &gf)
   {
      std::ofstream ofs(MakeParFilename(dir + "/" + name + ".", myid));
      MFEM_VERIFY(ofs.good(),
                  "Ricci2DNonlinear::SaveCheckpoint: cannot open field '"
                  << name << "' for writing.");
      ofs.precision(17);
      gf.Save(ofs);
   };
   for (int i = 0; i < NUM_VARS; ++i)
   {
      save_gf(checkpoint_var_name(i), *_vars[i]);
   }
   save_gf("phi", *_phi);

   // Make sure rank 0's meta.txt is visible to any spawned reader.
   MPI_Barrier(_comm);
}

void Ricci2DNonlinear::LoadCheckpointMeta(const std::string &dir,
                                          MPI_Comm comm,
                                          real_t &t, int &step)
{
   int myid;
   MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      std::ifstream ifs(dir + "/meta.txt");
      MFEM_VERIFY(ifs.good(),
                  "LoadCheckpointMeta: cannot open '" << dir
                  << "/meta.txt'.");
      std::string key;
      ifs >> key >> t;
      MFEM_VERIFY(key == "t", "LoadCheckpointMeta: expected 't' key.");
      ifs >> key >> step;
      MFEM_VERIFY(key == "step", "LoadCheckpointMeta: expected 'step' key.");
   }
   MPI_Bcast(&t,    1, MPITypeMap<real_t>::mpi_type, 0, comm);
   MPI_Bcast(&step, 1, MPI_INT,                      0, comm);
}

void Ricci2DNonlinear::syncGridFuncsFromBlocks(const BlockVector &u_blk)
{
   for (int i = 0; i < NUM_VARS; ++i)
      _vars[i]->SetFromTrueDofs(u_blk.GetBlock(i));
}

void Ricci2DNonlinear::syncBlocksFromGridFuncs(BlockVector &u_blk) const
{
   for (int i = 0; i < NUM_VARS; ++i)
      _vars[i]->ParallelProject(u_blk.GetBlock(i));
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

void Ricci2DNonlinear::reassembleK()
{
   // Refresh the K matrix with the new V_d, V_E, phi values
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

   HypreParMatrix *M = _ricci.MassMat();
   HypreParMatrix *K = _ricci.KMat();
   _M_plus_gK.reset(Add(1.0, *M, _gamma, *K));
}

void RicciImplicitStageOp::Mult(const Vector &k_vec, Vector &R_vec) const
{
   MFEM_VERIFY(_u_pred != nullptr,
               "RicciImplicitStageOp: SetParameters() not called before Mult().");

   add(*_u_pred, _gamma, k_vec, _z);

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

namespace
{
// Backtracking Armijo line search on top of MFEM's NewtonSolver. Without 
// this, a Newton step in the stiff/swinging-phi regime can move the iterate
// to a state where the reaction exp(Lambda - phi/T) is many orders of 
// magnitude away from the previous iterate, causing the residual to grow
class BacktrackingNewtonSolver : public NewtonSolver
{
public:
#ifdef MFEM_USE_MPI
   explicit BacktrackingNewtonSolver(MPI_Comm comm) : NewtonSolver(comm) {}
#endif

   // Armijo backtrack: find alpha in (0, 1] such that
   //     ||F(x - alpha c) - b|| <= (1 - sigma * alpha) ||F(x) - b||
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
      // Couldn't satisfy Armijo; return the smallest tried
      return alpha;
   }
};

// Block Krylov solver consumed by NewtonSolver as its linear solver.
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

      for (int s = 0; s < NUM_VARS; ++s)
         _amgs[s]->SetOperator(
            dynamic_cast<const HypreParMatrix&>(bop->GetBlock(s, s)));

      if (!_bdp)
      {
         _bdp = std::make_unique<BlockDiagonalPreconditioner>(bop->RowOffsets());

         for (int s = 0; s < NUM_VARS; ++s)
            _bdp->SetDiagonalBlock(s, _amgs[s].get());

         _gmres.SetPreconditioner(*_bdp);
      }

      _gmres.SetOperator(*bop);
      height = bop->Height();
      width  = bop->Width();
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      // Propagate adaptive rtol set by NewtonSolver since the last call. 
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

RicciTimeOperator::RicciTimeOperator(MPI_Comm comm, Ricci2DNonlinear & ricci,
                                     real_t newton_rtol, int newton_max_iter,
                                     real_t lin_rtol, int lin_max_iter,
                                     int kdim)
   : TimeDependentOperator(ricci.BlockTrueOffsets().Last(),
                           0.0,
                           TimeDependentOperator::IMPLICIT),
     _ricci(ricci),
     _stage_op(ricci),
     _newton(new BacktrackingNewtonSolver(comm)),
     _lin_solver(new BlockNewtonLinearSolver(comm, lin_rtol,
                                             lin_max_iter, kdim))
{
   _newton->iterative_mode = false;
   _newton->SetSolver(*_lin_solver);
   _newton->SetOperator(_stage_op);
   _newton->SetRelTol(newton_rtol);
   _newton->SetAbsTol(0.0);
   _newton->SetMaxIter(newton_max_iter);
   _newton->SetPrintLevel(1);
}

void RicciTimeOperator::EnableEisenstatWalker(real_t rtol0, real_t rtol_max)
{
   _newton->SetAdaptiveLinRtol(/*type=*/2, rtol0, rtol_max);
}

void RicciTimeOperator::ImplicitSolve(real_t gamma, const Vector &u_pred,
                                      Vector &k)
{
   BlockVector u_pred_blk(const_cast<Vector&>(u_pred),
                          _ricci.BlockTrueOffsets());

   _ricci.pullOmegaFromBlocks(u_pred_blk);
   _ricci.updatePhi();
   _ricci.reassembleK();
   _ricci.projectPhiToQuadrature();

   // Set up M + γK
   _stage_op.SetParameters(gamma, u_pred_blk);

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

   // Options
   const char *device_config = "cpu";
   int    order             = 1;
   int    ref_levels        = 0;
   int    ode_solver_type   = 21; // 21 = Backward Euler; 23 = SDIRK23
   real_t xL                = 100.0;
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
   std::string save_dir = "Ricci2DNonlinear";

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure() "
                  "(e.g. 'cpu', 'cuda', 'hip').");
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the H1 finite element space.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform parallel refinement levels applied to "
                  "the 64x64 base mesh.");
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
   args.AddOption(&save_dir, "-sd", "--save-directory",
                  "Top-level output directory.  ParaView frames are written "
                  "to <save_dir>/Step/, single-slot checkpoint state to "
                  "<save_dir>/Checkpoint/.  If the checkpoint directory "
                  "already contains a meta.txt at startup, the run resumes "
                  "from it; otherwise it starts fresh and writes new "
                  "checkpoints there at every visualisation step.  Restart "
                  "requires the same MPI rank count as the previous run.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(std::cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(std::cout); }

   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // The checkpoint directory is a fixed subfolder of the save directory.
   const std::string cdir = save_dir + "/Checkpoint";
   real_t t    = 0.0;
   int    step = 0;
   bool   restart = false;
   {
      // Check if there is a checkpoint metadata file
      restart = std::filesystem::exists(cdir + "/meta.txt");
      if (restart)
      {
         Ricci2DNonlinear::LoadCheckpointMeta(cdir, MPI_COMM_WORLD, t, step);
         if (myid == 0)
         {
            std::cout << "Restarting from checkpoint '" << cdir
                      << "' at t = " << t << ", step = " << step << "."
                      << std::endl;
         }
      }
      else if (myid == 0)
      {
         std::cout << "No checkpoint at '" << cdir
                   << "': starting fresh.  ParaView output -> '"
                   << save_dir << "/Step', checkpoints -> '" << cdir
                   << "'." << std::endl;
      }
   }

   Ricci2DNonlinear ricci(MPI_COMM_WORLD, order, ref_levels, xL, yL,
                          num_active, save_dir);
   ricci.Setup(restart ? cdir : "");

   ricci.syncGridFuncsFromBlocks(*ricci.StateBlocks());
   if (!restart)
   {
      ricci.updateDataCollection(0, 0.0);
      ricci.Save();
   }

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

   const real_t t_tol = 1e-12 * std::max(t_final, dt);
   while (t < t_final - t_tol)
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
         ricci.SaveCheckpoint(cdir, t, step);
      }
   }

   return 0;
}
