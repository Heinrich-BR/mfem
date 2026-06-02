#include "ex43p.hpp"

RogersRicciReactionOp::RogersRicciReactionOp(
   ParFiniteElementSpace &fes,
   const QuadratureSpace &qspace,
   const IntegrationRule &ir,
   const Array<int> &block_offsets,
   const QuadratureFunction &phi_qf,
   const QuadratureFunction &Sn_qf,
   int num_active,
   AssemblyLevel asm_level,
   real_t Lambda,
   real_t eps2)
   : Operator(block_offsets.Last()),
     _fes(fes),
     _qspace(qspace),
     _ir(ir),
     _block_offsets(block_offsets),
     _phi_qf(&phi_qf),
     _Sn_qf(&Sn_qf),
     _num_active(num_active),
     _Lambda(Lambda),
     _eps2(eps2),
     _er(fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC)),
     _qi(fes.GetQuadratureInterpolator(ir)),
     _T_qf(const_cast<QuadratureSpace&>(qspace)),
     _n_qf(const_cast<QuadratureSpace&>(qspace)),
     _F_T_qf(const_cast<QuadratureSpace&>(qspace)),
     _F_omega_qf(const_cast<QuadratureSpace&>(qspace)),
     _F_n_qf(const_cast<QuadratureSpace&>(qspace)),
     _F_T_qfc(_F_T_qf),
     _F_omega_qfc(_F_omega_qf),
     _F_n_qfc(_F_n_qf),
     _dFT_dT_qf(const_cast<QuadratureSpace&>(qspace)),
     _dFw_dT_qf(const_cast<QuadratureSpace&>(qspace)),
     _dFn_dT_qf(const_cast<QuadratureSpace&>(qspace)),
     _dFn_dn_qf(const_cast<QuadratureSpace&>(qspace)),
     _dFT_dT_qfc(_dFT_dT_qf),
     _dFw_dT_qfc(_dFw_dT_qf),
     _dFn_dT_qfc(_dFn_dT_qf),
     _dFn_dn_qfc(_dFn_dn_qf),
     _F_block(block_offsets)
{

   // Scratch L- and E-vectors
   _l_vec_T.SetSize(fes.GetVSize());     _l_vec_T.UseDevice(true);
   _l_vec_n.SetSize(fes.GetVSize());     _l_vec_n.UseDevice(true);
   _e_vec_T.SetSize(_er->Height());      _e_vec_T.UseDevice(true);
   _e_vec_n.SetSize(_er->Height());      _e_vec_n.UseDevice(true);

   // One LF per variable
   auto make_lf = [&](QuadratureFunctionCoefficient &qfc)
   {
      auto lf = std::make_unique<ParLinearForm>(&fes);
      auto *integ = new DomainLFIntegrator(qfc);
      integ->SetIntRule(&ir);
      lf->AddDomainIntegrator(integ);
      // Device-resident assembly of the QuadratureFunctionCoefficient residual
      lf->UseFastAssembly(true);
      return lf;
   };
   _lf_T     = make_lf(_F_T_qfc);
   _lf_omega = make_lf(_F_omega_qfc);
   _lf_n     = make_lf(_F_n_qfc);

   // One ParBilinearForm per non-zero F' block
   auto make_bf = [&](QuadratureFunctionCoefficient &qfc)
   {
      auto bf = std::make_unique<ParBilinearForm>(&fes);
      auto *integ = new MassIntegrator(qfc);
      integ->SetIntRule(&ir);
      bf->AddDomainIntegrator(integ);
      bf->SetAssemblyLevel(asm_level);
      return bf;
   };
   _bf_TT = make_bf(_dFT_dT_qfc);
   _bf_wT = make_bf(_dFw_dT_qfc);
   _bf_nT = make_bf(_dFn_dT_qfc);
   _bf_nn = make_bf(_dFn_dn_qfc);
}

void RogersRicciReactionOp::Mult(const Vector &z_true, Vector &R_true) const
{
   R_true.UseDevice(true);

   // Sample z at QPs: T -> L -> E -> Q
   sampleStateToQPs(z_true);
   computeReactionResidualAtQPs();

   _lf_T->Assemble();
   if (_num_active >= 2) { _lf_omega->Assemble(); }
   if (_num_active >= 3) { _lf_n->Assemble(); }

   BlockVector R_blk(R_true, _block_offsets);
   R_blk.UseDevice(true);
   for (int s = 0; s < NUM_VARS; ++s) { R_blk.GetBlock(s).UseDevice(true); }

   R_blk.SyncToBlocks();

   _lf_T->ParallelAssemble(R_blk.GetBlock(T_IDX));

   if (_num_active >= 2)
      _lf_omega->ParallelAssemble(R_blk.GetBlock(OMEGA_IDX));
   else
      R_blk.GetBlock(OMEGA_IDX) = 0.0;

   if (_num_active >= 3)
      _lf_n->ParallelAssemble(R_blk.GetBlock(N_IDX));
   else
      R_blk.GetBlock(N_IDX) = 0.0;

   R_blk.SyncFromBlocks();
}

BlockOperator &RogersRicciReactionOp::GetGradient(const Vector &z_true) const
{
   
   sampleStateToQPs(z_true);
   computeReactionJacobianAtQPs();

   Array<int> empty_ess;
   auto rebuild = [&](ParBilinearForm &bf, OperatorHandle &out)
   {
      bf.Update();
      bf.Assemble();
      bf.FormSystemMatrix(empty_ess, out);
   };

   rebuild(*_bf_TT, _J_TT);
   _F_block.SetBlock(T_IDX, T_IDX, _J_TT.Ptr());

   if (_num_active >= 2)
   {
      rebuild(*_bf_wT, _J_wT);
      _F_block.SetBlock(OMEGA_IDX, T_IDX, _J_wT.Ptr());
   }
   if (_num_active >= 3)
   {
      rebuild(*_bf_nT, _J_nT);
      _F_block.SetBlock(N_IDX, T_IDX, _J_nT.Ptr());
      rebuild(*_bf_nn, _J_nn);
      _F_block.SetBlock(N_IDX, N_IDX, _J_nn.Ptr());
   }

   return _F_block;
}

void RogersRicciReactionOp::sampleStateToQPs(const Vector &z_true) const
{
   const_cast<Vector&>(z_true).UseDevice(true);
   BlockVector z_blk(const_cast<Vector&>(z_true), _block_offsets);
   z_blk.UseDevice(true);
   for (int s = 0; s < NUM_VARS; ++s) { z_blk.GetBlock(s).UseDevice(true); }
   z_blk.SyncToBlocks();

   const Operator *P = _fes.GetProlongationMatrix();

   // T branch
   P->Mult(z_blk.GetBlock(T_IDX), _l_vec_T);
   _er->Mult(_l_vec_T, _e_vec_T);
   _qi->Values(_e_vec_T, _T_qf);

   // n branch
   P->Mult(z_blk.GetBlock(N_IDX), _l_vec_n);
   _er->Mult(_l_vec_n, _e_vec_n);
   _qi->Values(_e_vec_n, _n_qf);
}

void RogersRicciReactionOp::computeReactionResidualAtQPs() const
{
   const int nq = _qspace.GetSize();

   const real_t *T_ptr   = _T_qf.Read();
   const real_t *n_ptr   = _n_qf.Read();
   const real_t *phi_ptr = _phi_qf->Read();
   const real_t *Sn_ptr  = _Sn_qf->Read();

   real_t *FT_ptr = _F_T_qf.Write();
   real_t *Fw_ptr = _F_omega_qf.Write();
   real_t *Fn_ptr = _F_n_qf.Write();

   const real_t Lambda = _Lambda;
   const real_t eps2   = _eps2;

   mfem::forall(nq, [=] MFEM_HOST_DEVICE (int q)
   {
      const real_t T   = T_ptr[q];
      const real_t n   = n_ptr[q];
      const real_t phi = phi_ptr[q];
      const real_t Sn  = Sn_ptr[q];

      const real_t Treg = sqrt(T*T + eps2);
      const real_t A    = Lambda - phi / Treg;
      const real_t eA   = exp(A);

      FT_ptr[q] = (T / 36.0) * (1.71 * eA - 0.71) - Sn;
      Fw_ptr[q] = (eA - 1.0) / 24.0;
      Fn_ptr[q] = (n / 24.0) * eA - Sn;
   });
}

void RogersRicciReactionOp::computeReactionJacobianAtQPs() const
{
   const int nq = _qspace.GetSize();

   const real_t *T_ptr   = _T_qf.Read();
   const real_t *n_ptr   = _n_qf.Read();
   const real_t *phi_ptr = _phi_qf->Read();

   real_t *dFT_dT_ptr = _dFT_dT_qf.Write();
   real_t *dFw_dT_ptr = _dFw_dT_qf.Write();
   real_t *dFn_dT_ptr = _dFn_dT_qf.Write();
   real_t *dFn_dn_ptr = _dFn_dn_qf.Write();

   const real_t Lambda = _Lambda;
   const real_t eps2   = _eps2;

   mfem::forall(nq, [=] MFEM_HOST_DEVICE (int q)
   {
      const real_t T   = T_ptr[q];
      const real_t n   = n_ptr[q];
      const real_t phi = phi_ptr[q];

      const real_t Treg  = sqrt(T*T + eps2);
      const real_t A     = Lambda - phi / Treg;
      const real_t eA    = exp(A);

      // dA/dT = phi*T/Treg^3 (chain rule on Treg).
      const real_t Treg3 = Treg * Treg * Treg;
      const real_t deA_dT = eA * phi * T / Treg3;

      dFT_dT_ptr[q] = (1.71 * eA - 0.71) / 36.0
                      + (T * 1.71 / 36.0) * deA_dT;
      dFw_dT_ptr[q] = deA_dT / 24.0;
      dFn_dT_ptr[q] = n * deA_dT / 24.0;
      dFn_dn_ptr[q] = eA / 24.0;
   });
}

Ricci2DNonlinear::Ricci2DNonlinear(MPI_Comm comm, int order, int ref_levels,
                                   real_t xL, real_t yL, int num_active,
                                   bool matrix_free, bool phi_lor,
                                   const std::string &save_dir)
   : _xL(xL), _yL(yL), _order(order), _ref_levels(ref_levels),
     _num_active(num_active),
     _matrix_free(matrix_free),
     _phi_lor(phi_lor),
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
   {
      _vars[i] = std::make_unique<ParGridFunction>(_h1_fes.get());
      _vars[i]->UseDevice(true);
   }

   _phi = std::make_unique<ParGridFunction>(_h1_fes.get());
   _phi->UseDevice(true);

   _block_trueOffsets.SetSize(NUM_VARS + 1);
   _block_trueOffsets[0] = 0;

   for (int i = 0; i < NUM_VARS; ++i)
      _block_trueOffsets[i + 1] = _h1_fes->GetTrueVSize();

   _block_trueOffsets.PartialSum();

   _var_blocks = std::make_unique<BlockVector>(_block_trueOffsets);
   _var_blocks->UseDevice(true);
   for (int i = 0; i < NUM_VARS; ++i)
      _var_blocks->GetBlock(i).UseDevice(true);
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

   // Cache the FES-owned device-friendly pipeline for projectPhiToQuadrature.
   _er_h1 = _h1_fes->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
   _qi_h1 = _h1_fes->GetQuadratureInterpolator(_qspace->GetIntRule(0));
   _phi_e_vec.SetSize(_er_h1->Height());
   _phi_e_vec.UseDevice(true);
}

void Ricci2DNonlinear::projectPhiToQuadrature()
{
   _er_h1->Mult(*_phi, _phi_e_vec);
   _qi_h1->Values(_phi_e_vec, *_phi_qf);
}

void Ricci2DNonlinear::buildMassMatrix()
{
   _M_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _M_form->AddDomainIntegrator(new MassIntegrator);

   if (_matrix_free)
   {
      _M_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      _M_form->Assemble();
      Array<int> empty_ess;
      _M_form->FormSystemMatrix(empty_ess, _M_op);
      _M_diag.SetSize(_h1_fes->GetTrueVSize());
      _M_diag.UseDevice(true);
      _M_form->AssembleDiagonal(_M_diag);
   }
   else
   {
      _M_form->Assemble();
      _M_form->Finalize();
      _M_mat.reset(_M_form->ParallelAssemble());
   }
}

void Ricci2DNonlinear::buildKForm()
{
   // Poisson + SUW terms
   // K = -(1/B) (v_d · grad u, v) + (c v_E ⊗ v_E : grad u ⊗ grad v).
   // The integrators reference the live coefficients owned by Ricci2DNonlinear
   _K_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _K_form->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv));
   _K_form->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef));
   if (_matrix_free)
   {
      _K_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      // Companion form (SUW diffusion only) for the Jacobi diagonal of K.
      _K_diff_form = std::make_unique<ParBilinearForm>(_h1_fes.get());
      _K_diff_form->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef));
      _K_diff_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
}

void Ricci2DNonlinear::buildPhiSolver()
{
   Array<int> ess_bdr(_pmesh->bdr_attributes.Max());
   ess_bdr = 1;
   _h1_fes->GetEssentialTrueDofs(ess_bdr, _ess_tdof_list_phi);

   _a_phi = std::make_unique<ParBilinearForm>(_h1_fes.get());
   _a_phi->AddDomainIntegrator(new DiffusionIntegrator());
   if (_phi_lor)
   {
      _a_phi->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      _a_phi->Assemble();
   }
   else
   {
      _a_phi->Assemble();
      _a_phi->Finalize();
   }

   _omega_coef = std::make_unique<GridFunctionCoefficient>(
                        _vars[OMEGA_IDX].get());
   _neg_omega_coef = std::make_unique<ProductCoefficient>(-1.0, *_omega_coef);
   _b_phi = std::make_unique<ParLinearForm>(_h1_fes.get());
   _b_phi->AddDomainIntegrator(new DomainLFIntegrator(*_neg_omega_coef));
   _b_phi->UseFastAssembly(true);

   _a_phi->FormSystemMatrix(_ess_tdof_list_phi, _A_phi);

   if (_phi_lor)
   {
      _lor_disc = std::make_unique<ParLORDiscretization>(
                     *_a_phi, _ess_tdof_list_phi);
      _amg_phi = std::make_unique<HypreBoomerAMG>(
                     _lor_disc->GetAssembledMatrix());
   }
   else
   {
      _amg_phi = std::make_unique<HypreBoomerAMG>();
      _amg_phi->SetOperator(*_A_phi);
   }
   _amg_phi->SetPrintLevel(0);
   Solver *phi_prec = _amg_phi.get();

   _cg_phi = std::make_unique<CGSolver>(_comm);
   _cg_phi->SetRelTol(1e-12);
   _cg_phi->SetMaxIter(2000);
   _cg_phi->SetPrintLevel(0);
   _cg_phi->SetOperator(*_A_phi);
   _cg_phi->SetPreconditioner(*phi_prec);
}

void Ricci2DNonlinear::buildNonlinearForm()
{
   _reaction_op = std::make_unique<RogersRicciReactionOp>(
      *_h1_fes, *_qspace, _qspace->GetIntRule(0),
      _block_trueOffsets, *_phi_qf, *_Sn_qf,
      _num_active,
      _matrix_free ? AssemblyLevel::PARTIAL : AssemblyLevel::LEGACY,
      _Lambda, _eps2);
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
   u_blk.SyncToBlocks();
   for (int i = 0; i < NUM_VARS; ++i)
      _vars[i]->SetFromTrueDofs(u_blk.GetBlock(i));
}

void Ricci2DNonlinear::syncBlocksFromGridFuncs(BlockVector &u_blk) const
{
   for (int i = 0; i < NUM_VARS; ++i)
      _vars[i]->ParallelProject(u_blk.GetBlock(i));
   u_blk.SyncFromBlocks();
}

void Ricci2DNonlinear::pullOmegaFromBlocks(const BlockVector &u_blk)
{
   u_blk.SyncToBlocks();
   _vars[OMEGA_IDX]->SetFromTrueDofs(u_blk.GetBlock(OMEGA_IDX));
}

void Ricci2DNonlinear::updatePhi()
{
   _b_phi->Assemble();

   OperatorPtr A;
   Vector X, B;
   _a_phi->FormLinearSystem(_ess_tdof_list_phi, *_phi, *_b_phi, A, X, B);

   _cg_phi->Mult(B, X);
   _a_phi->RecoverFEMSolution(X, *_b_phi, *_phi);
}

void Ricci2DNonlinear::reassembleK()
{
   // Refresh K with the new V_d, V_E, phi values.
   _K_form->Update();
   _K_form->Assemble();

   if (_matrix_free)
   {
      Array<int> empty_ess;
      _K_form->FormSystemMatrix(empty_ess, _K_op);
      _K_diff_form->Update();
      _K_diff_form->Assemble();
      _K_diag.SetSize(_h1_fes->GetTrueVSize());
      _K_diag.UseDevice(true);
      _K_diff_form->AssembleDiagonal(_K_diag);
   }
   else
   {
      _K_form->Finalize();
      _K_mat.reset(_K_form->ParallelAssemble());
   }
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
   _z.UseDevice(true);
   _tmp_block.UseDevice(true);
}

void RicciImplicitStageOp::SetParameters(real_t gamma, const BlockVector &u_pred)
{
   _gamma = gamma;
   _u_pred = &u_pred;

   if (_ricci.MatrixFree())
   {
      // Matrix-free M + γK
      _M_plus_gK_mf = std::make_unique<SumOperator>(
         _ricci.MassOp(), 1.0, _ricci.KOp(), _gamma,
         /*ownA=*/false, /*ownB=*/false);
   }
   else
   {
      HypreParMatrix *M = _ricci.MassMatHypre();
      HypreParMatrix *K = _ricci.KMatHypre();
      _M_plus_gK_hypre.reset(Add(1.0, *M, _gamma, *K));
   }
}

void RicciImplicitStageOp::Mult(const Vector &k_vec, Vector &R_vec) const
{
   MFEM_VERIFY(_u_pred != nullptr,
               "RicciImplicitStageOp: SetParameters() not called before Mult().");

   const_cast<Vector&>(k_vec).UseDevice(true);
   R_vec.UseDevice(true);

   // z = u_pred + gamma * k
   add(*_u_pred, _gamma, k_vec, _z);

   _z.SyncToBlocks();

   // R = F(z)
   _ricci.ReactionOp()->Mult(_z, R_vec);

   // R += M·k + K·z
   const Array<int> &offs = _ricci.BlockTrueOffsets();
   BlockVector k_blk(const_cast<Vector&>(k_vec), offs);
   BlockVector R_blk(R_vec, offs);
   k_blk.UseDevice(true);
   R_blk.UseDevice(true);
   for (int s = 0; s < NUM_VARS; ++s)
   {
      k_blk.GetBlock(s).UseDevice(true);
      R_blk.GetBlock(s).UseDevice(true);
   }
   k_blk.SyncToBlocks();
   R_blk.SyncToBlocks();

   Operator *M = _ricci.MassOp();
   Operator *K = _ricci.KOp();
   for (int s = 0; s < NUM_VARS; ++s)
   {
      M->Mult(k_blk.GetBlock(s), _tmp_block);
      R_blk.GetBlock(s) += _tmp_block;
      K->Mult(_z.GetBlock(s), _tmp_block);
      R_blk.GetBlock(s) += _tmp_block;
   }

   R_blk.SyncFromBlocks();
}

Operator &RicciImplicitStageOp::GetGradient(const Vector &k_vec) const
{
   MFEM_VERIFY(_u_pred != nullptr,
               "RicciImplicitStageOp: SetParameters() not called before "
               "GetGradient().");

   // z = u_pred + gamma * k.
   add(*_u_pred, _gamma, k_vec, _z);
   _z.SyncToBlocks();

   // PA F'_* blocks (unscaled)
   BlockOperator &Fgrad = _ricci.ReactionOp()->GetGradient(_z);

   const int  n_active = _ricci.NumActive();
   Operator  *M        = _ricci.MassOp();
   Operator  *MgK      = MassPlusGammaK();   // built in SetParameters

   // T row is always active.
   _diag_T = std::make_unique<SumOperator>(MgK, 1.0,
                                            &Fgrad.GetBlock(T_IDX, T_IDX),
                                            _gamma,
                                            /*ownA=*/false, /*ownB=*/false);
   _Jac_block.SetDiagonalBlock(T_IDX, _diag_T.get());

   // omega row: M + γK if active (F'_{ω,ω} = 0), bare M if frozen.
   _Jac_block.SetDiagonalBlock(OMEGA_IDX,
                               (n_active >= 2) ? MgK : M);

   // n row: (M+γK) ⊕ γ·F'_{n,n} if active, bare M if frozen.
   if (n_active >= 3)
   {
      _diag_n = std::make_unique<SumOperator>(MgK, 1.0,
                                               &Fgrad.GetBlock(N_IDX, N_IDX),
                                               _gamma,
                                               false, false);
      _Jac_block.SetDiagonalBlock(N_IDX, _diag_n.get());
   }
   else
   {
      _diag_n.reset();
      _Jac_block.SetDiagonalBlock(N_IDX, M);
   }

   // Off-diagonals: γ·F'_{ω,T}, γ·F'_{n,T}.
   if (n_active >= 2)
   {
      _gFwT = std::make_unique<ScaledOperator>(
         &Fgrad.GetBlock(OMEGA_IDX, T_IDX), _gamma);
      _Jac_block.SetBlock(OMEGA_IDX, T_IDX, _gFwT.get());
   }
   if (n_active >= 3)
   {
      _gFnT = std::make_unique<ScaledOperator>(
         &Fgrad.GetBlock(N_IDX, T_IDX), _gamma);
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
      // Device-resident scratch, sized once and reused across Newton iters.
      _x_trial.SetSize(x.Size()); _x_trial.UseDevice(true);
      _r_trial.SetSize(r.Size()); _r_trial.UseDevice(true);
      real_t alpha = 1.0;

      for (int i = 0; i < max_tries; ++i)
      {
         add(x, -alpha, c, _x_trial);
         oper->Mult(_x_trial, _r_trial);
         if (b.Size() == _r_trial.Size()) { _r_trial -= b; }
         const real_t norm_trial = Norm(_r_trial);
         if (norm_trial <= (1.0 - sigma * alpha) * norm0) { return alpha; }
         alpha *= 0.5;
      }
      // Couldn't satisfy Armijo; return the smallest tried
      return alpha;
   }

private:
   mutable Vector _x_trial, _r_trial;
};

// Block Krylov solver consumed by NewtonSolver as its linear solver.
class BlockNewtonLinearSolver : public IterativeSolver
{
public:
   BlockNewtonLinearSolver(MPI_Comm comm,
                           bool matrix_free,
                           real_t rtol = 1e-10,
                           int max_it = 200,
                           int kdim = 50)
      : IterativeSolver(comm),
        _matrix_free(matrix_free),
        _gmres(comm)
   {
      SetRelTol(rtol);
      SetAbsTol(0.0);
      SetMaxIter(max_it);
      _gmres.SetKDim(kdim);
      _gmres.SetPrintLevel(0);
      _jacobi.resize(NUM_VARS);
      if (!_matrix_free)
      {
         _amgs.resize(NUM_VARS);
         for (int s = 0; s < NUM_VARS; ++s)
         {
            _amgs[s] = std::make_unique<HypreBoomerAMG>();
            _amgs[s]->SetPrintLevel(0);
         }
      }
   }

   // Legacy path: pin the AMG hierarchies to the assembled (M+γK)/M operands.
   void SetPrecondOperands(HypreParMatrix &M,
                           HypreParMatrix &M_plus_gK,
                           int num_active)
   {
      _amgs[T_IDX]->SetOperator(M_plus_gK);
      _amgs[OMEGA_IDX]->SetOperator((num_active >= 2) ? M_plus_gK : M);
      _amgs[N_IDX]->SetOperator((num_active >= 3) ? M_plus_gK : M);
   }

   void SetPrecondDiagonals(const Vector &d_T, const Vector &d_w,
                            const Vector &d_n)
   {
      refreshJacobi(T_IDX,     d_T);
      refreshJacobi(OMEGA_IDX, d_w);
      refreshJacobi(N_IDX,     d_n);
   }

   void SetOperator(const Operator &op) override
   {
      const BlockOperator *bop = dynamic_cast<const BlockOperator*>(&op);
      MFEM_VERIFY(bop != nullptr,
                  "BlockNewtonLinearSolver: expected a BlockOperator from "
                  "RicciImplicitStageOp::GetGradient().");

      if (!_bdp)
      {
         _bdp = std::make_unique<BlockDiagonalPreconditioner>(bop->RowOffsets());

         for (int s = 0; s < NUM_VARS; ++s)
         {
            Solver *prec = _matrix_free ? (Solver*)_jacobi[s].get()
                                        : (Solver*)_amgs[s].get();
            _bdp->SetDiagonalBlock(s, prec);
         }

         _gmres.SetPreconditioner(*_bdp);
      }

      _gmres.SetOperator(*bop);
      height = bop->Height();
      width  = bop->Width();
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      const_cast<Vector&>(b).UseDevice(true);
      x.UseDevice(true);

      // Propagate adaptive rtol set by NewtonSolver since the last call.
      _gmres.SetRelTol(rel_tol);
      _gmres.SetAbsTol(abs_tol);
      _gmres.SetMaxIter(max_iter);
      _gmres.Mult(b, x);
   }

private:
   void refreshJacobi(int s, const Vector &d)
   {
      if (!_jacobi[s])
         _jacobi[s] = std::make_unique<OperatorJacobiSmoother>(d, _empty_ess);
      else
         _jacobi[s]->Setup(d);
   }

   bool                                         _matrix_free;
   mutable GMRESSolver                          _gmres;
   std::unique_ptr<BlockDiagonalPreconditioner> _bdp;
   std::vector<std::unique_ptr<HypreBoomerAMG>> _amgs;       // legacy path
   std::vector<std::unique_ptr<OperatorJacobiSmoother>> _jacobi; // matrix-free
   Array<int>                                   _empty_ess;
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
     _lin_solver(new BlockNewtonLinearSolver(comm, ricci.MatrixFree(),
                                             lin_rtol, lin_max_iter, kdim))
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
   k.UseDevice(true);
   const_cast<Vector&>(u_pred).UseDevice(true);

   BlockVector u_pred_blk(const_cast<Vector&>(u_pred),
                          _ricci.BlockTrueOffsets());
   u_pred_blk.UseDevice(true);
   for (int s = 0; s < NUM_VARS; ++s) { u_pred_blk.GetBlock(s).UseDevice(true); }
   u_pred_blk.SyncToBlocks();

   _ricci.pullOmegaFromBlocks(u_pred_blk);
   _ricci.updatePhi();
   _ricci.reassembleK();
   _ricci.projectPhiToQuadrature();

   // Set up M + γK
   _stage_op.SetParameters(gamma, u_pred_blk);

   auto *block_solver =
      dynamic_cast<BlockNewtonLinearSolver*>(_lin_solver.get());
   MFEM_VERIFY(block_solver,
               "RicciTimeOperator: linear solver is not a "
               "BlockNewtonLinearSolver.");

   if (_ricci.MatrixFree())
   {
      // Per-block Jacobi diagonals: diag(M)+γ·diag(K) on active rows, diag(M)
      // on frozen rows.  (γ·F' diagonal omitted, matching the legacy AMG.)
      const int na = _ricci.NumActive();
      const Vector &dM = _ricci.MassDiag();
      const Vector &dK = _ricci.KDiag();
      Vector d_T(dM.Size()), d_w(dM.Size()), d_n(dM.Size());
      d_T.UseDevice(true); d_w.UseDevice(true); d_n.UseDevice(true);
      add(dM, gamma, dK, d_T);
      if (na >= 2) { add(dM, gamma, dK, d_w); } else { d_w = dM; }
      if (na >= 3) { add(dM, gamma, dK, d_n); } else { d_n = dM; }
      block_solver->SetPrecondDiagonals(d_T, d_w, d_n);
   }
   else
   {
      block_solver->SetPrecondOperands(*_ricci.MassMatHypre(),
                                       *_stage_op.MassPlusGammaKHypre(),
                                       _ricci.NumActive());
   }

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
   std::string assembly_level = "partial";
   bool   phi_lor           = false;
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
   args.AddOption(&assembly_level, "-al", "--assembly-level",
                  "Assembly level for the block operators M, K, and F': "
                  "'partial' = matrix-free PA + Jacobi block preconditioner "
                  "(GPU-friendly), 'legacy' = assembled HypreParMatrix + "
                  "BoomerAMG.");
   args.AddOption(&phi_lor, "-lor", "--phi-lor", "-no-lor", "--no-phi-lor",
                  "Solve the phi Poisson problem with a PA operator "
                  "preconditioned by LOR-AMG (on) or an assembled Laplacian "
                  "with BoomerAMG (off).");
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

   bool matrix_free = true;
   if (assembly_level == "legacy")       { matrix_free = false; }
   else if (assembly_level == "partial") { matrix_free = true; }
   else
   {
      if (myid == 0)
      {
         std::cout << "Unknown --assembly-level '" << assembly_level
                   << "'; expected 'partial' or 'legacy'." << std::endl;
      }
      return 1;
   }

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
                          num_active, matrix_free, phi_lor, save_dir);
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
