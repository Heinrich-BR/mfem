#ifndef MFEM_EX43P_HPP
#define MFEM_EX43P_HPP

#include "mfem.hpp"
#include <filesystem>

using namespace mfem;

enum VarIdx : int { T_IDX = 0, OMEGA_IDX = 1, N_IDX = 2 };
constexpr int NUM_VARS = 3;

// Device-native nonlinear reaction operator for the Rogers-Ricci system.
// Replaces the host-only RogersRicciNLFIntegrator + ParBlockNonlinearForm
// pipeline used by the original ex42p.  Pipeline:
//   1. sampleStateToQPs:   true-dof z = (T, ω, n) -> per-block QFs for T, n
//      (Prolongation -> ElementRestriction -> QuadratureInterpolator::Values,
//       all device-native, all mfem::forall).
//   2. computeReactionResidualAtQPs / computeReactionJacobianAtQPs: custom
//      mfem::forall kernels that read T_qf, n_qf, phi_qf, Sn_qf and write
//      F_*_qf (residual) and dF_*/du_*_qf (Jacobian entries).
//   3. Mult:        three ParLinearForms with DomainLFIntegrator(QFC) take the
//                   freshly-populated F_*_qf and assemble the residual blocks
//                   on device (DLFEvalAssemble path).
//   4. GetGradient: four ParBilinearForms with MassIntegrator(QFC) and
//                   AssemblyLevel::PARTIAL produce device-native PA operators
//                   wrapped by FormSystemOperator so they act on true-dofs.
//                   The returned BlockOperator holds unscaled F'_* in its
//                   slots — the stage op applies γ when composing with M+γK.
class RogersRicciReactionOp : public Operator
{
public:
   RogersRicciReactionOp(ParFiniteElementSpace &fes,
                         const QuadratureSpace &qspace,
                         const IntegrationRule &ir,
                         const Array<int> &block_offsets,
                         const QuadratureFunction &phi_qf,
                         const QuadratureFunction &Sn_qf,
                         int num_active,
                         real_t Lambda,
                         real_t eps2);

   void Mult(const Vector &z_true, Vector &R_true) const override;
   // Covariant return — base class returns Operator&; the caller can
   // address blocks directly without an explicit dynamic_cast.
   BlockOperator &GetGradient(const Vector &z_true) const override;

private:
   void sampleStateToQPs(const Vector &z_true) const;
   void computeReactionResidualAtQPs() const;
   void computeReactionJacobianAtQPs() const;

   ParFiniteElementSpace            &_fes;
   const QuadratureSpace            &_qspace;
   const IntegrationRule            &_ir;
   const Array<int>                 &_block_offsets;
   const QuadratureFunction         *_phi_qf;
   const QuadratureFunction         *_Sn_qf;
   int      _num_active;
   real_t   _Lambda, _eps2;

   // Cached, FES-owned, device-aware.
   const ElementRestrictionOperator *_er;
   const QuadratureInterpolator     *_qi;

   // QPs samples of the state.  T always populated; n only when num_active>=3.
   mutable QuadratureFunction _T_qf, _n_qf;

   // Per-QP residual contributions (vdim=1 each, on _qspace).
   mutable QuadratureFunction _F_T_qf, _F_omega_qf, _F_n_qf;
   mutable QuadratureFunctionCoefficient _F_T_qfc, _F_omega_qfc, _F_n_qfc;

   // Per-QP Jacobian entries.
   mutable QuadratureFunction _dFT_dT_qf, _dFw_dT_qf, _dFn_dT_qf, _dFn_dn_qf;
   mutable QuadratureFunctionCoefficient _dFT_dT_qfc, _dFw_dT_qfc,
                                          _dFn_dT_qfc, _dFn_dn_qfc;

   // Scratch L- and E-vectors (allocated once, reused).
   mutable Vector _l_vec_T, _l_vec_n;
   mutable Vector _e_vec_T, _e_vec_n;

   // Residual assembly via DomainLFIntegrator(QFC) (device path).
   mutable std::unique_ptr<ParLinearForm> _lf_T, _lf_omega, _lf_n;

   // Jacobian: MassIntegrator(QFC) with AssemblyLevel::PARTIAL.  Output is
   // an L-vector-acting PA op wrapped by FormSystemOperator into a true-dof
   // acting Operator.
   mutable std::unique_ptr<ParBilinearForm> _bf_TT, _bf_wT, _bf_nT, _bf_nn;
   mutable OperatorHandle                   _J_TT, _J_wT, _J_nT, _J_nn;

   // Persistent 3x3 BlockOperator returned by GetGradient (holds unscaled
   // F'_* operators; the stage op applies γ and composes with M+γK).
   mutable BlockOperator _F_block;
};

// Owns the mesh, the H1 ParFiniteElementSpace shared by the three
// time-evolved variables (T, omega, n), the auxiliary potential phi, the
// drift-velocity coefficients derived from phi, the mass matrix,
// the per-stage advection+SUW matrix, and the nonlinear reaction form.
class Ricci2DNonlinear
{
public:

   // num_active = 1, 2, or 3 picks how many of (T, omega, n) are time-evolved.
   //   1 - T only (omega, n frozen at their initial values; phi stays uniform)
   //   2 - T and omega (n frozen)
   //   3 - full system
   Ricci2DNonlinear(MPI_Comm comm,
                    int order, int ref_levels,
                    real_t xL, real_t yL,
                    int num_active,
                    const std::string &save_dir);
   
   void Setup(const std::string &restart_dir);

   // Save and checkpoint
   void Save();
   std::string CheckpointDir() const { return _save_dir + "/Checkpoint"; }
   const std::string &SaveDir() const { return _save_dir; }
   void SaveCheckpoint(const std::string &dir, real_t t, int step) const;
   static void LoadCheckpointMeta(const std::string &dir, MPI_Comm comm,
                                  real_t &t, int &step);
   
   void pullOmegaFromBlocks(const BlockVector &u_blk);
   void updatePhi();
   void reassembleK();
   void updateDataCollection(int step, real_t t);
   void syncGridFuncsFromBlocks(const BlockVector &u_blk);
   void syncBlocksFromGridFuncs(BlockVector &u_blk) const;

   const Array<int>      &BlockTrueOffsets() const { return _block_trueOffsets; }
   ParFiniteElementSpace *H1FES()             const { return _h1_fes.get();     }
   RogersRicciReactionOp *ReactionOp()        const { return _reaction_op.get(); }
   HypreParMatrix        *MassMat()           const { return _M_mat.get();      }
   HypreParMatrix        *KMat()              const { return _K_mat.get();      }
   ParGridFunction       *Phi()               const { return _phi.get();        }
   BlockVector           *StateBlocks()       const { return _var_blocks.get(); }
   int                    NumActive()         const { return _num_active;       }

   real_t _xL, _yL;
   int    _order, _ref_levels;
   int    _dim    = 2;
   real_t _eps2   = 1e-4;
   real_t _Binv   = 40.0;
   real_t _Lambda = 3.0;
   real_t _S_0n   = 0.03;
   real_t _h      = 0.0; // element size
   int    _num_active = NUM_VARS;

private:
   MPI_Comm _comm;

   void buildMesh();
   void buildSpaces();
   void buildState();
   void buildCoefficients();
   void buildQuadratureSamples();
   void buildMassMatrix();
   void buildKForm();
   void buildPhiSolver();
   void buildNonlinearForm();
   void setInitialConditions();
   void setOutput();

   void loadFieldsFromCheckpoint();
   std::string _restart_dir;
   std::string _save_dir;

public:

   void projectPhiToQuadrature();

private:

   std::unique_ptr<ParMesh> _pmesh;
   std::unique_ptr<ParFiniteElementSpace> _h1_fes;

   std::vector<std::unique_ptr<ParGridFunction>> _vars;
   std::unique_ptr<ParGridFunction> _phi;
   std::unique_ptr<BlockVector> _var_blocks;
   Array<int> _block_trueOffsets;

   std::unique_ptr<ParBilinearForm> _M_form;
   std::unique_ptr<HypreParMatrix> _M_mat;
   std::unique_ptr<ParBilinearForm> _K_form;
   std::unique_ptr<HypreParMatrix> _K_mat;

   Array<int>                                    _ess_tdof_list_phi;
   std::unique_ptr<GridFunctionCoefficient>      _omega_coef;
   std::unique_ptr<ProductCoefficient>           _neg_omega_coef;
   std::unique_ptr<ParBilinearForm>              _a_phi;
   std::unique_ptr<ParLinearForm>                _b_phi;
   OperatorPtr                                   _A_phi;
   std::unique_ptr<HypreBoomerAMG>               _amg_phi;
   std::unique_ptr<CGSolver>                     _cg_phi;

   DenseMatrix                                          _rotmat;
   std::unique_ptr<Coefficient>                         _r;
   std::unique_ptr<Coefficient>                         _Sn;
   std::unique_ptr<MatrixConstantCoefficient>           _grad_rotate;
   std::unique_ptr<GradientGridFunctionCoefficient>     _grad_phi;
   std::unique_ptr<MatrixVectorProductCoefficient>      _vd;
   std::unique_ptr<ScalarVectorProductCoefficient>      _v_E;
   std::unique_ptr<InnerProductCoefficient>             _v_E_sq;
   std::unique_ptr<TransformedCoefficient>              _suw_scalar;
   std::unique_ptr<OuterProductCoefficient>             _v_E_outer;
   std::unique_ptr<ScalarMatrixProductCoefficient>      _SUW_matcoef;

   // Cached quadrature samples for phi and Sn
   std::unique_ptr<QuadratureSpace>                     _qspace;
   std::unique_ptr<QuadratureFunction>                  _phi_qf;
   std::unique_ptr<QuadratureFunction>                  _Sn_qf;
   std::unique_ptr<QuadratureFunctionCoefficient>       _phi_qfc;
   std::unique_ptr<QuadratureFunctionCoefficient>       _Sn_qfc;

   // Cached pointers + scratch E-vector for the device-friendly phi → QP
   // pipeline used by projectPhiToQuadrature.  Both pointers are FES-owned;
   // we don't free them.  The E-vector is sized to _er_h1->Height().
   const ElementRestrictionOperator                    *_er_h1 = nullptr;
   const QuadratureInterpolator                        *_qi_h1 = nullptr;
   Vector                                               _phi_e_vec;

   std::unique_ptr<RogersRicciReactionOp>               _reaction_op;

   std::unique_ptr<ParaViewDataCollection>              _dc;
};

// Residual + Jacobian wrapper consumed by NewtonSolver.
//
//     R(k) = M·k + K·(u_pred + γ·k) + F(u_pred + γ·k) = 0
//
// for a stage derivative `k`, given the stage predictor `u_pred` and stage
// step `γ` (= dt for backward Euler, = a_ii·dt for SDIRK).  NewtonSolver
// calls Mult to get R and GetGradient to get
//
//     J(k) = M_diag + γ·K_diag + γ·F'(u_pred + γ·k)
// ---------------------------------------------------------------------------
class RicciImplicitStageOp : public Operator
{
public:
   explicit RicciImplicitStageOp(Ricci2DNonlinear &ricci);

   // Cache the parameters for the current implicit stage
   void SetParameters(real_t gamma, const BlockVector &u_pred);

   void Mult(const Vector &k, Vector &R) const override;
   Operator &GetGradient(const Vector &k) const override;

private:
   Ricci2DNonlinear   &_ricci;
   real_t              _gamma;
   const BlockVector  *_u_pred;

   mutable BlockVector              _z;
   mutable Vector                   _tmp_block;

   mutable BlockOperator                   _Jac_block;
   mutable std::unique_ptr<HypreParMatrix> _M_plus_gK;   // built per stage

   // Per-Newton-iter composition wrappers (cheap, no SpMV):
   //   off-diagonals: γ·F'_{ω,T}, γ·F'_{n,T}
   //   diagonals:     (M+γK) ⊕ γ·F'_{T,T},  (M+γK) ⊕ γ·F'_{n,n}
   mutable std::unique_ptr<ScaledOperator> _gFwT;
   mutable std::unique_ptr<ScaledOperator> _gFnT;
   mutable std::unique_ptr<SumOperator>    _diag_T;
   mutable std::unique_ptr<SumOperator>    _diag_n;

public:
   // Exposed so the linear solver can pin AMG hierarchies to the static
   // (M + γK) operand once per implicit stage instead of re-doing AMG
   // setup every Newton iter.
   HypreParMatrix *MassPlusGammaK() const { return _M_plus_gK.get(); }
};

// TimeDependentOperator that the ODESolver drives.
class RicciTimeOperator : public TimeDependentOperator
{
public:
   RicciTimeOperator(MPI_Comm comm, Ricci2DNonlinear &ricci,
                     real_t newton_rtol = 1e-8, int newton_max_iter = 15,
                     real_t lin_rtol = 1e-10, int lin_max_iter = 200,
                     int kdim = 50);

   void Mult(const Vector&, Vector&) const override
   {
      MFEM_ABORT("RicciTimeOperator::Mult: only implicit solvers are supported.");
   }

   void ImplicitSolve(real_t gamma, const Vector &u_pred, Vector &k) override;

   // Turn on Eisenstat-Walker forcing so the inner Krylov tolerance loosens
   // when Newton is far from convergence.
   void EnableEisenstatWalker(real_t rtol0 = 0.5, real_t rtol_max = 0.9);

private:

   Ricci2DNonlinear & _ricci;
   RicciImplicitStageOp _stage_op;
   std::unique_ptr<NewtonSolver>  _newton;
   std::unique_ptr<IterativeSolver> _lin_solver;
};

#endif // MFEM_EX43P_HPP
