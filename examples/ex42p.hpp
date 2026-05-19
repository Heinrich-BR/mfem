#ifndef MFEM_EX42P_HPP
#define MFEM_EX42P_HPP

#include "mfem.hpp"
#include <filesystem>

using namespace mfem;

enum VarIdx : int { T_IDX = 0, OMEGA_IDX = 1, N_IDX = 2 };
constexpr int NUM_VARS = 3;

// Custom BlockNonlinearFormIntegrator implementing the nonlinear reaction
// terms F_T, F_omega, F_n
class RogersRicciNLFIntegrator : public BlockNonlinearFormIntegrator
{
public:
   // num_active selects how many variables are time-evolved:
   //   1 → T only;  omega and n stay at their initial values
   //   2 → T and omega;  n stays at its initial value
   //   3 → full T, omega, n system
   // phi and Sn are passed as base Coefficient& so the caller can pass either
   // a GridFunctionCoefficient (legacy path) or, preferably, a
   // QuadratureFunctionCoefficient backed by a phi/Sn that has been projected
   // once per stage — O(1) lookup per QP instead of a basis evaluation.
   RogersRicciNLFIntegrator(Coefficient &phi,
                          Coefficient &Sn,
                          int num_active = NUM_VARS,
                          real_t Lambda = 3.0,
                          real_t eps2 = 1e-4)
      : _phi(&phi), _Sn(&Sn),
        _Lambda(Lambda), _eps2(eps2),
        _num_active(num_active)
   {
      MFEM_VERIFY(num_active >= 1 && num_active <= NUM_VARS,
                  "RogersRicciNLFIntegrator: num_active out of range.");
   }

   // Residual contribution to each block:
   //   elvec[T_IDX]     += integral of F_T(T, phi) * shape
   //   elvec[OMEGA_IDX] += integral of F_omega(T, phi) * shape
   //   elvec[N_IDX]     += integral of F_n(n, T, phi) * shape
   void AssembleElementVector(const Array<const FiniteElement*> &el,
                              ElementTransformation &Tr,
                              const Array<const Vector*> &elfun,
                              const Array<Vector*> &elvec) override;

   // Jacobian (state-derivative of the residual).  Non-zero blocks:
   //   (T,T)     dF_T/dT
   //   (omega,T) dF_omega/dT
   //   (n,T)     dF_n/dT
   //   (n,n)     dF_n/dn
   // All others are explicitly zeroed.
   void AssembleElementGrad(const Array<const FiniteElement*> &el,
                            ElementTransformation &Tr,
                            const Array<const Vector*> &elfun,
                            const Array2D<DenseMatrix*> &elmats) override;

private:
   Coefficient *_phi;
   Coefficient *_Sn;
   real_t _Lambda;
   real_t _eps2;
   int    _num_active;
};

// ---------------------------------------------------------------------------
// Ricci2DNonlinear — problem container for ex42p.
//
// Owns the mesh, the single H1 ParFiniteElementSpace shared by the three
// time-evolved variables (T, omega, n), the auxiliary potential phi, the
// drift-velocity coefficients derived from phi, the (frozen) mass matrix,
// the per-stage advection+SUW matrix, and the nonlinear reaction form
// (RogersRicciNLFIntegrator wrapped in a ParBlockNonlinearForm).
//
// The class is passive: it exposes assemble/refresh primitives but does not
// drive a time loop.  RicciTimeOperator (added next) orchestrates the calls.
// ---------------------------------------------------------------------------
class Ricci2DNonlinear
{
public:

   // num_active = 1, 2, or 3 picks how many of (T, omega, n) are time-evolved.
   //   1 → T only (omega, n frozen at their initial values; phi stays uniform)
   //   2 → T and omega (n frozen)
   //   3 → full system
   // The block layout is always 3-wide regardless; "frozen" simply means the
   // implicit-stage equation for those rows reduces to M*k_s = 0.
   // xL and yL are in normalised units where 1 length unit = rho_s0
   // (matching the Firedrake reference; the source position rs = 20, the
   // source width Ls = 0.5, and the advection coefficient _Binv = R/rho_s0
   // = 40 are all expressed in these units, so do not change xL/yL unless
   // you also rescale those constants).
   Ricci2DNonlinear(MPI_Comm comm,
                    int order, int ref_levels,
                    real_t xL, real_t yL,
                    int num_active,
                    const std::string &save_dir);

   // Path layout, derived from `save_dir`:
   //   <save_dir>/Step.pvd          (ParaView registry, see setOutputCollection)
   //   <save_dir>/Step/Cycle*       (ParaView per-frame data, written by Save)
   //   <save_dir>/Checkpoint/...    (single-slot state snapshot, see CheckpointDir)
   std::string CheckpointDir() const { return _save_dir + "/Checkpoint"; }
   const std::string &SaveDir() const { return _save_dir; }

   // One-shot setup: builds everything, sets ICs, assembles M and an initial
   // K_mat, and registers ParaView fields.  Call after construction, before
   // entering the time loop.
   //
   // If restart_dir is non-empty, Setup loads the mesh and the four
   // ParGridFunctions (T, omega, n, phi) from that directory instead of
   // constructing a fresh Cartesian mesh and applying initial conditions.
   // The caller is responsible for having read the scalar metadata
   // (t, step) via LoadCheckpointMeta beforehand.
   void Setup(const std::string &restart_dir);

   // Write a single-slot checkpoint into `dir`.  All ranks must participate.
   // Overwrites any previous checkpoint in `dir`.  The output is:
   //   <dir>/meta.txt          (rank 0; "t <value>\nstep <int>\n")
   //   <dir>/mesh.NNNNNN       (per-rank ParMesh::ParPrint)
   //   <dir>/T.NNNNNN          (per-rank ParGridFunction::Save)
   //   <dir>/omega.NNNNNN
   //   <dir>/n.NNNNNN
   //   <dir>/phi.NNNNNN
   void SaveCheckpoint(const std::string &dir, real_t t, int step) const;

   // Read `dir/meta.txt` on rank 0 and broadcast t/step to all ranks.
   // Static so main() can call it before constructing the problem object.
   static void LoadCheckpointMeta(const std::string &dir, MPI_Comm comm,
                                  real_t &t, int &step);

   // Per-implicit-stage primitives -----------------------------------------
   //   Pull the omega block out of a true-dof BlockVector (the predictor
   //   state passed by the ODE solver) into the omega ParGridFunction so
   //   that updatePhi() sees the right rhs.
   void pullOmegaFromBlocks(const BlockVector &u_blk);
   //   Solve  ∇²phi = omega  with Dirichlet boundary values held at the
   //   current values of _phi (i.e. preserved from setup / previous stage).
   void updatePhi();
   //   Re-evaluate v_d-derived coefficients (cheap, just bookkeeping).
   void refreshDriftCoefs();
   //   Re-assemble the monolithic K (advection + SUW) HypreParMatrix.
   //   Must be called after updatePhi/refreshDriftCoefs each stage.
   void reassembleK();

   // I/O -------------------------------------------------------------------
   void Save();
   void updateDataCollection(int step, real_t t);

   // Sync helpers used by RicciTimeOperator -------------------------------
   //   Push a true-dof BlockVector into the per-variable ParGridFunctions
   //   (so coefficients/output reflect the current state).
   void syncGridFuncsFromBlocks(const BlockVector &u_blk);
   //   Pack the per-variable ParGridFunctions into a true-dof BlockVector
   //   (used once at the start of the run to seed the ODE state).
   void syncBlocksFromGridFuncs(BlockVector &u_blk) const;

   const Array<int>      &BlockTrueOffsets() const { return _block_trueOffsets; }
   ParFiniteElementSpace *H1FES()             const { return _h1_fes.get();     }
   ParBlockNonlinearForm *FForm()             const { return _F_form.get();     }
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
   void setOutputCollection();

   // Restart helpers.  `_restart_dir` is set at the top of Setup and read by
   // buildMesh (to decide between Cartesian-fresh and load-from-disk) and by
   // the IC dispatch (setInitialConditions vs loadFieldsFromCheckpoint).
   void loadFieldsFromCheckpoint();
   std::string _restart_dir;
   std::string _save_dir;

public:
   // Resample phi from the H1 grid function into the quadrature-point
   // buffer used by the reaction integrator.  Cheap (O(nelem * nQP)) and
   // amortises the per-QP basis evaluation across every Newton iteration of
   // the implicit stage.  Called by RicciTimeOperator::ImplicitSolve right
   // after updatePhi().
   void projectPhiToQuadrature();

private:

   std::unique_ptr<ParMesh>                _pmesh;
   std::unique_ptr<ParFiniteElementSpace>  _h1_fes;

   std::vector<std::unique_ptr<ParGridFunction>> _vars;
   std::unique_ptr<ParGridFunction>              _phi;
   std::unique_ptr<BlockVector>                  _var_blocks;
   Array<int>                                    _block_trueOffsets;

   std::unique_ptr<ParBilinearForm>        _M_form;
   std::unique_ptr<HypreParMatrix>         _M_mat;
   std::unique_ptr<ParBilinearForm>        _K_form;
   std::unique_ptr<HypreParMatrix>         _K_mat;

   // Cached phi-Poisson solver.  Laplacian operator and Dirichlet BC pattern
   // are static, so the matrix, AMG hierarchy, and CG solver are all built
   // once in Setup and reused every implicit stage; only the RHS is rebuilt.
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

   // Cached quadrature samples for phi and Sn so the reaction integrator
   // does O(1) lookups instead of a full GridFunctionCoefficient::Eval
   // (basis + interpolation) per QP per Newton step.  Sn is geometry-only
   // and projected once in Setup; phi is reprojected each stage after
   // updatePhi() — see projectPhiToQuadrature().
   std::unique_ptr<QuadratureSpace>                     _qspace;
   std::unique_ptr<QuadratureFunction>                  _phi_qf;
   std::unique_ptr<QuadratureFunction>                  _Sn_qf;
   std::unique_ptr<QuadratureFunctionCoefficient>       _phi_qfc;
   std::unique_ptr<QuadratureFunctionCoefficient>       _Sn_qfc;

   std::unique_ptr<ParBlockNonlinearForm>               _F_form;

   std::unique_ptr<ParaViewDataCollection>              _dc;
};

// ---------------------------------------------------------------------------
// RicciImplicitStageOp — residual + Jacobian wrapper consumed by NewtonSolver.
//
// Mirrors the role of ex10p's ReducedSystemOperator: encodes the implicit-
// stage equation
//
//     R(k) = M·k + K·(u_pred + γ·k) + F(u_pred + γ·k) = 0
//
// for a stage derivative `k`, given the stage predictor `u_pred` and stage
// step `γ` (= dt for backward Euler, = a_ii·dt for SDIRK).  NewtonSolver
// calls Mult to get R and GetGradient to get
//
//     J(k) = M_diag + γ·K_diag + γ·F'(u_pred + γ·k)
//
// assembled monolithically.  M and K are block-diagonal in the same H1 mass /
// advection matrix; the stage operator does the block-by-block expansion
// internally so Ricci2DNonlinear only needs to store the single-block forms.
// ---------------------------------------------------------------------------
class RicciImplicitStageOp : public Operator
{
public:
   explicit RicciImplicitStageOp(Ricci2DNonlinear &ricci);

   // Cache the parameters for the current implicit stage.  RicciTimeOperator
   // calls this from inside ImplicitSolve before invoking NewtonSolver.  We
   // pre-compute M + gamma*K here (constant for the whole stage) so that the
   // per-Newton GetGradient call only has to add in the small reaction
   // contributions.
   void SetParameters(real_t gamma, const BlockVector &u_pred);

   void Mult(const Vector &k, Vector &R) const override;
   Operator &GetGradient(const Vector &k) const override;

private:
   Ricci2DNonlinear   &_ricci;
   real_t              _gamma;
   const BlockVector  *_u_pred;

   // Workspace
   mutable BlockVector              _z;
   mutable Vector                   _tmp_block;

   // Persistent block Jacobian.  We never materialise the monolithic
   // HypreParMatrix; the linear solver (GMRES + block-diag AMG) operates
   // directly on the block form, so each Newton iter only has to rebuild
   // the diagonal reaction additions and the two off-diagonal F' blocks.
   mutable BlockOperator                   _Jac_block;
   mutable std::unique_ptr<HypreParMatrix> _M_plus_gK; // built per stage
   mutable std::unique_ptr<HypreParMatrix> _diag_T;    // built per Newton iter
   mutable std::unique_ptr<HypreParMatrix> _diag_n;    // (n_active >= 3)
   mutable std::unique_ptr<HypreParMatrix> _gFwT;      // (n_active >= 2)
   mutable std::unique_ptr<HypreParMatrix> _gFnT;      // (n_active >= 3)
};

// ---------------------------------------------------------------------------
// RicciTimeOperator — TimeDependentOperator that the ODESolver drives.
//
// For each implicit stage the ODE solver calls ImplicitSolve(γ, u_pred, k):
//   1. update phi from the predictor's omega block (Poisson solve),
//   2. refresh v_d-derived coefficients,
//   3. re-assemble K (advection + SUW) at the new v_d,
//   4. configure the stage equation, then
//   5. run NewtonSolver to convergence on R(k) = 0.
// ---------------------------------------------------------------------------
class RicciTimeOperator : public TimeDependentOperator
{
public:
   RicciTimeOperator(MPI_Comm comm, Ricci2DNonlinear &ricci,
                     real_t newton_rtol = 1e-8, int newton_max_iter = 15,
                     real_t lin_rtol = 1e-10, int lin_max_iter = 200,
                     int kdim = 50);

   // Explicit RHS is not supported — only DIRK-family solvers via ImplicitSolve.
   void Mult(const Vector&, Vector&) const override
   {
      MFEM_ABORT("RicciTimeOperator::Mult: only implicit solvers are supported.");
   }

   void ImplicitSolve(real_t gamma, const Vector &u_pred, Vector &k) override;

   // Turn on Eisenstat-Walker forcing so the inner Krylov tolerance loosens
   // when Newton is far from convergence.  Useful when Newton typically takes
   // many iterations per stage; usually unhelpful when it converges in 1-2.
   void EnableEisenstatWalker(real_t rtol0 = 0.5, real_t rtol_max = 0.9);

private:
   Ricci2DNonlinear              &_ricci;
   RicciImplicitStageOp           _stage_op;
   std::unique_ptr<NewtonSolver>  _newton;       // BacktrackingNewtonSolver
                                                  // internally; type-erased so
                                                  // the header stays free of
                                                  // the line-search subclass
   std::unique_ptr<IterativeSolver> _lin_solver; // BlockNewtonLinearSolver
                                                  // internally (GMRES + block-
                                                  // diag AMG); same reason
};

#endif // MFEM_EX42P_HPP
