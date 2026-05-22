#ifndef MFEM_EX42P_HPP
#define MFEM_EX42P_HPP

#include "mfem.hpp"
#include <filesystem>

using namespace mfem;

enum VarIdx : int { T_IDX = 0, OMEGA_IDX = 1, N_IDX = 2 };
constexpr int NUM_VARS = 3;

// Custom BlockNonlinearFormIntegrator implementing the nonlinear terms
// F_T, F_omega, F_n
class RogersRicciNLFIntegrator : public BlockNonlinearFormIntegrator
{
public:

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
   // when Newton is far from convergence.  Useful when Newton typically takes
   // many iterations per stage; usually unhelpful when it converges in 1-2.
   void EnableEisenstatWalker(real_t rtol0 = 0.5, real_t rtol_max = 0.9);

private:

   Ricci2DNonlinear & _ricci;
   RicciImplicitStageOp _stage_op;
   std::unique_ptr<NewtonSolver>  _newton;
   std::unique_ptr<IterativeSolver> _lin_solver;
};

#endif // MFEM_EX42P_HPP
