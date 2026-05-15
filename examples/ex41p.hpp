#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using namespace mfem;

enum VarIdx : int { T_IDX = 0, OMEGA_IDX = 1, N_IDX = 2 };

// Number of variables actually evolved in time. Set at compile time:
//   1 → only T (ω and n stay at their initial values; ϕ stays uniform)
//   2 → T and ω (n stays at its initial value)
//   3 → full T, ω, n system
// Variables with index >= NUM_VARS are not allocated; access to them is
// guarded throughout the code by `IDX < NUM_VARS` checks.
constexpr int NUM_VARS = 2;

class Ricci2D
{

public:

   Ricci2D(int order = 1, int ref_levels = 0, real_t xL = 1.2, real_t yL = 1.2,
           real_t max_t = 120, int nstep = 100);
   ~Ricci2D() = default;

   void Save();
   void formMKB();
   void updateDataCollection(int step);
   void updateVars();
   Coefficient * div(VectorCoefficient &vc);

   // Simulation parameters
   real_t _max_t;
   real_t _dt;
   real_t _xL;
   real_t _yL;
   int _nstep;
   int _order;
   int _ref_levels;
   int _dim = 2;
   real_t _eps2 = 1e-12;
   real_t _h;

   std::unique_ptr<ConstantCoefficient> _dt_coef;

   // Mesh
   std::unique_ptr<ParMesh> _pmesh;

   // Physical parameters
   real_t _Binv; // Factor in front of Poisson brackets
   real_t _Lambda; // Factor in the exponential term
   real_t _S_0n; // Particle source amplitude

   // System matrix and RHS
   Array2D<const mfem::HypreParMatrix *> _h_blocks;
   std::shared_ptr<HypreParMatrix> _M;
   std::shared_ptr<HypreParMatrix> _K;
   std::shared_ptr<BlockVector> _B;
   std::shared_ptr<HypreParMatrix> _M_Kdt;
   std::vector<Array<int>> _ess_tdof_lists;

   // Variables — _vars indexed by VarIdx (T, ω, n) to match the PDF derivation.
   std::vector<std::unique_ptr<ParGridFunction>> _vars;
   std::unique_ptr<ParGridFunction> _phi;
   
   std::unique_ptr<BlockVector> _var_blocks;

   Array<int> _block_offsets;

private:

   void buildMesh(real_t xL, real_t yL);
   void buildGridFunctions();
   void setInitialConditions();
   void setOutput();
   void updatePhi();
   void updateGFCoefs();
   void updateVd();
   void updateBLFCoefs();
   void updateLFCoefs();
   void makeRadialCoefficient();
   void makeSn();
   void DeleteAllBlocks();
   void ClearAllBlocks();

   // FES
   std::unique_ptr<ParFiniteElementSpace> _h1_fes;
   std::unique_ptr<ParFiniteElementSpace> _hdiv_fes;

   // Auxiliary variables and coefficients
   DenseMatrix _rotmat;
   std::unique_ptr<ParGridFunction> _hdiv_gf;
   std::unique_ptr<GradientGridFunctionCoefficient> _grad_phi;
   std::unique_ptr<MatrixConstantCoefficient> _grad_rotate;
   std::unique_ptr<MatrixVectorProductCoefficient> _vd;
   std::unique_ptr<NormalizedVectorCoefficient> _norm_vd;
   std::unique_ptr<ScalarVectorProductCoefficient> _scaled_norm_vd;
   std::unique_ptr<Coefficient> _div_vd;
   std::unique_ptr<ProductCoefficient> _div_vd_scaled;
   
   // These are just test GFs to visualise terms while developing
   // They will not be in the final version of this example
   std::unique_ptr<ParGridFunction> _vd_test_gf;
   std::unique_ptr<ParGridFunction> _Sn_test_gf;
   std::unique_ptr<ParGridFunction> _r_test_gf;
   ////////////////////////////////////////////////////////////

   // GridFunctionCoefficients for the time-evolved variables, indexed by VarIdx.
   std::vector<std::unique_ptr<GridFunctionCoefficient>> _var_gfcoefs;
   std::unique_ptr<GridFunctionCoefficient> _phi_gfcoef;

   // Coefficients for the Gateaux derivatives in the BLFs
   std::unique_ptr<Coefficient> _DnFn;
   std::unique_ptr<Coefficient> _DTFn;
   std::unique_ptr<Coefficient> _DTFT;
   std::unique_ptr<Coefficient> _DTFw;

   // Source-term coefficients F_x indexed by VarIdx.
   std::vector<std::unique_ptr<Coefficient>> _F;

   std::unique_ptr<TransformedCoefficient> _thermal_exp;
   std::unique_ptr<TransformedCoefficient> _Fn_no_source;
   std::unique_ptr<TransformedCoefficient> _FT_no_source;

   std::unique_ptr<ProductCoefficient> _n_DnFn;
   std::unique_ptr<ProductCoefficient> _T_DTFn;
   std::unique_ptr<ProductCoefficient> _T_DTFT;
   std::unique_ptr<ProductCoefficient> _T_DTFw;

   // Some useful coefficients for the linear forms
   std::unique_ptr<Coefficient> _r;
   std::unique_ptr<Coefficient> _Sn;
   
   std::unique_ptr<MatrixCoefficient> _SUW_matcoef;
      
   // Data collection
   std::unique_ptr<ParaViewDataCollection> _dc;
};

class FE_Evolution : public TimeDependentOperator
{

public:

   FE_Evolution(std::shared_ptr<HypreParMatrix> M_Kdt, std::shared_ptr<BlockVector> b,
                Array<int> block_offsets, MPI_Comm comm);

   void updateMKB(std::shared_ptr<HypreParMatrix> M_Kdt, std::shared_ptr<BlockVector> b);
   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &y) override;
   void SetTimeStep(real_t dt);

private:

   std::shared_ptr<HypreParMatrix> _M_Kdt;
   std::shared_ptr<BlockVector> _b;
   std::unique_ptr<SuperLURowLocMatrix> _SA;
   mutable BlockVector _z;
   std::unique_ptr<Solver> _M_prec;
   SuperLUSolver _M_solver;
   Array<int> _block_offsets;
   real_t _dt;

};


