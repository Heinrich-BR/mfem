#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

class Ricci2D
{

public:

    Ricci2D(int order = 1, int ref_levels = 0, real_t xL = 1.2, real_t yL = 1.2, real_t max_t = 120, int nstep = 100);
    ~Ricci2D() = default;

    void Save();
    void updateVd();
    void updatePhi();
    void formMKB();
    void updateDataCollection(int step);
    DivergenceGridFunctionCoefficient * div(VectorCoefficient &vc);
    
    // Simulation parameters
    real_t _max_t;
    real_t _dt;
    int _nstep;
    int _order;
    int _ref_levels;
    int _dim = 2;

    // Mesh
    std::unique_ptr<ParMesh> _pmesh;

    // Physical parameters
    ConstantCoefficient _Binv; // Factor in front of Poisson brackets
    real_t _Lambda; // Factor in the exponential term
    real_t _kh3; // Factor in front of the third term containing the thermal exponential

    // System matrix and RHS
    std::unique_ptr<BlockOperator> _M;
    std::unique_ptr<BlockOperator> _K;
    std::unique_ptr<BlockVector> _B;
    std::vector<Array<int>> _ess_tdof_lists;

    // Variables
    std::unique_ptr<ParGridFunction> _phi;
    std::unique_ptr<ParGridFunction> _omega;
    std::unique_ptr<ParGridFunction> _T;

    std::unique_ptr<ParGridFunction> _dt_omega;
    std::unique_ptr<ParGridFunction> _dt_T;

    Array<int> _block_offsets;

private:

    void buildMesh(real_t xL, real_t yL);
    void buildGridFunctions();
    void setInitialConditions();
    void setOutput();
    void updateThermalExponential();
    
    // FES
    std::unique_ptr<ParFiniteElementSpace> _h1_fes;
    std::unique_ptr<ParFiniteElementSpace> _hdiv_fes;

    // Auxiliary variables and coefficients
    DenseMatrix _rotmat;
    ConstantCoefficient _one{1.0};

    std::unique_ptr<ParGridFunction> _hdiv_gf;
    std::unique_ptr<GradientGridFunctionCoefficient> _grad_phi;
    std::unique_ptr<MatrixConstantCoefficient> _grad_rotate;
    std::unique_ptr<MatrixVectorProductCoefficient> _vd;
    
    std::unique_ptr<GridFunctionCoefficient> _phi_gfcoef;
    std::unique_ptr<GridFunctionCoefficient> _T_gfcoef;

    std::unique_ptr<TransformedCoefficient> _thermal_exp;
    std::unique_ptr<Coefficient> _KH3_coef;

    // Data collection
    std::unique_ptr<ParaViewDataCollection> _visit_dc;
};

class FE_Evolution : public TimeDependentOperator
{

public:

   FE_Evolution(BlockOperator &M, BlockOperator &K, BlockVector * b, Array<int> block_offsets, MPI_Comm comm);
   ~FE_Evolution() override;

   void updateMKB(BlockOperator &M, BlockOperator &K, BlockVector * b);
   void Mult(const Vector &x, Vector &y) const override;
   
private:

   BlockOperator * _M;
   BlockOperator * _K;
   BlockVector * _b;
   mutable BlockVector _z;
   Solver *_M_prec;
   CGSolver _M_solver;
   Array<int> _block_offsets;
   
};


