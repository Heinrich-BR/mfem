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

    // Physical parameters
    ConstantCoefficient _Binv; // Factor in front of Poisson brackets
    real_t _Lambda; // Factor in the exponential term
    real_t _kh3; // Factor in front of the third term containing the thermal exponential

    // Matrices
    std::unique_ptr<ParBilinearForm> _m;
    std::unique_ptr<ParBilinearForm> _k;
    std::unique_ptr<ParLinearForm> _b;

    // Variables
    std::unique_ptr<ParGridFunction> _omega;
    std::unique_ptr<ParGridFunction> _phi;
    std::unique_ptr<ParGridFunction> _T;

private:

    void buildMesh(real_t xL, real_t yL);
    void buildGridFunctions();
    void setInitialConditions();
    void setOutput();
    void updateThermalExponential();
    
    // Mesh and FES
    std::unique_ptr<ParMesh> _pmesh;
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
    std::unique_ptr<SumCoefficient> _KH3_coef;

    // Data collection
    std::unique_ptr<ParaViewDataCollection> _visit_dc;
};
class FE_Evolution : public TimeDependentOperator
{

public:

   FE_Evolution(ParBilinearForm &M, ParBilinearForm &K, Vector * b);
   ~FE_Evolution() override;

   void updateMKB(ParBilinearForm &M, ParBilinearForm &K, Vector * b);
   void Mult(const Vector &x, Vector &y) const override;
   

private:

   OperatorHandle _M, _K;
   Vector * _b;
   mutable Vector _z;
   Solver *_M_prec;
   CGSolver _M_solver;
   
};


int main(int argc, char *argv[])
{
    // Initialise MPI and device
    Mpi::Init();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    Device device("cpu");
    if (myid == 0) { device.Print(); }

    // Simulation parameters
    int order = 1;
    int ref_levels = 0;
    int nstep = 10;
    real_t max_t = 12;
    real_t xL = 1.2;
    real_t yL = 1.2;
    
    // Create the Ricci2D problem and obtain initial values
    Ricci2D ricci(order, ref_levels, xL, yL, max_t, nstep);
    ricci.updatePhi();
    ricci.formMKB();
    ricci.updateDataCollection(0);
    ricci.Save();

    // Create the time evolution operator
    FE_Evolution t_op(*ricci._m, *ricci._k, ricci._b.get());

    // Set up the ODE solver
    std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(1);
    ode_solver->Init(t_op);

    real_t t = 0.0;

    // Time evolution
    for (int s = 0; s < ricci._nstep; ++s)
    {

        if (s!=0)
        {
            ricci.updatePhi();
            ricci.formMKB();
            t_op.updateMKB(*ricci._m, *ricci._k, ricci._b.get());
        }

        ode_solver->Step(*ricci._omega, t, ricci._dt);

        ricci.updateDataCollection(s+1);
        ricci.Save();
    }


    return 0;
}


// Ricci2D methods

Ricci2D::Ricci2D(int order, int ref_levels, real_t xL, real_t yL, real_t max_t, int nstep) : 
            _max_t(max_t), 
            _nstep(nstep),
            _order(order),
            _ref_levels(ref_levels),
            _Binv(40.0),
            _Lambda(3.0),
            _kh3(1.0/24.0),
            _rotmat({{0, -1},{1, 0}}),
            _grad_rotate{std::make_unique<MatrixConstantCoefficient>(_rotmat)}
{
    _dt = _max_t / _nstep;
    buildMesh(xL, yL);
    buildGridFunctions();
    setInitialConditions();
    setOutput();

    HYPRE_BigInt size = _h1_fes->GlobalTrueVSize();
    if (Mpi::WorldRank() == 0)
        std::cout << "Number of finite element unknowns: " << size << std::endl;
}

void Ricci2D::Save()
{
    _visit_dc->Save();
}

void Ricci2D::updateVd()
{
    _grad_phi.reset(new GradientGridFunctionCoefficient(_phi.get()));
    _vd.reset(new MatrixVectorProductCoefficient(*_grad_rotate, *_grad_phi));
}

void Ricci2D::updatePhi()
{
    // Boundary conditions
    Array<int> ess_tdof_list, ess_bdr(_phi->ParFESpace()->GetParMesh()->bdr_attributes.Max());
    ess_bdr = 1;
    _h1_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Assemble the equation system
    ParBilinearForm a(_h1_fes.get());
    ConstantCoefficient one(1.0);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    ParLinearForm b(_h1_fes.get());
    GridFunctionCoefficient _omega_coef(_omega.get());
    b.AddDomainIntegrator(new DomainLFIntegrator(_omega_coef));
    b.Assemble();
    OperatorPtr _A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, *_phi, b, _A, X, B);

    // Solve
    HypreBoomerAMG _prec;
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(2000);
    cg.SetPrintLevel(1);
    if (_prec) { cg.SetPreconditioner(_prec); }
    cg.SetOperator(*_A);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, *_phi);

    updateVd();
    updateThermalExponential();
}

void Ricci2D::updateThermalExponential()
{
    _phi_gfcoef.reset(new GridFunctionCoefficient(_phi.get()));
    _T_gfcoef.reset(new GridFunctionCoefficient(_T.get()));

    _thermal_exp.reset(new TransformedCoefficient(
    _phi_gfcoef.get(),
    _T_gfcoef.get(),
    [this](real_t phi, real_t T) { return exp(_Lambda - phi / T); }
    ));

    _KH3_coef.reset(new SumCoefficient(_one, *_thermal_exp, -_kh3, _kh3));
}

DivergenceGridFunctionCoefficient * Ricci2D::div(VectorCoefficient &vc)
{
    _hdiv_gf->ProjectCoefficient(vc);
    return new DivergenceGridFunctionCoefficient(_hdiv_gf.get());
}

void Ricci2D::formMKB()
{
    _m.reset(new ParBilinearForm(_h1_fes.get()));
    _k.reset(new ParBilinearForm(_h1_fes.get()));
    _b.reset(new ParLinearForm(_h1_fes.get()));

    _b->AddDomainIntegrator(new DomainLFIntegrator(*_KH3_coef));
    _b->Assemble();

    _m->AddDomainIntegrator(new MassIntegrator);
    _m->Assemble();
    _m->Finalize();

    _k->AddDomainIntegrator(new ConvectionIntegrator(*_vd, _Binv.constant));
    _k->AddDomainIntegrator(new MassIntegrator(*div(*_vd)));
    _k->Assemble();
    _k->Finalize();
}

void Ricci2D::buildMesh(real_t xL, real_t yL)
{
    Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true, xL, yL, false);
    for (int l = 0; l < _ref_levels; l++)
        mesh.UniformRefinement();

    _pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
}

void Ricci2D::buildGridFunctions()
{
    _dim = _pmesh->Dimension();
    _h1_fes = std::make_unique<ParFiniteElementSpace>(_pmesh.get(), new H1_FECollection(_order, _dim));
    _hdiv_fes = std::make_unique<ParFiniteElementSpace>(_pmesh.get(), new RT_FECollection(_order, _dim));

    _phi = std::make_unique<ParGridFunction>(_h1_fes.get());
    _omega = std::make_unique<ParGridFunction>(_h1_fes.get());
    _T = std::make_unique<ParGridFunction>(_h1_fes.get());

    // Intermediate gf for computing the divergence of v_d
    _hdiv_gf = std::make_unique<ParGridFunction>(_hdiv_fes.get());
}

void Ricci2D::setInitialConditions()
{
    *_phi = 0.03;
    *_omega = 0.0;
    *_T = 1.0e-4;
}

void Ricci2D::setOutput()
{
    _visit_dc = std::make_unique<ParaViewDataCollection>("Ricci2D/Step", _pmesh.get());
    _visit_dc->RegisterField("phi", _phi.get());
    _visit_dc->RegisterField("omega", _omega.get());
    _visit_dc->RegisterField("T", _T.get());
}

void Ricci2D::updateDataCollection(int step){

    _visit_dc->SetCycle(step);
    _visit_dc->SetTime(step*_dt);
}


// FE_Evolution methods

FE_Evolution::FE_Evolution(ParBilinearForm &M, ParBilinearForm &K, Vector * b)
   : TimeDependentOperator(M.ParFESpace()->GetTrueVSize()), _b(b),
     _M_solver(M.ParFESpace()->GetComm()),
     _z(height)
{
    updateMKB(M, K, b);
}

void FE_Evolution::updateMKB(ParBilinearForm &M, ParBilinearForm &K, Vector * b)
{
    if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
    {
       _M.Reset(M.ParallelAssemble(), true);
       _K.Reset(K.ParallelAssemble(), true);
    }
    else
    {
       _M.Reset(&M, false);
       _K.Reset(&K, false);
    }

    _M_solver.SetOperator(*_M);

    Array<int> ess_tdof_list;
    if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
    {
       HypreParMatrix &M_mat = *_M.As<HypreParMatrix>();
       HypreParMatrix &K_mat = *_K.As<HypreParMatrix>();
       HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
       _M_prec = hypre_prec;
    }
    else
    {
       _M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
    }

    _M_solver.SetPreconditioner(*_M_prec);
    _M_solver.SetRelTol(1e-9);
    _M_solver.SetAbsTol(0.0);
    _M_solver.SetMaxIter(100);
    _M_solver.SetPrintLevel(0);

    _b = b;


}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    // y = M^{-1} (K x + b)
    _K->Mult(x, _z);
    _z += *_b;
    _M_solver.Mult(_z, y);
    std::cout << "domega/dt norml2 = " << y.Norml2() << std::endl;
}

FE_Evolution::~FE_Evolution()
{
   delete _M_prec;
}