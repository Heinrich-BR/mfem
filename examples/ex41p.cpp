#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

class Ricci2D
{

public:

    Ricci2D(int order = 1, int ref_levels = 0, real_t xL = 1.2, real_t yL = 1.2, real_t max_t = 120, int nt = 100) : 
            _max_t(max_t), 
            _nt(nt),
            _Binv(40.0),
            _rotmat({{0, -1},{1, 0}}),
            _grad_rotate{std::make_unique<MatrixConstantCoefficient>(_rotmat)}
    {
        _dt = _max_t / _nt;
        buildMesh(ref_levels, xL, yL);
        buildGridFunctions(order);
        setInitialConditions();
        setOutput();

        HYPRE_BigInt size = _h1_fes->GlobalTrueVSize();
        if (Mpi::WorldRank() == 0)
            std::cout << "Number of finite element unknowns: " << size << std::endl;
    }

    ~Ricci2D() = default;

    void Save()
    {
        _visit_dc->Save();
    }

    void updateVd()
    {
        _grad_phi.reset(new GradientGridFunctionCoefficient(_phi.get()));
        _vd.reset(new MatrixVectorProductCoefficient(*_grad_rotate, *_grad_phi));
    }

    void updatePhi()
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

        OperatorPtr A;
        Vector B, X;
        a.FormLinearSystem(ess_tdof_list, *_phi, b, A, X, B);

        // Solve
        HypreBoomerAMG prec;
        CGSolver cg(MPI_COMM_WORLD);
        cg.SetRelTol(1e-12);
        cg.SetMaxIter(2000);
        cg.SetPrintLevel(1);
        if (prec) { cg.SetPreconditioner(prec); }
        cg.SetOperator(*A);
        cg.Mult(B, X);
        a.RecoverFEMSolution(X, b, *_phi);

        updateVd();
    }

    void formMKB()
    {
        _m.reset(new ParBilinearForm(_h1_fes.get()));
        _k.reset(new ParBilinearForm(_h1_fes.get()));
        _b.reset(new ParLinearForm(_h1_fes.get()));

        _m->AddDomainIntegrator(new MassIntegrator);
        _m->Assemble();
        _m->Finalize();

        _k->AddDomainIntegrator(new ConvectionIntegrator(*_vd, _Binv.constant));
        //_k->AddDomainIntegrator(new MassIntegrator(new DivergenceGridFunctionCoefficient(*_vd)))
        _k->Assemble();
        _k->Finalize();



    }

    real_t _max_t;
    int _nt;
    real_t _dt;
    ConstantCoefficient _Binv;

private:

    // Mesh and FES
    std::unique_ptr<ParMesh> _pmesh;
    std::unique_ptr<ParFiniteElementSpace> _h1_fes;

    // Variables
    std::unique_ptr<ParGridFunction> _phi;
    std::unique_ptr<ParGridFunction> _omega;
    //std::unique_ptr<ParGridFunction> _domega_dt;

    DenseMatrix _rotmat;
    std::unique_ptr<GradientGridFunctionCoefficient> _grad_phi;
    std::unique_ptr<MatrixConstantCoefficient> _grad_rotate;
    std::unique_ptr<MatrixVectorProductCoefficient> _vd;

    // Forms
    std::unique_ptr<ParBilinearForm> _m;
    std::unique_ptr<ParBilinearForm> _k;
    std::unique_ptr<ParLinearForm> _b;

    // Data collection
    std::unique_ptr<VisItDataCollection> _visit_dc;

    // Methods

    void buildMesh(int ref_levels = 0, real_t xL = 1.2, real_t yL = 1.2)
    {
        Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true, xL, yL, false);
        for (int l = 0; l < ref_levels; l++)
            mesh.UniformRefinement();
   
        _pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
    }

    void buildGridFunctions(int order = 1)
    {
        int dim = _pmesh->Dimension();
        _h1_fes = std::make_unique<ParFiniteElementSpace>(_pmesh.get(), new H1_FECollection(order, dim));
        _phi = std::make_unique<ParGridFunction>(_h1_fes.get());
        _omega = std::make_unique<ParGridFunction>(_h1_fes.get());
    }

    void setInitialConditions()
    {
        *_phi = 0.03;
        *_omega = 0.0;
    }

    void setOutput()
    {
        _visit_dc = std::make_unique<VisItDataCollection>("Ricci2D", _pmesh.get());
        _visit_dc->RegisterField("_phi", _phi.get());
        _visit_dc->RegisterField("_omega", _omega.get());
    }

};


int main(int argc, char *argv[])
{
    // Initialise MPI and device
    Mpi::Init();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    Device device("cpu");
    if (myid == 0) { device.Print(); }

    Ricci2D ricci;

    for (int t = 0; t < ricci._nt; ++t)
    {
        ricci.updatePhi();

    }

    




    ricci.Save();

    return 0;
}