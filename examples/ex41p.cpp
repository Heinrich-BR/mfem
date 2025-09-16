#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
    // Initialise MPI and device
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();
    Device device("cpu");
    if (myid == 0) { device.Print(); }

    // Make mesh and FES
    Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true, 1.2, 1.2, false);
    int dim = mesh.Dimension();
    int order = 1;
    int ref_levels = 0;

    for (int l = 0; l < ref_levels; l++)
       mesh.UniformRefinement();
   
    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    ParFiniteElementSpace h1_fes(&pmesh, new H1_FECollection(order, dim));

    HYPRE_BigInt size = h1_fes.GlobalTrueVSize();
    if (myid == 0)
       std::cout << "Number of finite element unknowns: " << size << std::endl;

    // Make gridfunctions
    ParGridFunction phi(&h1_fes);
    ParGridFunction omega(&h1_fes);
    phi = 0.03;
    omega = 0.0;

    // Boundary conditions
    Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 1;
    h1_fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Assemble the equation system
    ParBilinearForm a(&h1_fes);
    ConstantCoefficient one(1.0);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    ParLinearForm b(&h1_fes);
    GridFunctionCoefficient omega_coef(&omega);
    b.AddDomainIntegrator(new DomainLFIntegrator(omega_coef));
    b.Assemble();

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, phi, b, A, X, B);

    // Solve
    HypreBoomerAMG prec;
    CGSolver cg(MPI_COMM_WORLD);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(2000);
    cg.SetPrintLevel(1);
    if (prec) { cg.SetPreconditioner(prec); }
    cg.SetOperator(*A);
    cg.Mult(B, X);
    a.RecoverFEMSolution(X, b, phi);

    // Save solution
    VisItDataCollection visit_dc("Ricci2D", &pmesh);
    visit_dc.RegisterField("phi", &phi);
    visit_dc.Save();

    return 0;
}