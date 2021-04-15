#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>

using namespace std;
using namespace mfem;

void E_exact(const Vector &xvec, Vector &E)
{
   double x=xvec[0], y=xvec[1];
   constexpr double pi = M_PI;

   E[0] = sin(2*pi*x)*sin(4*pi*y);
   E[1] = sin(4*pi*x)*sin(2*pi*y);
}

void f_exact(const Vector &xvec, Vector &f)
{
   double x=xvec[0], y=xvec[1];
   constexpr double pi = M_PI;
   constexpr double pi2 = M_PI*M_PI;

   f[0] = 8*pi2*cos(4*pi*x)*cos(2*pi*y) + (1 + 16*pi2)*sin(2*pi*x)*sin(4*pi*y);
   f[1] = 8*pi2*cos(2*pi*x)*cos(4*pi*y) + (1 + 16*pi2)*sin(4*pi*x)*sin(2*pi*y);
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 0;
   int order = 3;
   const char *fe = "n";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&fe, "-fe", "--fe-type", "FE type. n for Hcurl, r for Hdiv");
   args.ParseCheck();

   bool ND;
   if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { ND = false; }
   else { MFEM_ABORT("Bad FE type. Must be 'n' or 'r'."); }

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   int b1 = BasisType::GaussLobatto, b2 = BasisType::Integrated;
   unique_ptr<FiniteElementCollection> fec;
   if (ND) { fec.reset(new ND_FECollection(order, dim, b1, b2)); }
   else { fec.reset(new RT_FECollection(order-1, dim, b1, b2)); }

   FiniteElementSpace fes(&mesh, fec.get());
   Array<int> ess_dofs_ho;
   fes.GetBoundaryTrueDofs(ess_dofs_ho);

   BilinearForm a_ho(&fes);
   a_ho.AddDomainIntegrator(new VectorFEMassIntegrator);
   if (ND) { a_ho.AddDomainIntegrator(new CurlCurlIntegrator); }
   else { a_ho.AddDomainIntegrator(new DivDivIntegrator); }
   a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_ho.Assemble();

   LinearForm b(&fes);
   VectorFunctionCoefficient f_coeff(dim, f_exact);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.Assemble();

   GridFunction x(&fes);
   x = 0.0;

   Vector X, B;
   OperatorHandle A;
   a_ho.FormLinearSystem(ess_dofs_ho, x, b, A, X, B);

   LORSolver<UMFPackSolver> lor_solver(a_ho, ess_dofs_ho);

   CGSolver cg;
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);
   cg.SetOperator(*A);
   cg.SetPreconditioner(lor_solver);
   cg.Mult(B, X);
   a_ho.RecoverFEMSolution(X, b, x);

   VectorFunctionCoefficient exact_coeff(dim, E_exact);
   double er = x.ComputeL2Error(exact_coeff);
   std::cout << "L^2 error: " << er << '\n';

   ParaViewDataCollection dc("LOR", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("u", &x);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   return 0;
}
