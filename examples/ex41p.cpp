#include "ex41p.hpp"

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
   int nstep = 50;
   real_t max_t = 2.4;
   real_t xL = 1.2;
   real_t yL = 1.2;

   // Create the Ricci2D problem and obtain initial values
   Ricci2D ricci(order, ref_levels, xL, yL, max_t, nstep);
   ricci.formMKB();
   ricci.updateDataCollection(0);
   ricci.Save();

   // Create the time evolution operator
   FE_Evolution t_op(*ricci._M, *ricci._K, ricci._B.get(), ricci._block_offsets,
                     ricci._pmesh->GetComm());

   // Set up the ODE solver
   std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(2);
   ode_solver->Init(t_op);

   real_t t = 0.0;

   // Time evolution
   for (int s = 0; s < ricci._nstep; ++s)
   {

      if (s!=0)
      {
         ricci.formMKB();
         t_op.updateMKB(*ricci._M, *ricci._K, ricci._B.get());
      }

      ode_solver->Step(*ricci._var_blocks, t, ricci._dt);
      ricci.updateVars();
      std::cout << "Omega Norml2 = " << ricci._omega->Norml2() << std::endl;
      std::cout << "T Norml2 = "     << ricci._T->Norml2() << std::endl;
      std::cout << "n Norml2 = "     << ricci._n->Norml2() << std::endl;
      std::cout << "Phi Norml2 = "   << ricci._phi->Norml2() << std::endl;

      ricci.updateDataCollection(s+1);
      ricci.Save();
   }


   return 0;
}



// Ricci2D methods

Ricci2D::Ricci2D(int order, int ref_levels, real_t xL, real_t yL, real_t max_t,
                 int nstep) :
   _max_t(max_t),
   _nstep(nstep),
   _order(order),
   _ref_levels(ref_levels),
   _xL(xL),
   _yL(yL),
   _Binv(40.0),
   _Lambda(3.0),
   _rotmat({{0, -1},{1, 0}}),
_grad_rotate{std::make_unique<MatrixConstantCoefficient>(_rotmat)}
{
   _dt = _max_t / _nstep;
   buildMesh(xL, yL);
   buildGridFunctions();
   makeRadialCoefficient();
   makeSn();
   setInitialConditions();
   setOutput();

   HYPRE_BigInt size = _h1_fes->GlobalTrueVSize();
   if (Mpi::WorldRank() == 0)
   {
      std::cout << "Number of finite element unknowns: " << size << std::endl;
   }
}

void Ricci2D::makeRadialCoefficient()
{
   // Make a radial coefficient whose origin is at the centre of the mesh
   _r.reset(new TransformedCoefficient(
               new CartesianXCoefficient,
               new CartesianYCoefficient,
   [this](real_t x, real_t y) { return sqrt(pow((x-_xL/2),2)+pow((y-_yL/2),2));}
            ));
}

void Ricci2D::makeSn()
{
   // Source term S_n, which in the normalised equations is equal to S_T

   real_t r_unit = 0.5;

   _Sn.reset(new TransformedCoefficient(
                _r.get(),
                [r_unit](real_t r)
   {
      real_t r_norm = r/r_unit;
      return 0.03*(1-tanh( (r_norm-20)/0.5 ));
   }
             ));
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
   Array<int> ess_tdof_list,
         ess_bdr(_phi->ParFESpace()->GetParMesh()->bdr_attributes.Max());
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
}

void Ricci2D::updateThermalExponential()
{
   _phi_gfcoef.reset(new GridFunctionCoefficient(_phi.get()));
   _T_gfcoef.reset(new GridFunctionCoefficient(_T.get()));

   _thermal_exp.reset(new TransformedCoefficient(
                         _phi_gfcoef.get(),
                         _T_gfcoef.get(),
   [this](real_t phi, real_t T) { return exp(_Lambda - phi / sqrt(pow(T,2)+_eps2)); }
                      ));

}

void Ricci2D::updateLFCoefs()
{
   _poisson_omega.reset(poissonBracket(*_omega));
   _poisson_T.reset(poissonBracket(*_T));
   _poisson_n.reset(poissonBracket(*_n));

   // Omega term
   // Check potential sign issues!

   _omega_LF_coef.reset(new TransformedCoefficient(
                           _thermal_exp.get(),
                           _poisson_omega.get(),
                           [this](real_t exp_term, real_t poisson)
   {
      return _Binv.constant * poisson + (1./24.) * (1-exp_term);
   }
                        ));

   // T term
   _T_LF_coef.reset(new TransformedCoefficient(
                       new TransformedCoefficient(_thermal_exp.get(),
                                                  _T_gfcoef.get(),
                                                  [this] (real_t thermal_exp, real_t T)
   {
      return (1.71 * thermal_exp - 0.71) * T;
   }),
   new SumCoefficient(*_poisson_T, *_Sn, _Binv.constant, 1.0),
   [this](real_t thermal_term, real_t poisson_source)
   {
      return poisson_source - (1./36.) * thermal_term;
   }
                    ));

   // n term
   _n_gfcoef.reset(new GridFunctionCoefficient(_n.get()));

   _n_LF_coef.reset(new TransformedCoefficient(
                       new ProductCoefficient(*_thermal_exp, *_n_gfcoef),
                       new SumCoefficient(*_poisson_n, *_Sn, _Binv.constant, 1.0),
                       [this](real_t thermal_term, real_t poisson_source)
   {
      return poisson_source - (1./24.) * thermal_term;
   }
                    ));
}

void Ricci2D::updateVars()
{

   *_omega = _var_blocks->GetBlock(0);
   *_T = _var_blocks->GetBlock(1);
   *_n = _var_blocks->GetBlock(2);
}

Coefficient * Ricci2D::div(VectorCoefficient &vc)
{
   _hdiv_gf->ProjectCoefficient(vc);
   return new DivergenceGridFunctionCoefficient(_hdiv_gf.get());
}

Coefficient * Ricci2D::poissonBracket(GridFunction &gf)
{
   GridFunctionCoefficient _gfcoef(&gf);
   ScalarVectorProductCoefficient gf_vd(_gfcoef, *_vd);
   return div(gf_vd);
}

void Ricci2D::formMKB()
{
   updatePhi();
   updateVd();
   updateThermalExponential();
   updateLFCoefs();

   // Make system blocks
   _block_offsets.SetSize(4); // number of variables + 1
   _block_offsets[0] = 0;
   _block_offsets[1] = _h1_fes->GetVSize();
   _block_offsets[2] = _h1_fes->GetVSize();
   _block_offsets[3] = _h1_fes->GetVSize();
   _block_offsets.PartialSum();

   _M.reset(new BlockOperator(_block_offsets));
   _K.reset(new BlockOperator(_block_offsets));
   _B.reset(new BlockVector(_block_offsets));

   _M->owns_blocks = true;
   _K->owns_blocks = true;

   _var_blocks.reset(new BlockVector(_block_offsets));
   _var_blocks->GetBlock(0) = *_omega;
   _var_blocks->GetBlock(1) = *_T;
   _var_blocks->GetBlock(2) = *_n;

   // Omega terms

   ParBilinearForm * m_00 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_00 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_0 = new ParLinearForm(_h1_fes.get());

   m_00->AddDomainIntegrator(new MassIntegrator);
   m_00->Assemble();
   m_00->Finalize();

   k_00->Assemble();
   k_00->Finalize();

   b_0->AddDomainIntegrator(new DomainLFIntegrator(*_omega_LF_coef));
   b_0->Assemble();

   _M->SetBlock(0, 0, m_00);
   _K->SetBlock(0, 0, k_00);
   _B->GetBlock(0) = *b_0;

   // T terms

   ParBilinearForm * m_11 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_11 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_1 = new ParLinearForm(_h1_fes.get());

   m_11->AddDomainIntegrator(new MassIntegrator);
   m_11->Assemble();
   m_11->Finalize();

   k_11->Assemble();
   k_11->Finalize();

   b_1->AddDomainIntegrator(new DomainLFIntegrator(*_T_LF_coef));
   b_1->Assemble();

   _M->SetBlock(1, 1, m_11);
   _K->SetBlock(1, 1, k_11);
   _B->GetBlock(1) = *b_1;

   // n terms

   ParBilinearForm * m_22 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_22 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_2 = new ParLinearForm(_h1_fes.get());

   m_22->AddDomainIntegrator(new MassIntegrator);
   m_22->Assemble();
   m_22->Finalize();

   k_22->Assemble();
   k_22->Finalize();

   b_2->AddDomainIntegrator(new DomainLFIntegrator(*_n_LF_coef));
   b_2->Assemble();

   _M->SetBlock(2, 2, m_22);
   _K->SetBlock(2, 2, k_22);
   _B->GetBlock(2) = *b_2;
}

void Ricci2D::buildMesh(real_t xL, real_t yL)
{
   Mesh mesh = Mesh::MakeCartesian2D(64, 64, Element::QUADRILATERAL, true, xL, yL,
                                     false);
   for (int l = 0; l < _ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   _pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, mesh);
}

void Ricci2D::buildGridFunctions()
{
   _dim = _pmesh->Dimension();
   _h1_fes = std::make_unique<ParFiniteElementSpace>(_pmesh.get(),
                                                     new H1_FECollection(_order, _dim));
   _hdiv_fes = std::make_unique<ParFiniteElementSpace>(_pmesh.get(),
                                                       new RT_FECollection(_order, _dim));

   _phi = std::make_unique<ParGridFunction>(_h1_fes.get());
   _omega = std::make_unique<ParGridFunction>(_h1_fes.get());
   _T = std::make_unique<ParGridFunction>(_h1_fes.get());
   _n = std::make_unique<ParGridFunction>(_h1_fes.get());

   // Intermediate gf for computing the divergence of v_d
   _hdiv_gf = std::make_unique<ParGridFunction>(_hdiv_fes.get());

   // List is empty for pure Neumann BCs
   _ess_tdof_lists.resize(2);
}

void Ricci2D::setInitialConditions()
{
   *_phi = 0.03;
   *_omega = 0.0;
   *_T = 1.0e-4;
   *_n = 1.0e-4;

}

void Ricci2D::setOutput()
{
   _visit_dc = std::make_unique<ParaViewDataCollection>("Ricci2D/Step",
                                                        _pmesh.get());
   _visit_dc->RegisterField("phi", _phi.get());
   _visit_dc->RegisterField("omega", _omega.get());
   _visit_dc->RegisterField("T", _T.get());
   _visit_dc->RegisterField("n", _n.get());
}

void Ricci2D::updateDataCollection(int step)
{

   _visit_dc->SetCycle(step);
   _visit_dc->SetTime(step*_dt);
}


// FE_Evolution methods

FE_Evolution::FE_Evolution(BlockOperator &M, BlockOperator &K, BlockVector * b,
                           Array<int> block_offsets, MPI_Comm comm)
   : TimeDependentOperator(M.Height(), M.Width()),
     _z(block_offsets),
     _block_offsets(block_offsets),
     _M_solver(comm)
{
   updateMKB(M, K, b);
}

void FE_Evolution::updateMKB(BlockOperator &M, BlockOperator &K,
                             BlockVector * b)
{
   _M = &M;
   _K = &K;
   _b = b;
   _M_solver.SetOperator(*_M);

   _M_prec = new BlockDiagonalPreconditioner(_block_offsets);

   _M_solver.SetPreconditioner(*_M_prec);
   _M_solver.SetRelTol(1e-9);
   _M_solver.SetAbsTol(0.0);
   _M_solver.SetMaxIter(100);
   _M_solver.SetPrintLevel(0);

}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   _K->Mult(x, _z);
   _z += *_b;
   _M_solver.Mult(_z, y);
}

FE_Evolution::~FE_Evolution()
{
   delete _M_prec;
}