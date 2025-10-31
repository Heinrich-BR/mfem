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
   int order = 2;
   int ref_levels = 1;
   real_t xL = 1.0; // Lengths measured in units of 100*rho_s0
   real_t yL = 1.0;

   int nstep = 100;
   real_t max_t = 3.0;


   // Create the Ricci2D problem and obtain initial values
   Ricci2D ricci(order, ref_levels, xL, yL, max_t, nstep);
   ricci.formMKB();
   ricci.updateDataCollection(0);
   ricci.Save();

   // Create the time evolution operator
   FE_Evolution t_op(ricci._M, ricci._K, ricci._B, ricci._block_offsets,
                     ricci._pmesh->GetComm());

   // Set up the ODE solver
   std::unique_ptr<ODESolver> ode_solver = ODESolver::Select(21);
   ode_solver->Init(t_op);

   real_t t = 0.0;

   // Time evolution
   for (int s = 0; s < ricci._nstep; ++s)
   {

      if (s!=0)
      {
         ricci.formMKB();
         t_op.updateMKB(ricci._M, ricci._K, ricci._B);
      }

      ode_solver->Step(*ricci._var_blocks, t, ricci._dt);
      ricci.updateVars();
      std::cout << "--------------------------" << std::endl;
      std::cout << "t = " << t << std::endl;
      std::cout << "--------------------------" << std::endl;
      std::cout << "Omega Norml2 = " << ricci._omega->Norml2() << std::endl;
      std::cout << "T Norml2     = "     << ricci._T->Norml2() << std::endl;
      std::cout << "n Norml2     = "     << ricci._n->Norml2() << std::endl;
      std::cout << "Phi Norml2   = "   << ricci._phi->Norml2() << std::endl;
      std::cout << "---------------------------" << std::endl;

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
   _S_0n(0.03),
   _rotmat({{0, -1},{1, 0}}),
   _grad_rotate{std::make_unique<MatrixConstantCoefficient>(_rotmat)}
{
   _dt = _max_t / _nstep;
   _dt_coef = std::make_unique<ConstantCoefficient>(_dt);
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
   
   _r_test_gf = std::make_unique<ParGridFunction>(_h1_fes.get());
   _r_test_gf->ProjectCoefficient(*_r);
}

void Ricci2D::makeSn()
{
   // Source term S_n, which in the normalised equations is equal to S_T

   real_t r_unit = 1e-2; // 100 rho_s0 in normalised units

   _Sn.reset(new TransformedCoefficient(
                _r.get(),
                [r_unit, this](real_t r)
   {
      real_t r_norm = r/r_unit;
      return _S_0n*(1-tanh( (r_norm-20)/0.5 ))/2.0;
   }
             ));

   _Sn_test_gf->ProjectCoefficient(*_Sn);
}

void Ricci2D::Save()
{
   _dc->Save();
}

void Ricci2D::updateVd()
{
   _grad_phi.reset(new GradientGridFunctionCoefficient(_phi.get()));
   _vd.reset(new MatrixVectorProductCoefficient(*_grad_rotate, *_grad_phi));

   _norm_vd.reset(new NormalizedVectorCoefficient(*_vd, _eps2));
   _scaled_norm_vd.reset(new ScalarVectorProductCoefficient(_h/2.0, *_norm_vd));
   _div_vd.reset(div(*_vd));
   _div_vd_scaled.reset(new ProductCoefficient(-_Binv, *_div_vd));
   _vd_test_gf->ProjectCoefficient(*_vd);
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
   ProductCoefficient _neg_omega_coef(-1.0, _omega_coef);
   b.AddDomainIntegrator(new DomainLFIntegrator(_neg_omega_coef));
   b.Assemble();
   OperatorPtr _A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, *_phi, b, _A, X, B);

   // Solve
   HypreBoomerAMG _prec;
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(0);
   if (_prec) { cg.SetPreconditioner(_prec); }
   cg.SetOperator(*_A);
   cg.Mult(B, X);
   a.RecoverFEMSolution(X, b, *_phi);
}

void Ricci2D::updateGFCoefs()
{
   _n_gfcoef.reset(new GridFunctionCoefficient(_n.get()));
   _T_gfcoef.reset(new GridFunctionCoefficient(_T.get()));
   _omega_gfcoef.reset(new GridFunctionCoefficient(_omega.get()));
   _phi_gfcoef.reset(new GridFunctionCoefficient(_phi.get()));
}

void Ricci2D::updateBLFCoefs()
{
   // Update the thermal exponential coefficient to simplify some expressions
   _thermal_exp.reset(new TransformedCoefficient(
                 _T_gfcoef.get(), _phi_gfcoef.get(),
   [this](real_t T, real_t phi)
   { return exp(_Lambda-phi/sqrt(T*T+_eps2)); }
              ));

   // Update the streamline upwinding coefficient
   _SUW_matcoef.reset(new OuterProductCoefficient(*_vd, *_scaled_norm_vd));

   // Update the Gateaux derivative coefficients
   _DnFn.reset(new TransformedCoefficient(
                    _thermal_exp.get(),
   [this](real_t exp_term)
   { return (_dt/24.)*exp_term; }
                 ));
   
   _DTFw.reset(new TransformedCoefficient(
                    _T_gfcoef.get(), _phi_gfcoef.get(),
   [this](real_t T, real_t phi)
   { return (-_dt/24.)*exp(_Lambda-phi/sqrt(T*T+_eps2))*phi/(T*T+_eps2); }
                 ));

   _DTFn.reset(new TransformedCoefficient(
                    _n_gfcoef.get(), _DTFw.get(),
   [this](real_t n, real_t DTFw)
   { return -DTFw*n; }
                 ));
   
   _DTFT.reset(new TransformedCoefficient(
                    _T_gfcoef.get(), _phi_gfcoef.get(),
   [this](real_t T, real_t phi)
   { return (_dt/36.)*(1.71*exp(_Lambda-phi/sqrt(T*T+_eps2))-0.71)+_dt*0.0475*exp(_Lambda-phi/sqrt(T*T+_eps2))*phi/sqrt(T*T+_eps2); }
                 ));

}

void Ricci2D::updateLFCoefs()
{
   _n_DnFn.reset(new ProductCoefficient(*_n_gfcoef, *_DnFn));
   _T_DTFn.reset(new ProductCoefficient(*_T_gfcoef, *_DTFn));
   _T_DTFT.reset(new ProductCoefficient(*_T_gfcoef, *_DTFT));
   _T_DTFw.reset(new ProductCoefficient(*_T_gfcoef, *_DTFw));
   
   _Fn_no_source.reset(new TransformedCoefficient(
                 _n_gfcoef.get(), _thermal_exp.get(),
   [this](real_t n, real_t exp_term)
   { return (n/24.)*exp_term; }
              ));
   
   _Fn.reset(new TransformedCoefficient(
                 _Fn_no_source.get(), _Sn.get(),
   [this](real_t Fn_no_source, real_t Sn)
   { return Fn_no_source - Sn; }
              ));

   _FT_no_source.reset(new TransformedCoefficient(
                 _T_gfcoef.get(), _thermal_exp.get(),
   [this](real_t T, real_t exp_term)
   { return (T/36.)*(1.71*exp_term-0.71); }
              ));

   _FT.reset(new TransformedCoefficient(
                 _FT_no_source.get(), _Sn.get(),
   [this](real_t FT_no_source, real_t Sn)
   { return FT_no_source - Sn; }
              ));

   _Fw.reset(new TransformedCoefficient(
                 _thermal_exp.get(),
   [this](real_t exp_term)
   { return (1./24.)*(1.-exp_term); }
              ));

}

void Ricci2D::updateVars()
{
   *_n = _var_blocks->GetBlock(0);
   *_T = _var_blocks->GetBlock(1);
   *_omega = _var_blocks->GetBlock(2);
}

Coefficient * Ricci2D::div(VectorCoefficient &vc)
{
   _hdiv_gf->ProjectCoefficient(vc);
   return new DivergenceGridFunctionCoefficient(_hdiv_gf.get());
}

void Ricci2D::formMKB()
{
   updatePhi();
   updateVd();
   updateGFCoefs();
   updateBLFCoefs();
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
   _var_blocks->GetBlock(0) = *_n;
   _var_blocks->GetBlock(1) = *_T;
   _var_blocks->GetBlock(2) = *_omega;
   
   // Diagonal n terms

   ParBilinearForm * m_00 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_00 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_0 = new ParLinearForm(_h1_fes.get());

   m_00->AddDomainIntegrator(new MassIntegrator); // Time derivative term
   m_00->AddDomainIntegrator(new MassIntegrator(*_DnFn)); // Gateaux derivative term, already scaled by dt
   m_00->Assemble();
   m_00->Finalize();

   k_00->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv)); // Poisson bracket first term
   k_00->AddDomainIntegrator(new MassIntegrator(*_div_vd_scaled)); // Poisson bracket second term
   k_00->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef)); // SUW term
   k_00->Assemble();
   k_00->Finalize();

   b_0->AddDomainIntegrator(new DomainLFIntegrator(*_Fn));
   b_0->Assemble();

   _M->SetBlock(0, 0, m_00);
   _K->SetBlock(0, 0, k_00);
   _B->GetBlock(0) = *b_0;

   // Diagonal T terms

   ParBilinearForm * m_11 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_11 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_1 = new ParLinearForm(_h1_fes.get());

   m_11->AddDomainIntegrator(new MassIntegrator); // Time derivative term
   m_11->AddDomainIntegrator(new MassIntegrator(*_DTFT)); // Gateaux derivative term, already scaled by dt
   m_11->Assemble();
   m_11->Finalize();

   k_11->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv)); // Poisson bracket first term
   k_11->AddDomainIntegrator(new MassIntegrator(*_div_vd_scaled)); // Poisson bracket second term
   k_11->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef)); // SUW term
   k_11->Assemble();
   k_11->Finalize();

   b_1->AddDomainIntegrator(new DomainLFIntegrator(*_FT));
   b_1->Assemble();

   _M->SetBlock(1, 1, m_11);
   _K->SetBlock(1, 1, k_11);
   _B->GetBlock(1) = *b_1;

   // Diagonal Omega terms

   ParBilinearForm * m_22 = new ParBilinearForm(_h1_fes.get());
   ParBilinearForm * k_22 = new ParBilinearForm(_h1_fes.get());
   ParLinearForm * b_2 = new ParLinearForm(_h1_fes.get());

   m_22->AddDomainIntegrator(new MassIntegrator); // Time derivative term
   m_22->Assemble();
   m_22->Finalize();

   k_22->AddDomainIntegrator(new ConvectionIntegrator(*_vd, -_Binv)); // Poisson bracket first term
   k_22->AddDomainIntegrator(new MassIntegrator(*_div_vd_scaled)); // Poisson bracket second term
   k_22->AddDomainIntegrator(new DiffusionIntegrator(*_SUW_matcoef)); // SUW term
   k_22->Assemble();
   k_22->Finalize();

   b_2->AddDomainIntegrator(new DomainLFIntegrator(*_Fw));
   b_2->Assemble();

   _M->SetBlock(2, 2, m_22);
   _K->SetBlock(2, 2, k_22);
   _B->GetBlock(2) = *b_2;

   // Off-diagonal terms

   ParBilinearForm * m_01 = new ParBilinearForm(_h1_fes.get());
   m_01->AddDomainIntegrator(new MassIntegrator(*_DTFn)); // Gateaux derivative term, already scaled by dt
   m_01->Assemble();
   m_01->Finalize();
   _M->SetBlock(0, 1, m_01);

   ParBilinearForm * m_21 = new ParBilinearForm(_h1_fes.get());
   m_21->AddDomainIntegrator(new MassIntegrator(*_DTFw)); // Gateaux derivative term, already scaled by dt
   m_21->Assemble();
   m_21->Finalize();
   _M->SetBlock(2, 1, m_21);

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
   _h = _pmesh->GetElementSize(_pmesh->GetTypicalElementTransformation());
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

   _vd_test_gf = std::make_unique<ParGridFunction>(_hdiv_fes.get());
   _Sn_test_gf = std::make_unique<ParGridFunction>(_h1_fes.get());

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
   _dc = std::make_unique<ParaViewDataCollection>("Ricci2D/Step",
                                                        _pmesh.get());
   _dc->RegisterField("phi", _phi.get());
   _dc->RegisterField("omega", _omega.get());
   _dc->RegisterField("T", _T.get());
   _dc->RegisterField("n", _n.get());

   _dc->RegisterField("Vd", _vd_test_gf.get());
   _dc->RegisterField("Sn", _Sn_test_gf.get());
   _dc->RegisterField("r", _r_test_gf.get());
}

void Ricci2D::updateDataCollection(int step)
{
   _dc->SetCycle(step);
   _dc->SetTime(step*_dt);
}


// FE_Evolution methods

FE_Evolution::FE_Evolution(std::shared_ptr<BlockOperator> M, std::shared_ptr<BlockOperator> K, std::shared_ptr<BlockVector> b,
                           Array<int> block_offsets, MPI_Comm comm)
   : TimeDependentOperator(M->Height(), M->Width()),
     _z(block_offsets),
     _block_offsets(block_offsets),
     _M_solver(comm)
{
   updateMKB(M, K, b);
}

void FE_Evolution::updateMKB(std::shared_ptr<BlockOperator> M, std::shared_ptr<BlockOperator> K,
                             std::shared_ptr<BlockVector> b)
{
   _M = M;
   _K = K;
   _b = b;
   _M_solver.SetOperator(*_M);

   _M_prec.reset(new BlockDiagonalPreconditioner(_block_offsets));

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

void FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &y)
{
   // RHS of the equation: -K*x-b
   Vector minus_x = x;
   minus_x.Neg();
   _K->Mult(minus_x, _z);
   _z -= *_b;
   SetTimeStep(dt);
   _M_solver.Mult(_z, y);
}

void FE_Evolution::SetTimeStep(real_t dt)
{
   if (_dt != dt)
   {
      _dt = dt;
      // Form operator A = M + dt*K
      _A.reset(new SumOperator(_K.get(), _dt, _M.get(), 1.0, false, false));
      // this will also call SetOperator on the preconditioner
      _M_solver.SetOperator(*_A);
   }
}

