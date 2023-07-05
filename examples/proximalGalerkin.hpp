#ifndef PROXIMAL_GALERKIN_HPP
#define PROXIMAL_GALERKIN_HPP
#include "mfem.hpp"


double sigmoid(const double x)
{
   if (x < 0)
   {
      const double exp_x = std::exp(x);
      return exp_x / (1.0 + exp_x);
   }
   return 1.0 / (1.0 + std::exp(-x));
}

double dsigmoiddx(const double x)
{
   const double tmp = sigmoid(x);
   return tmp - std::pow(tmp, 2);
}

double logit(const double x)
{
   return std::log(x / (1.0 - x));
}

double simpRule(const double rho, const int exponent, const double rho0)
{
   return rho0 + (1.0 - rho0)*std::pow(rho, exponent);
}

double dsimpRuledx(const double rho, const int exponent, const double rho0)
{
   return exponent*(1.0 - rho0)*std::pow(rho, exponent - 1);
}

double d2simpRuledx2(const double rho, const int exponent, const double rho0)
{
   return exponent*(exponent - 1)*(1.0 - rho0)*std::pow(rho, exponent - 2);
}


namespace mfem
{
class MappedGridFunctionCoefficient :public GridFunctionCoefficient
{
   typedef std::__1::function<double (const double)> Mapping;
public:
   MappedGridFunctionCoefficient(GridFunction *gf, Mapping f_x, int comp=1)
      :GridFunctionCoefficient(gf, comp), map(f_x) { }

   /// Evaluate the coefficient at @a ip.
   virtual double Eval(ElementTransformation &T,
                       const IntegrationPoint &ip)
   {
      return map(GridFunctionCoefficient::Eval(T, ip));
   }

   inline void SetMapping(Mapping &f_x) { map = f_x; }
private:
   Mapping map;
};

MappedGridFunctionCoefficient SIMPCoefficient(GridFunction *gf,
                                              const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return simpRule(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient DerSIMPCoefficient(GridFunction *gf,
                                                 const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return dsimpRuledx(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient Der2SIMPCoefficient(GridFunction *gf,
                                                  const int exponent, const double rho0)
{
   auto map = [exponent, rho0](const double x) {return d2simpRuledx2(x, exponent, rho0); };
   return MappedGridFunctionCoefficient(gf, map);
}

MappedGridFunctionCoefficient SigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, sigmoid);
}

MappedGridFunctionCoefficient DerSigmoidCoefficient(GridFunction *gf)
{
   return MappedGridFunctionCoefficient(gf, dsigmoiddx);
}

double VolumeProjection(GridFunction &psi, const double target_volume,
                        const double tol=1e-12,
                        const int max_its=10)
{
   auto sigmoid_psi = SigmoidCoefficient(&psi);
   auto der_sigmoid_psi = DerSigmoidCoefficient(&psi);

   LinearForm int_sigmoid_psi(psi.FESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   LinearForm int_der_sigmoid_psi(psi.FESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      const double f = int_sigmoid_psi.Sum() - target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      const double df = int_der_sigmoid_psi.Sum();

      const double dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   if (!done)
   {
      mfem_warning("Projection reached maximum iteration without converging. Result may not be accurate.");
   }
   int_sigmoid_psi.Assemble();
   return int_sigmoid_psi.Sum();
}

#ifdef MFEM_USE_MPI
double ParVolumeProjection(ParGridFunction &psi, const double target_volume,
                           const double tol=1e-12, const int max_its=10)
{
   auto sigmoid_psi = SigmoidCoefficient(&psi);
   auto der_sigmoid_psi = DerSigmoidCoefficient(&psi);
   auto comm = psi.ParFESpace()->GetComm();

   ParLinearForm int_sigmoid_psi(psi.ParFESpace());
   int_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(sigmoid_psi));
   ParLinearForm int_der_sigmoid_psi(psi.ParFESpace());
   int_der_sigmoid_psi.AddDomainIntegrator(new DomainLFIntegrator(
                                              der_sigmoid_psi));
   bool done = false;
   for (int k=0; k<max_its; k++) // Newton iteration
   {
      int_sigmoid_psi.Assemble(); // Recompute f(c) with updated ψ
      const double myf = int_sigmoid_psi.Sum();
      double f;
      MPI_Allreduce(&myf, &f, 1, MPI_DOUBLE, MPI_SUM, comm);
      f -= target_volume;

      int_der_sigmoid_psi.Assemble(); // Recompute df(c) with updated ψ
      const double mydf = int_der_sigmoid_psi.Sum();
      double df;
      MPI_Allreduce(&mydf, &df, 1, MPI_DOUBLE, MPI_SUM, comm);

      const double dc = -f/df;
      psi += dc;
      if (abs(dc) < tol) { done = true; break; }
   }
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if ((rank == 0) & (!done))
   {
      mfem_warning("Projection reached maximum iteration without converging. Result may not be accurate.");
   }
   const double myf = int_sigmoid_psi.Sum();
   double f;
   MPI_Allreduce(&myf, &f, 1, MPI_DOUBLE, MPI_SUM, comm);
   return f;
}
#endif // end of MFEM_USE_MPI for ParProjit

Array<int> getOffsets(Array<FiniteElementSpace*> &spaces)
{
   Array<int> offsets(spaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<spaces.Size(); i++)
   {
      offsets[i + 1] = spaces[i]->GetVSize();
   }
   offsets.PartialSum();
   return offsets;
}

class BlockLinearSystem
{
public:
   bool own_blocks = false;
   BlockLinearSystem(Array<int> &offsets_,
                     Array<FiniteElementSpace*> &array_of_spaces,
                     Array2D<int> &array_of_ess_bdr)
      : offsets(offsets_), spaces(array_of_spaces), ess_bdr(array_of_ess_bdr),
        numSpaces(array_of_spaces.Size()),
        A(offsets_), b(offsets_), prec(offsets_)
   {
      A_forms.SetSize(numSpaces, numSpaces);
      A_forms = nullptr;

      b_forms.SetSize(numSpaces);
      for (int i=0; i<numSpaces; i++) { b_forms[i] = new LinearForm(spaces[i]); }

      prec.owns_blocks = true;
   }

   void SetBlockMatrix(int i, int j, Matrix *mat)
   {

      if (i == j)
      {
         auto bilf = static_cast<BilinearForm*>(mat);
         if (!bilf) { mfem_error("Cannot convert provided Matrix to BilinearForm"); }
         SetDiagBlockMatrix(i, bilf);
         return;
      }
      auto bilf = static_cast<MixedBilinearForm*>(mat);
      if (!bilf) { mfem_error("Cannot convert provided Matrix to MixedBilinearForm"); }
      if (bilf->TestFESpace() != spaces[i])
      {
         mfem_error("The provided BilinearForm's test space does not match with the provided array of spaces. Check the initialization and block index");
      }
      if (bilf->TrialFESpace() != spaces[j])
      {
         mfem_error("The provided BilinearForm's trial space does not match with the provided array of spaces. Check the initialization and block index");
      }
      A_forms(i, j) = bilf;
   }

   void SetDiagBlockMatrix(int i, BilinearForm *bilf)
   {
      if (bilf->FESpace() != spaces[i]) {mfem_error("The provided BilinearForm's space does not match with the provided array of spaces. Check the initialization and block index"); }
      A_forms(i, i) = bilf;
   }

   inline MixedBilinearForm *GetBlock(int i, int j)
   {
      if (i == j) { mfem_error("For diagonal block, use GetDiagBlock(i)."); }
      return static_cast<MixedBilinearForm*>(A_forms(i, j));
   }
   inline BilinearForm *GetDiagBlock(int i)
   {
      return static_cast<BilinearForm*>(A_forms(i, i));
   }
   inline LinearForm *GetLinearForm(int i)
   {
      return b_forms[i];
   }

   /**
    * @brief Assemble block matrices. X should contain essential boundary condition.
    *
    * @param x Solution vector that contains the essential boundary condition.
    */
   void Assemble(BlockVector &x);

   void GMRES(BlockVector &x);
   void CG(BlockVector &x) { mfem_error("Not yet implemented."); }

   ~BlockLinearSystem() = default;

private:
   Array<int> &offsets;
   Array<FiniteElementSpace*> &spaces;
   Array2D<int> &ess_bdr;
   int numSpaces;
   BlockOperator A;
   BlockVector b;
   Array2D<Matrix*> A_forms;
   Array<LinearForm*> b_forms;
   BlockDiagonalPreconditioner prec;
};


void BlockLinearSystem::Assemble(BlockVector &x)
{
   Array<int> trial_ess_bdr;
   Array<int> test_ess_bdr;
   b = 0.0;
   for (int row = 0; row < numSpaces; row++)
   {
      b_forms[row]->Assemble();
      test_ess_bdr.MakeRef(ess_bdr.GetRow(row), ess_bdr.NumCols());
      if (A_forms(row, row))
      {
         BilinearForm* bilf = this->GetDiagBlock(row);

         bilf->SetDiagonalPolicy(mfem::Operator::DIAG_ONE);
         bilf->Assemble();
         bilf->EliminateEssentialBC(test_ess_bdr, x.GetBlock(row), b.GetBlock(row));
         bilf->Finalize();
         A.SetBlock(row, row, &(bilf->SpMat()));
         prec.SetDiagonalBlock(row, new GSSmoother(bilf->SpMat()));
      }
      for (int col = 0; col < numSpaces; col++)
      {
         if (col == row) // if diagonal, already handled using bilinear form
         {
            continue;
         }
         trial_ess_bdr.MakeRef(ess_bdr.GetRow(col), ess_bdr.NumCols());
         if (A_forms(row, col))
         {
            MixedBilinearForm* bilf = this->GetBlock(row, col);
            bilf->Assemble();
            bilf->EliminateTrialDofs(trial_ess_bdr, x.GetBlock(col), b.GetBlock(col));
            bilf->EliminateTestDofs(test_ess_bdr);
            bilf->Finalize();
            A.SetBlock(row, col, &(bilf->SpMat()));
         }
      }
   }
}

void BlockLinearSystem::GMRES(BlockVector &x)
{
   mfem::GMRES(A, prec, b, x, 0, 200, 50, 1e-12, 0.0);
   for (int row=0; row < numSpaces; row++)
   {
      GetDiagBlock(row)->LoseMat();
      for (int col=0; col<numSpaces; col++)
      {
         if (col == row) { continue; }
         GetBlock(row, col)->LoseMat();
      }
   }
}
} // end of namespace mfem

#endif // end of proximalGalerkin.hpp