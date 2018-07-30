// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../config/config.hpp"
#include "hiop.hpp"


#ifdef MFEM_USE_HIOP
#include <iostream>

#pragma message "Compiling " __FILE__ "..."

using namespace hiop;

namespace mfem
{ 


HiopNlpOptimizer::HiopNlpOptimizer()
{
  _optProb = new HiopProblemSpec();
  _hiopInstance = new hiopNlpDenseConstraints(*_optProb);
}

#ifdef MFEM_USE_MPI
HiopNlpOptimizer::HiopNlpOptimizer(MPI_Comm _comm) 
  : IterativeSolver(_comm)
{
  _optProb = NULL;
  _hiopInstance = NULL;
};
#endif

HiopNlpOptimizer::~HiopNlpOptimizer()
{
  if(_optProb) delete _optProb;
  if(_hiopInstance) delete _hiopInstance;
}

void HiopNlpOptimizer::Mult(const Vector &xt, Vector &x) const
{
 
}

void HiopNlpOptimizer::SetPreconditioner(Solver &pr)
{
   mfem_error("HiopNlpOptimizer::SetPreconditioner() : "
              "not meaningful for this solver");
}

void HiopNlpOptimizer::SetOperator(const Operator &op)
{
   mfem_error("HiopNlpOptimizer::SetOperator() : "
              "not meaningful for this solver");
}

} // mfem namespace


#endif // MFEM_USE_HIOP
