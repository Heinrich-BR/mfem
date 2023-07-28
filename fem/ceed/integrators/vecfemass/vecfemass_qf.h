// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_LIBCEED_VECFEMASS_QF_H
#define MFEM_LIBCEED_VECFEMASS_QF_H

#include "../util/util_qf.h"

#define LIBCEED_VECFEMASS_COEFF_COMP_MAX 6

struct VectorFEMassContext
{
   CeedInt dim, space_dim;
   CeedScalar coeff[LIBCEED_VECFEMASS_COEFF_COMP_MAX];
};

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a scalar constant coefficient
CEED_QFUNCTION(f_build_hdivmass_const_scalar)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = qw[i] * coeff0 * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a vector constant coefficient
CEED_QFUNCTION(f_build_hdivmass_const_vector)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, coeff, 1, 2, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, coeff, 1, 2, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a matrix constant coefficient
CEED_QFUNCTION(f_build_hdivmass_const_matrix)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, coeff, 1, 6, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, coeff, 1, 6, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a scalar constant coefficient
CEED_QFUNCTION(f_build_hcurlmass_const_scalar)(void *ctx, CeedInt Q,
                                               const CeedScalar *const *in,
                                               CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            qd[i] = qw[i] * coeff0 / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 1, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a vector constant coefficient
CEED_QFUNCTION(f_build_hcurlmass_const_vector)(void *ctx, CeedInt Q,
                                               const CeedScalar *const *in,
                                               CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 2, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 2, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a matrix constant coefficient
CEED_QFUNCTION(f_build_hcurlmass_const_matrix)(void *ctx, CeedInt Q,
                                               const CeedScalar *const *in,
                                               CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[1] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 3, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 6, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 6, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a scalar coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hdivmass_quad_scalar)(void *ctx, CeedInt Q,
                                             const CeedScalar *const *in,
                                             CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] * J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a vector coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hdivmass_quad_vector)(void *ctx, CeedInt Q,
                                             const CeedScalar *const *in,
                                             CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=space_dim, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, c + i, Q, 2, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, c + i, Q, 2, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(div) mass operator
/// with a matrix coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hdivmass_quad_matrix)(void *ctx, CeedInt Q,
                                             const CeedScalar *const *in,
                                             CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ21(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ22(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ32(J + i, Q, c + i, Q, 6, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultJtCJ33(J + i, Q, c + i, Q, 6, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a scalar coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hcurlmass_quad_scalar)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            qd[i] = qw[i] * c[i] / J[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 1, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a vector coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hcurlmass_quad_vector)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=space_dim, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 2, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 2, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for building quadrature data for an H(curl) mass operator
/// with a matrix coefficient evaluated at quadrature points
CEED_QFUNCTION(f_build_hcurlmass_quad_matrix)(void *ctx, CeedInt Q,
                                              const CeedScalar *const *in,
                                              CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0] is coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div)) and store the symmetric part
   // of the result
   const CeedScalar *c = in[0], *J = in[1], *qw = in[2];
   CeedScalar *qd = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 3, qw[i], Q, qd + i);
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 6, qw[i], Q, qd + i);
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 6, qw[i], Q, qd + i);
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying a vector FE mass operator
CEED_QFUNCTION(f_apply_vecfemass)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   const CeedScalar *u = in[0], *qd = in[1];
   CeedScalar *v = out[0];
   switch (bc->dim)
   {
      case 1:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            v[i] = qd[i] * u[i];
         }
         break;
      case 2:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[i + Q * 0] * u0 + qd[i + Q * 1] * u1;
            v[i + Q * 1] = qd[i + Q * 1] * u0 + qd[i + Q * 2] * u1;
         }
         break;
      case 3:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[i + Q * 0] * u0 + qd[i + Q * 1] * u1 + qd[i + Q * 2] * u2;
            v[i + Q * 1] = qd[i + Q * 1] * u0 + qd[i + Q * 3] * u1 + qd[i + Q * 4] * u2;
            v[i + Q * 2] = qd[i + Q * 2] * u0 + qd[i + Q * 4] * u1 + qd[i + Q * 5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) mass operator with a scalar
/// constant coefficient
CEED_QFUNCTION(f_apply_hdivmass_mf_const_scalar)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = qw[i] * coeff0 * J[i];
            v[i] = qd * u[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, coeff, 1, 1, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) mass operator with a vector
/// constant coefficient
CEED_QFUNCTION(f_apply_hdivmass_mf_const_vector)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, coeff, 1, 2, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, coeff, 1, 2, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) mass operator with a matrix
/// constant coefficient
CEED_QFUNCTION(f_apply_hdivmass_mf_const_matrix)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, coeff, 1, 3, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, coeff, 1, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, coeff, 1, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) mass operator with a scalar
/// constant coefficient
CEED_QFUNCTION(f_apply_hcurlmass_mf_const_scalar)(void *ctx, CeedInt Q,
                                                  const CeedScalar *const *in,
                                                  CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar coeff0 = coeff[0];
            const CeedScalar qd = qw[i] * coeff0 / J[i];
            v[i] = qd * u[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 1, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) mass operator with a vector
/// constant coefficient
CEED_QFUNCTION(f_apply_hcurlmass_mf_const_vector)(void *ctx, CeedInt Q,
                                                  const CeedScalar *const *in,
                                                  CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 2, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 2, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) mass operator with a matrix
/// constant coefficient
CEED_QFUNCTION(f_apply_hcurlmass_mf_const_matrix)(void *ctx, CeedInt Q,
                                                  const CeedScalar *const *in,
                                                  CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[2] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *coeff = bc->coeff;
   const CeedScalar *u = in[0], *J = in[1], *qw = in[2];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, coeff, 1, 3, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, coeff, 1, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, coeff, 1, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, coeff, 1, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) operator with a scalar
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hdivmass_mf_quad_scalar)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] * J[i];
            v[i] = qd * u[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, c + i, Q, 1, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) operator with a vector
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hdivmass_mf_quad_vector)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, c + i, Q, 2, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, c + i, Q, 2, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(div) operator with a matrix
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hdivmass_mf_quad_matrix)(void *ctx, CeedInt Q,
                                                const CeedScalar *const *in,
                                                CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultJtCJ21(J + i, Q, c + i, Q, 3, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ22(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultJtCJ32(J + i, Q, c + i, Q, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultJtCJ33(J + i, Q, c + i, Q, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) operator with a scalar
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hcurlmass_mf_quad_scalar)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=1, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 11:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            const CeedScalar qd = qw[i] * c[i] / J[i];
            v[i] = qd * u[i];
         }
         break;
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 1, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 1, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) operator with a vector
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hcurlmass_mf_quad_vector)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=space_dim, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 2, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 2, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

/// libCEED QFunction for applying an H(curl) operator with a matrix
/// coefficient evaluated at quadrature points
CEED_QFUNCTION(f_apply_hcurlmass_mf_quad_matrix)(void *ctx, CeedInt Q,
                                                 const CeedScalar *const *in,
                                                 CeedScalar *const *out)
{
   VectorFEMassContext *bc = (VectorFEMassContext *)ctx;
   // in[0], out[0] have shape [dim, ncomp=1, Q]
   // in[1] is coefficients with shape [ncomp=space_dim*(space_dim+1)/2, Q]
   // in[2] is Jacobians with shape [dim, ncomp=space_dim, Q]
   // in[3] is quadrature weights, size (Q)
   //
   // At every quadrature point, compute qw/det(J) adj(J) C adj(J)^T (for
   // H(curl)) or qw/det(J) J^T C J (for H(div))
   const CeedScalar *u = in[0], *c = in[1], *J = in[2], *qw = in[3];
   CeedScalar *v = out[0];
   switch (10 * bc->space_dim + bc->dim)
   {
      case 21:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd;
            MultAdjJCAdjJt21(J + i, Q, c + i, Q, 3, qw[i], 1, &qd);
            v[i] = qd * u[i];
         }
         break;
      case 22:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt22(J + i, Q, c + i, Q, 3, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 32:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[3];
            MultAdjJCAdjJt32(J + i, Q, c + i, Q, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1;
            v[i + Q * 1] = qd[1] * u0 + qd[2] * u1;
         }
         break;
      case 33:
         CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++)
         {
            CeedScalar qd[6];
            MultAdjJCAdjJt33(J + i, Q, c + i, Q, 6, qw[i], 1, qd);
            const CeedScalar u0 = u[i + Q * 0];
            const CeedScalar u1 = u[i + Q * 1];
            const CeedScalar u2 = u[i + Q * 2];
            v[i + Q * 0] = qd[0] * u0 + qd[1] * u1 + qd[2] * u2;
            v[i + Q * 1] = qd[1] * u0 + qd[3] * u1 + qd[4] * u2;
            v[i + Q * 2] = qd[2] * u0 + qd[4] * u1 + qd[5] * u2;
         }
         break;
   }
   return 0;
}

#endif // MFEM_LIBCEED_VECFEMASS_QF_H