//                                MFEM Example 15
//

#include "../mfem.hpp"
#include <fstream>
#include <iostream>
#include "../../algoim/src/algoim_quad.hpp"
#include "blitz/tinyvec2.h"
#include "../linalg/ttensor.hpp"

using namespace std;
using namespace mfem;
using namespace Algoim;

template<typename T>
double GetTValue(T x)
{
    return (double)x;
};

template<int N>
double GetTValue(Algoim::Interval<N> x)
{
    return x.alpha;
};

template<typename T>
void SetTValue(T &x, double xv)
{
    x = (T)xv;
};

template<int N>
void SetTValue(Algoim::Interval<N> &x, double xv)
{
    x.alpha = xv;
};

#define radius 0.6
#define lstype 2
template<int N, typename T>
T ReturnFuncValue(blitz::TinyVector<T,N> &xc)
{
    T fx;
    if (lstype == 1) {
        fx = 1*(xc(0)*xc(0) + xc(1)*xc(1) - radius*radius);
    }
    else if (lstype == 2) {
        double a1 = 20., a2 = 2., a3 = 3.;
        T yv = a1*(xc(1)-0.5),
          xv = a2*sin(a3*(xc(0)-0.5)*M_PI);
        fx = tanh(yv + xv + 1);
    }
    else {
        T fx = (T)(0.);
    }
    return fx;
};

template<int N, typename T>
blitz::TinyVector<T,N> ReturnFuncGradient(blitz::TinyVector<T,N> &xc,
                                          DenseMatrix &J)
{
    blitz::TinyVector<T,N> dfx;
    if (lstype == 1) {
        dfx = blitz::TinyVector<T,N>(2.0*J(0, 0)*xc(0) + 2.0*J(1, 0)*xc(1),
                                     2.0*J(0, 1)*xc(0) + 2.0*J(1, 1)*xc(1));
    }
    else if (lstype == 2) {
        double a1 = 20., a2 = 2., a3 = 3.;
        T yv = a1*(xc(1)-0.5),
          xv = a2*sin(a3*(xc(0)-0.5)*M_PI),
       scale = sech(yv+xv+1);
        dfx  = blitz::TinyVector<T,N>(scale*xv*a3*J(0, 0) + scale*a1*J(1, 0),
                                      scale*xv*a3*J(0, 1) + scale*a1*J(1, 1));
    }
    else {
        dfx = blitz::TinyVector<T,N>((T)(0.), (T)(0.));
    }
    return dfx;
};

double levelset(const Vector &x)
{
   blitz::TinyVector<double, 2> xc;
   for (int i = 0; i < x.Size(); i++) { xc(i) = x(i); }
   double val = ReturnFuncValue(xc);
   return val;
}


template<int N>
struct AnalyticalLevelSet
{
private:
    ElementTransformation *Tr;
public:
    AnalyticalLevelSet(ElementTransformation &Tr_) : Tr(&Tr_) { }

    template<typename T>
    T operator() (blitz::TinyVector<T,N>& x) const
    {
        Vector xm(N);
        for (int i = 0; i < N; i++) {
            xm(i) = GetTValue(x(i));
        }
        Vector xtm(N);
        IntegrationPoint *ip = new IntegrationPoint();
        ip->Set2(xm(0), xm(1));
        Tr->Transform(*ip, xtm);
        delete ip;
        blitz::TinyVector<T,N> xc = x;
        for (int i = 0; i < N; i++) {
            SetTValue(xc(i), xtm(i));
        }
        return ReturnFuncValue(xc);
    }

    template<typename T>
    blitz::TinyVector<T,N> grad(blitz::TinyVector<T,N>& x) const
    {
        Vector xm(N);
        for (int i = 0; i < N; i++) {
            xm(i) = GetTValue(x(i));
        }
        Vector xtm(N);
        IntegrationPoint *ip = new IntegrationPoint();
        ip->Set2(xm(0), xm(1));
        Tr->Transform(*ip, xtm);
        blitz::TinyVector<T,N> xc = x;
        for (int i = 0; i < N; i++) {
            SetTValue(xc(i), xtm(i));
        }
        DenseMatrix J(N);
        Tr->SetIntPoint(ip);
        J = Tr->Jacobian();
        delete ip;

        return ReturnFuncGradient(xc, J);
    }
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2;
   double t_final = 1.0;
   double max_elem_error = 5.0e-3;
   double hysteresis = 0.15; // derefinement safety coefficient
   int ref_levels = 0;
   int nc_limit = 3;         // maximum level of hanging nodes
   bool visualization = true;
   bool visit = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Mesh mesh_(mesh_file, 1, 1, false);

   H1_FECollection h1fec_(order, mesh_.Dimension());
   FiniteElementSpace h1fes_(&mesh_, &h1fec_);
   GridFunction x0(&h1fes_);

   FunctionCoefficient ind(levelset);
   x0.ProjectCoefficient(ind);
   {
       osockstream sock(19916, "localhost");
       sock << "solution\n";
       mesh_.Print(sock);
       x0.Save(sock);
       sock.send();
       sock << "window_title 'Level set'\n"
            << "window_geometry "
            << 1200 << " " << 0 << " " << 600 << " " << 600 << "\n"
            << "keys jRmclA" << endl;
   }

   double area = 0.0;
   ElementTransformation *Tr = NULL;
   IntegrationPoint *ip = new IntegrationPoint();
   Vector el_area(mesh_.GetNE());
   el_area = 0.0;
   ofstream myfile;
   myfile.open ("qpts.out");
   for (int e = 0; e < mesh_.GetNE(); e++) {
       Tr = mesh_.GetElementTransformation(e);
       AnalyticalLevelSet<2> phi(*Tr);
       auto q = Algoim::quadGen<2>(phi, Algoim::BoundingBox<double,2>(0.0, 1.0), -2, -1, order);
       double elsum = 0.0;
       for (const auto& pt : q.nodes)
       {
           ip->Set2(pt.x(0), pt.x(1));
           Tr->SetIntPoint(ip);
           Vector xtm(2);
           Tr->Transform(*ip, xtm);
           area += Tr->Weight() * pt.w;
           elsum += Tr->Weight() * pt.w;
           myfile << xtm(0) << " " << xtm(1) << endl; //write quadrature points to file.
       }
       el_area(e) = elsum;
   }
   myfile.close();
   double exact_area;
   if (lstype == 1) {
       exact_area = M_PI*radius*radius/4;
   }
   else if (lstype == 2) {
       exact_area = 0.45;
   }
   std::cout << " Location of integration points output in qpts.out.\n";
   cout << " Numerical area: " <<  std::setprecision(5) << area << endl;
   cout << " Exact area:     " <<  std::setprecision(5) << exact_area  << endl;
   cout << " Error:          " <<  std::setprecision(5) << std::fabs(area-exact_area) << endl;
//   el_area.Print();


   return 0;
}
