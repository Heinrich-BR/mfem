//                                MFEM Example 18
//
// Compile with: make ex18
//
// Sample runs:
//
//       ex18 -p 1 -r 2 -o 1 -s 3
//       ex18 -p 1 -r 1 -o 3 -s 4
//       ex18 -p 1 -r 0 -o 5 -s 6
//       ex18 -p 2 -r 1 -o 1 -s 3
//       ex18 -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include <fstream>
#include <iostream>
#include <sstream>

#include "mfem.hpp"

// Classes HyperbolicConservationLaws, NumericalFlux, and FaceIntegrator
// shared between the serial and parallel version of the example.
#include "hyperbolic_conservation_laws.hpp"

// Choice for the problem setup. See InitialCondition in ex18.hpp.
int problem;
void ShallowWaterInitialCondition(const Vector &x, Vector &y);

void UpdateSystem(FiniteElementSpace &fes, FiniteElementSpace &dfes,
                  FiniteElementSpace &vfes,
                  DGHyperbolicConservationLaws &shallowWater, GridFunction &sol,
                  ODESolver *ode_solver);

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  problem = 1;
  const double g = 9.81;

  const char *mesh_file = "../data/periodic-square-4x4.mesh";
  int ref_levels = 3;
  int order = 3;
  int ode_solver_type = 4;
  double t_final = 20.0;
  double dt = -0.01;
  double cfl = 0.3;
  bool visualization = true;
  int vis_steps = 10;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&problem, "-p", "--problem",
                 "Problem setup to use. See options in velocity_function().");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Forward Euler,\n\t"
                 "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step",
                 "Time step. Positive number skips CFL timestep calculation.");
  args.AddOption(&cfl, "-c", "--cfl-number",
                 "CFL number for timestep calculation.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");

  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // 2. Read the mesh from the given mesh file. This example requires a 2D
  //    periodic mesh, such as ../data/periodic-square.mesh.
  Mesh mesh(mesh_file, 1, 1);
  const int dim = mesh.Dimension();

  MFEM_ASSERT(dim == 2,
              "Need a two-dimensional mesh for the problem definition");

  const int num_equations = dim + 1;

  // 3. Define the ODE solver used for time integration. Several explicit
  //    Runge-Kutta methods are available.
  ODESolver *ode_solver = NULL;
  switch (ode_solver_type) {
    case 1:
      ode_solver = new ForwardEulerSolver;
      break;
    case 2:
      ode_solver = new RK2Solver(1.0);
      break;
    case 3:
      ode_solver = new RK3SSPSolver;
      break;
    case 4:
      ode_solver = new RK4Solver;
      break;
    case 6:
      ode_solver = new RK6Solver;
      break;
    default:
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  mesh.EnsureNCMesh();
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh.UniformRefinement();
  }
  mesh.Transform([](const Vector &x, Vector &y) {
    y(0) = x(0) * 50.0;
    y(1) = x(1) * 50.0;
    return;
  });

  // 5. Define the discontinuous DG finite element space of the given
  //    polynomial order on the refined mesh.
  DG_FECollection fec(order, dim);
  // Finite element space for a scalar (thermodynamic quantity)
  FiniteElementSpace fes(&mesh, &fec);
  // Finite element space for a mesh-dim vector quantity (momentum)
  FiniteElementSpace dfes(&mesh, &fec, dim, Ordering::byNODES);
  // Finite element space for all variables together (total thermodynamic state)
  FiniteElementSpace vfes(&mesh, &fec, num_equations, Ordering::byNODES);

  // This example depends on this ordering of the space.
  MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

  cout << "Number of unknowns: " << vfes.GetVSize() << endl;

  // 6. Define the initial conditions, save the corresponding mesh and grid
  //    functions to a file. This can be opened with GLVis with the -gc option.

  // The solution u has components {density, x-momentum, y-momentum, energy}.
  // These are stored contiguously in the BlockVector u_block.

  Array<int> offsets(num_equations + 1);
  for (int k = 0; k <= num_equations; k++) {
    offsets[k] = k * vfes.GetNDofs();
  }
  BlockVector u_block(offsets);

  // Momentum grid function on dfes for visualization.
  GridFunction height(&fes, u_block.GetData());
  // Initialize the state.
  VectorFunctionCoefficient u0(num_equations, ShallowWaterInitialCondition);
  GridFunction sol(&vfes, u_block.GetData());
  sol.ProjectCoefficient(u0);

  // Output the initial solution.
  {
    ofstream mesh_ofs("vortex.mesh");
    mesh_ofs.precision(precision);
    mesh_ofs << mesh;
    for (int k = 0; k < num_equations; k++) {
      GridFunction uk(&fes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "vortex-" << k << "-init.gf";
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
    }
  }

  // 7. Set up the nonlinear form corresponding to the DG discretization of the
  //    flux divergence, and assemble the corresponding mass matrix.
  MixedBilinearForm divA(&dfes, &fes);
  divA.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator()));

  ShallowWaterFaceIntegrator *shallowWaterFaceIntegrator =
      new ShallowWaterFaceIntegrator(new RusanovFlux(), dim, g);

  // 8. Define the time-dependent evolution operator describing the ODE
  //    right-hand side, and perform time-integration (looping over the time
  //    iterations, ti, with a time-step dt).
  ShallowWater shallowWater(vfes, divA, *shallowWaterFaceIntegrator);

  // Visualize the density
  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;

    sout.open(vishost, visport);
    if (!sout) {
      cout << "Unable to connect to GLVis server at " << vishost << ':'
           << visport << endl;
      visualization = false;
      cout << "GLVis visualization disabled.\n";
    } else {
      sout.precision(precision);
      sout << "solution\n" << mesh << height;
      sout << "pause\n";
      sout << flush;
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
    }
  }

  // Determine the minimum element size.
  double hmin = 0.0;
  if (cfl > 0) {
    hmin = mesh.GetElementSize(0, 1);
    for (int i = 1; i < mesh.GetNE(); i++) {
      hmin = min(mesh.GetElementSize(i, 1), hmin);
    }
  }

  // Start the timer.
  tic_toc.Clear();
  tic_toc.Start();

  double t = 0.0;
  shallowWater.SetTime(t);
  ode_solver->Init(shallowWater);

  //   Vector zeros(mesh.GetNE());
  //   zeros = 0.0;
  //   mesh.DerefineByError(zeros, 1.0);
  //   mesh.UniformRefinement();
  //   UpdateSystem(fes, dfes, vfes, shallowWater, sol, ode_solver);

  if (cfl > 0) {
    // Find a safe dt, using a temporary vector. Calling Mult() computes the
    // maximum char speed at all quadrature points on all faces.
    Vector z(shallowWater.Width());
    shallowWater.Mult(sol, z);
    // faceForm.Mult(sol, z);
    dt = cfl * hmin / shallowWater.getMaxCharSpeed() / (2 * order + 1);
  }

  // Integrate in time.
  bool done = false;
  for (int ti = 0; !done;) {
    double dt_real = min(dt, t_final - t);

    ode_solver->Step(sol, t, dt_real);
    if (cfl > 0) {
      dt = cfl * hmin / shallowWater.getMaxCharSpeed() / (2 * order + 1);
    }
    ti++;

    done = (t >= t_final - 1e-8 * dt);
    if (done || ti % vis_steps == 0) {
      cout << "time step: " << ti << ", time: " << t << endl;
      if (visualization) {
        sout << "solution\n" << mesh << height << flush;
      }
    }
  }

  tic_toc.Stop();
  cout << " done, " << tic_toc.RealTime() << "s." << endl;

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m vortex.mesh -g vortex-1-final.gf".
  for (int k = 0; k < num_equations; k++) {
    GridFunction uk(&fes, u_block.GetBlock(k));
    ostringstream sol_name;
    sol_name << "vortex-" << k << "-final.gf";
    ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }

  // 10. Compute the L2 solution error summed for all components.
  //   if (t_final == 2.0) {
  const double error = sol.ComputeLpError(2, u0);
  cout << "Solution error: " << error << endl;
  //   }

  // Free the used memory.
  delete ode_solver;

  return 0;
}

void UpdateSystem(FiniteElementSpace &fes, FiniteElementSpace &dfes,
                  FiniteElementSpace &vfes,
                  DGHyperbolicConservationLaws &shallowWater, GridFunction &sol,
                  ODESolver *ode_solver) {
  fes.Update();
  dfes.Update();
  vfes.Update();
  sol.Update();
  shallowWater.Update();
  ode_solver->Init(shallowWater);
  fes.UpdatesFinished();
  dfes.UpdatesFinished();
  vfes.UpdatesFinished();
}

// Initial condition
void ShallowWaterInitialCondition(const Vector &x, Vector &y) {
  if (problem == 1) {
    const double maxval = 10;
    const double minval = 6;
    const double r_sigma = 0.05;
    const double xc = 0.0;
    const double yc = 0.0;
    const double dx = x(0) - xc;
    const double dy = x(1) - yc;

    y(0) = maxval * exp(-0.5 * r_sigma * (dx * dx + dy * dy)) + minval;
    y(1) = 0.0;
    y(2) = 0.0;
  } else {
    mfem_error("Invalid problem.");
  }
}