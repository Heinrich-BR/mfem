//                                MFEM ShallowWater Equation examples
//
// Compile with: make shallowWater
//
// Sample runs:
//
//       shallow_water -p 1 -r 2 -o 1 -s 3
//       shallow_water -p 1 -r 1 -o 3 -s 4
//       shallow_water -p 1 -r 0 -o 5 -s 6
//       shallow_water -p 2 -r 1 -o 1 -s 3
//       shallow_water -p 2 -r 0 -o 3 -s 3
//
// Description:  This example code solves the compressible Shallow Water system
// of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a shallow_water is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the shallow_water returns
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
#include "fem/hyperbolic_conservation_laws.hpp"

// Choice for the problem setup. See InitialCondition in ex18.hpp.

typedef std::__1::function<void(const Vector &, Vector &)> SpatialFunction;

void ShallowWaterMesh(const int problem, const char **mesh_file);

SpatialFunction ShallowWaterInitialCondition(const int problem, const double g);

void UpdateSystem(FiniteElementSpace &fes, FiniteElementSpace &dfes,
                  FiniteElementSpace &vfes,
                  DGHyperbolicConservationLaws &shallowWater, GridFunction &sol,
                  ODESolver *ode_solver);

int main(int argc, char *argv[]) {
  Mpi::Init(argc, argv);
  const int numProcs = Mpi::WorldSize();
  const int myRank = Mpi::WorldRank();
  Hypre::Init();

  // 1. Parse command-line options.
  int problem = 1;
  const double g = 9.81;

  const char *mesh_file = "";
  int IntOrderOffset = 3;
  int ser_ref_levels = 0;
  int par_ref_levels = 2;
  int order = 3;
  int ode_solver_type = 4;
  double t_final = 20.0;
  double dt = -0.01;
  double cfl = 0.3;
  bool visualization = true;
  int vis_steps = 50;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&problem, "-p", "--problem",
                 "Problem setup to use. See options in velocity_function().");
  args.AddOption(&ser_ref_levels, "-rs", "--serial-refine",
                 "Number of times to refine the serial mesh uniformly.");
  args.AddOption(&par_ref_levels, "-rp", "--parallel-refine",
                 "Number of times to refine the parallel mesh uniformly.");
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
    if (Mpi::Root()) args.PrintUsage(cout);
    return 1;
  }
  // When the user does not provide mesh file,
  // use the default mesh file for the problem.
  if ((mesh_file == NULL) || (mesh_file[0] == '\0')) {  // if NULL or empty
    ShallowWaterMesh(problem, &mesh_file);  // get default mesh file name
  }
  if (Mpi::Root()) args.PrintOptions(cout);

  // 2. Read the mesh from the given mesh file.
  Mesh mesh = Mesh(mesh_file);
  const int dim = mesh.Dimension();
  const int num_equations = dim + 1;

  // perform uniform refine
  mesh.Transform([](const Vector &x, Vector &newx) {
    newx = x;
    newx *= 25.0;
  });

  if (numProcs > mesh.GetNE()) {
    if (Mpi::Root()) {
      mfem_warning(
          "The number of processor is larger than the number of elements.\n"
          "Refine serial meshes until the number of elements is large enough");
    }
    while (mesh.GetNE() < numProcs) {
      mesh.UniformRefinement();
    }
  }
  if (dim > 1) mesh.EnsureNCMesh();

  ParMesh pmesh = ParMesh(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  for (int lev = 0; lev < par_ref_levels; lev++) {
    pmesh.UniformRefinement();
  }
  if (dim > 1) pmesh.EnsureNCMesh();

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

  // 4. Define the discontinuous DG finite element space of the given
  //    polynomial order on the refined mesh.
  DG_FECollection fec(order, dim);
  // Finite element space for a scalar (thermodynamic quantity)
  ParFiniteElementSpace fes(&pmesh, &fec);
  // Finite element space for a mesh-dim vector quantity (momentum)
  ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);
  // Finite element space for all variables together (total thermodynamic state)
  ParFiniteElementSpace vfes(&pmesh, &fec, num_equations, Ordering::byNODES);

  // This example depends on this ordering of the space.
  MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

  if (Mpi::Root()) {
    cout << "Number of unknowns: " << vfes.GetVSize() << endl;
  }

  // 6. Define the initial conditions, save the corresponding mesh and grid
  //    functions to a file. This can be opened with GLVis with the -gc option.
  // Initialize the state.
  VectorFunctionCoefficient u0(num_equations,
                               ShallowWaterInitialCondition(problem, g));
  ParGridFunction sol(&vfes);
  sol.ProjectCoefficient(u0);

  // Output the initial solution.
  {
    ostringstream mesh_name;
    mesh_name << "shallow-water-mesh." << setfill('0') << setw(6)
              << Mpi::WorldRank();
    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(precision);
    mesh_ofs << pmesh;

    for (int k = 0; k < num_equations; k++) {
      ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
      ostringstream sol_name;
      sol_name << "shallow-water-" << k << "-init." << setfill('0') << setw(6)
               << Mpi::WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
    }
  }

  // 7. Set up the nonlinear form corresponding to the DG discretization of the
  //    flux divergence, and assemble the corresponding mass matrix.
  ShallowWaterElementFormIntegrator *shallowWaterElementFormIntegrator =
      new ShallowWaterElementFormIntegrator(dim, g, IntOrderOffset);

  NumericalFlux *numericalFlux = new RusanovFlux();
  ShallowWaterFaceFormIntegrator *shallowWaterFaceFormIntegrator =
      new ShallowWaterFaceFormIntegrator(numericalFlux, dim, g, IntOrderOffset);
  ParNonlinearForm nonlinForm(&vfes);

  // 8. Define the time-dependent evolution operator describing the ODE
  //    right-hand side, and perform time-integration (looping over the time
  //    iterations, ti, with a time-step dt).
  DGHyperbolicConservationLaws shallowWater(
      &vfes, nonlinForm, *shallowWaterElementFormIntegrator,
      *shallowWaterFaceFormIntegrator, num_equations);

  // Visualize the density
  socketstream sout;
  if (visualization) {
    char vishost[] = "localhost";
    int visport = 19916;

    sout.open(vishost, visport);
    if (!sout) {
      visualization = false;
      if (Mpi::Root()) {
        cout << "Unable to connect to GLVis server at " << vishost << ':'
             << visport << endl;
        cout << "GLVis visualization disabled.\n";
      }
    } else {
      ParGridFunction height(&fes, sol.GetData());
      sout << "parallel " << numProcs << " " << myRank << "\n";
      sout.precision(precision);
      sout << "solution\n" << pmesh << height;
      sout << "pause\n";
      sout << flush;
      if (Mpi::Root()) {
        cout << "GLVis visualization paused."
             << " Press space (in the GLVis window) to resume it.\n";
      }
      MPI_Barrier(pmesh.GetComm());
    }
  }

  // Determine the minimum element size.
  double hmin;
  if (cfl > 0) {
    double my_hmin = pmesh.GetNE() > 0 ? pmesh.GetElementSize(0, 1) : INFINITY;
    for (int i = 1; i < pmesh.GetNE(); i++) {
      my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
    }
    MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
  }

  // Start the timer.
  tic_toc.Clear();
  tic_toc.Start();

  double t = 0.0;
  shallowWater.SetTime(t);
  ode_solver->Init(shallowWater);

  if (cfl > 0) {
    // Find a safe dt, using a temporary vector. Calling Mult() computes the
    // maximum char speed at all quadrature points on all faces.
    Vector z(sol.Size());
    shallowWater.Mult(sol, z);

    double max_char_speed;
    double my_max_char_speed = shallowWater.getMaxCharSpeed();
    MPI_Allreduce(&my_max_char_speed, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                  pmesh.GetComm());
    dt = cfl * hmin / max_char_speed / (2 * order + 1);
  }

  // Integrate in time.
  bool done = false;
  for (int ti = 0; !done;) {
    double dt_real = min(dt, t_final - t);

    ode_solver->Step(sol, t, dt_real);
    if (cfl > 0) {
      double max_char_speed;
      double my_max_char_speed = shallowWater.getMaxCharSpeed();
      MPI_Allreduce(&my_max_char_speed, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());
      dt = cfl * hmin / max_char_speed / (2 * order + 1);
    }
    ti++;

    done = (t >= t_final - 1e-8 * dt);
    if (done || ti % vis_steps == 0) {
      if (Mpi::Root()) {
        cout << "time step: " << ti << ", time: " << t << endl;
      }
      if (visualization) {
        ParGridFunction height(&fes, sol.GetData());
        sout << "parallel " << numProcs << " " << myRank << "\n";
        sout << "solution\n" << pmesh << height << flush;
        MPI_Barrier(pmesh.GetComm());
      }
    }
  }
  MPI_Barrier(pmesh.GetComm());
  tic_toc.Stop();
  if (Mpi::Root()) {
    cout << " done, " << tic_toc.RealTime() << "s." << endl;
  }

  // 9. Save the final solution. This output can be viewed later using GLVis:
  //    "glvis -m shallow_water.mesh -g shallow_water-1-final.gf".
  {
    ostringstream mesh_name;
    mesh_name << "shallow-water-mesh-final." << setfill('0') << setw(6)
              << Mpi::WorldRank();
    ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(precision);
    mesh_ofs << pmesh;

    for (int k = 0; k < num_equations; k++) {
      ParGridFunction uk(&fes, sol.GetData() + k * fes.GetNDofs());
      ostringstream sol_name;
      sol_name << "shallow-water-" << k << "-final." << setfill('0') << setw(6)
               << Mpi::WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
    }
  }

  // 10. Compute the L2 solution error summed for all components.
  //   if (t_final == 2.0) {
  const double error = sol.ComputeLpError(2, u0);
  if (Mpi::Root()) {
    cout << "Solution error: " << error << endl;
  }

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

void ShallowWaterMesh(const int problem, const char **mesh_file) {
  switch (problem) {
    case 1:
      *mesh_file = "../data/periodic-square-4x4.mesh";
      break;
    default:
      throw invalid_argument("Default mesh is undefined");
  }
}

// Initial condition
SpatialFunction ShallowWaterInitialCondition(const int problem,
                                             const double g) {
  switch (problem) {
    case 1:
      return [g](const Vector &x, Vector &y) {
        const double maxval = 10;
        const double minval = 6;
        const double r_sigma = 0.05;
        const double xc = 0.0;
        const double yc = 0.0;
        const double dx = x(0) - xc;
        const double dy = x(1) - yc;

        y(0) = (maxval - minval) * exp(-0.5 * r_sigma * (dx * dx + dy * dy)) +
               minval;
        y(1) = 0.0;
        y(2) = 0.0;
      };
    default:
      throw invalid_argument("Invalid problem");
  }
}