// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../../config/config.hpp"

#ifdef MFEM_USE_JIT
#include "../communication.hpp"
#include "../error.hpp"
#include "jit.hpp"

#include <map>
#include <string>
#include <fstream>
#include <thread> // sleep_for
#include <chrono> // (milli) seconds

#include <cassert>
#include <cstring> // strlen
#include <cstdlib> // exit, system
#include <dlfcn.h> // dlopen/dlsym, not available on Windows
#include <signal.h> // signals
#include <unistd.h> // fork
#include <sys/file.h> // flock
#include <sys/wait.h> // waitpid
#include <sys/stat.h>

#if !(defined(__linux__) || defined(__APPLE__))
#error mmap implementation as defined in POSIX.1-2001 is not supported!
#else
#include <sys/mman.h> // mmap
#endif

// MFEM_AR, MFEM_XLINKER, MFEM_XCOMPILER, MFEM_SO_EXT
// MFEM_SO_PREFIX, MFEM_SO_POSTFIX and MFEM_INSTALL_BACKUP have to defined at
// compile time.
// They are set by default in defaults.mk and MjitCmakeUtilities.cmake.

// The 'MFEM_JIT_DEBUG' environement variable can be set to:
//   - ouput dl errors,
//   - force the dlopen to resolve all symbols 'NOW', instead of lazily,
//   - keep intermediate sources files.

// The 'MFEM_JIT_VERBOSE' environement variable adds a verbose (-v) flag during
// the compilation stage.

#if !(defined(MFEM_AR) && defined(MFEM_INSTALL_BACKUP) &&\
 defined(MFEM_SO_EXT) && defined(MFEM_SO_PREFIX) && \
 defined(MFEM_SO_POSTFIX) && defined(MFEM_XLINKER) && defined(MFEM_XCOMPILER))
#error MFEM_[SO_EXT, XCOMPILER, XLINKER, AR, INSTALL_BACKUP, SO_PREFIX, SO_POSTFIX] must be defined!
#define MFEM_AR
#define MFEM_SO_EXT
#define MFEM_XLINKER
#define MFEM_XCOMPILER
#define MFEM_SO_PREFIX
#define MFEM_SO_POSTFIX
#define MFEM_INSTALL_BACKUP
#endif

namespace mfem
{

namespace jit
{

namespace time
{

template <typename T, long TIMEOUT = 200> static
void Sleep(T &&op)
{
   std::chrono::milliseconds tick(0);
   constexpr std::chrono::milliseconds tock(TIMEOUT);
   for (; op(); tick += tock) { std::this_thread::sleep_for(tock); }
}

} // namespace time

namespace io
{

static inline bool Exists(const char *path)
{
   struct stat buf;
   // stat obtains information about the file pointed to by path
   return ::stat(path, &buf) == 0; // check successful completion
}

// fcntl wrapper to provide file locks
// Warning: 'FileLock' variables must be 'named' to live during the scope
class FileLock
{
   std::string s_name;
   const char *f_name;
   std::ofstream lock;
   int fd;
   int FCntl(int cmd, int type, bool check)
   {
      struct ::flock data {};
      (data.l_type = type, data.l_whence = SEEK_SET);
      const int ret = ::fcntl(fd, cmd, &data);
      if (check) { MFEM_VERIFY(ret != -1, "[JIT] fcntl error");}
      return check ? ret : (ret != -1);
   }

public:
   // if 'now' is set, the lock will immediately happen,
   // otherwise the user must use the 'Wait' function below.
   FileLock(std::string name, const char *ext, bool now = true):
      s_name(name + "." + ext), f_name(s_name.c_str()), lock(f_name),
      fd(::open(f_name, O_RDWR))
   {
      MFEM_VERIFY(lock.good() && fd > 0, "[JIT] File lock " << f_name << " error!");
      if (now) { FCntl(F_SETLKW, F_WRLCK, true); } // wait if locked
   }

   operator bool() { return FCntl(F_SETLK, F_WRLCK, false); }

   ~FileLock() // unlock, close and remove
   { (FCntl(F_SETLK, F_UNLCK, true), ::close(fd), std::remove(f_name)); }

   void Wait() const
   {
      time::Sleep([&]() { return static_cast<bool>(std::fstream(f_name)); });
   }
};

struct Command // convenient command builder & const char* type cast
{
   Command& operator<<(const char *c) { Jit::Command() << c << ' '; return *this; }
   Command& operator<<(const std::string &s) { return *this << s.c_str(); }
   operator const char *()
   {
      static thread_local std::string sl_cmd;
      sl_cmd = Jit::Command().str();
      (Jit::Command().clear(), Jit::Command().str("")); // real flush
      return sl_cmd.c_str();
   }
};

} // namespace io

namespace mpi
{

// Return true if MPI has been initialized
static bool IsInitialized()
{
#ifndef MFEM_USE_MPI
   return false;
#else
   return Mpi::IsInitialized();
#endif
}

// Does the MPI_Init, which should be called from Mpi::Init when MFEM_USE_JIT
static int Init(int *argc, char ***argv)
{
#ifdef MFEM_USE_MPI
   return ::MPI_Init(argc, argv);
#else
   MFEM_CONTRACT_VAR(argc);
   MFEM_CONTRACT_VAR(argv);
   return EXIT_SUCCESS;
#endif
}

// Return the MPI world rank if it has been initialized, 0 otherwise
static int Rank()
{
   int world_rank = 0;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized()) { world_rank = Mpi::WorldRank(); }
#endif
   return world_rank;
}

// Return the environment MPI world rank if set, -1 otherwise
static int EnvRank()
{
   const char *mv2   = std::getenv("MV2_COMM_WORLD_RANK"); // MVAPICH2
   const char *ompi  = std::getenv("OMPI_COMM_WORLD_RANK"); // OpenMPI
   const char *mpich = std::getenv("PMI_RANK"); // MPICH
   const char *rank  = mv2 ? mv2 : ompi ? ompi : mpich ? mpich : nullptr;
   return rank ? std::stoi(rank) : -1;
}

// Return true if the rank in world rank is zero
static bool Root() { return Rank() == 0; }

// Do a MPI barrier and status reduction if MPI has been initialized
static void Sync(int status = EXIT_SUCCESS)
{
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
   }
#endif
   MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] mpi::Sync error!");
}

// Do a MPI broadcast from rank 0 if MPI has been initialized
static int Bcast(int value)
{
   int status = EXIT_SUCCESS;
#ifdef MFEM_USE_MPI
   if (Mpi::IsInitialized())
   {
      int ret = MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
      status = ret == MPI_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
   }
#endif
   MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] mpi::Bcast error!");
   return value;
}

} // namespace mpi

namespace sys
{

// Acknowledgement status values
static constexpr int ACK = ~0, CALL = 0x3243F6A8, EXIT = 0x9e3779b9;

// Acknowledge functions (EQ and NE) with thread sleep
template <typename OP> static
void AckOP(int xx) { time::Sleep([&]() { return OP()(*Jit::Ack(), xx); }); }
static void AckEQ(int xx = ACK) { AckOP<std::equal_to<int>>(xx); }
static void AckNE(int xx = ACK) { AckOP<std::not_equal_to<int>>(xx); }

// Read, Write, Acknowledge, Send & Wait using the shared memory 's_ack'
static int Read() { return *Jit::Ack(); }
static int Write(int xx) { return *Jit::Ack() = xx; }
static void Acknowledge() { Write(ACK); }

static void Send(int xx) { AckNE(Write(xx)); } // blocks until != xx
static void Wait(bool EQ = true) { EQ ? AckEQ() : AckNE(); }

// Call/Exit/Ack decode 's_ack'
static bool IsCall() { return Read() == CALL; }
static bool IsExit() { return Read() == EXIT; }
static bool IsAck() { return Read() == ACK; }

// Ask the parent to launch a system call using this command
static int Call(const char *name = nullptr, const char *command = io::Command())
{
   MFEM_VERIFY(mpi::Root(), "[JIT] Only MPI root should launch commands!");
   if (name) { MFEM_WARNING("[" << name << "] " << command); }
   // In serial mode or with the std_system option set, just call std::system
   if (!mpi::IsInitialized() || Jit::StdSystem()) { return std::system(command); }
   // Otherwise, write the command to the child process
   MFEM_VERIFY((1+std::strlen(command)) < Jit::Size(), "[JIT] length error!");
   std::memcpy(Jit::Mem(), command, std::strlen(command) + 1);
   Send(CALL); // call std::system through the child process
   Wait(false); // wait for the acknowledgment after compilation
   return EXIT_SUCCESS;
}

} // namespace mem

namespace dl
{

static const char* Error(bool show = false) noexcept
{
   const char *error = dlerror();
   if ((show || Jit::Debug()) && error) { MFEM_WARNING("[JIT] " << error); }
   MFEM_VERIFY(!::dlerror(), "[JIT] Should result in NULL being returned!");
   return error;
}

static void* Sym(void *handle, const char *name) noexcept
{
   void *sym = ::dlsym(handle, name);
   return (Error(), sym);
}

static void *Open(const char *path)
{
   const int mode = (Jit::Debug() ? RTLD_NOW : RTLD_LAZY) | RTLD_LOCAL;
   void *handle = ::dlopen(path, mode);
   return (Error(), handle);
}

} // namespace dl

} // namespace jit

using namespace jit;

// Initialize the unique global Jit variable.
Jit Jit::jit_singleton;

Jit::Jit():
   path("."),
   lib_ar("libmjit.a"),
   lib_so("./libmjit." MFEM_SO_EXT),
   keep_cache(true),
   std_system(true)
{
   Get().includes.push_back("mfem.hpp");
   Get().includes.push_back("general/forall.hpp"); // for mfem::forall
   Get().includes.push_back("general/jit/jit.hpp"); // for Hash, Find
}

Jit::~Jit() // warning: can't use mpi::Root here
{
   if (!Get().keep_cache && Rank()==0 && io::Exists(Lib_ar()))
   {
      std::remove(Lib_ar());
   }
}

/// @brief Initialize JIT, used in the MPI communication singleton.
void Jit::Init(int *argc, char ***argv)
{
   MFEM_VERIFY(!mpi::IsInitialized(), "[JIT] MPI already initialized!");

   // if MFEM_USE_JIT_FORK is set, fork root process before mpi::Init
#ifdef MFEM_USE_JIT_FORK
   Get().std_system = false;
#endif

   if (Get().std_system) // each rank does the MPI init, nothing else
   {
      mpi::Init(argc, argv);
      Get().pid = getpid(); // set ourself to be not null for finalize
      return;
   }

   // first MPI rank is looked for in the environment (-1 if not found)
   const int env_rank = mpi::EnvRank();
   if (env_rank >= 0) // MPI rank is known from environment
   {
      if (env_rank == 0) { SysInit(); } // if set, only root will use mmap
      if (env_rank > 0) // other ranks only MPI_Init
      {
         mpi::Init(argc, argv);
         Get().pid = getpid(); // set our pid for JIT finalize to be an no-op
         return;
      }
   }
   // cannot know root before MPI::Init: everyone gets ready
   else { SysInit(); }

   if ((Get().pid = ::fork()) != 0)
   {
      mpi::Init(argc, argv);
      sys::Write(mpi::Rank()); // inform the child about our rank
      sys::Wait(false); // wait for the child to acknowledge
   }
   else
   {
      MFEM_VERIFY(Pid()==0, "[JIT] Child pid error!");
      MFEM_VERIFY(sys::IsAck(), "[JIT] Child initialize state error!");
      sys::Wait(); // wait for parent's rank
      const int rank = sys::Read(); // Save the rank
      sys::Acknowledge();
      if (rank == 0) // only root is kept for system calls
      {
         while (true)
         {
            sys::Wait(); // waiting for the root to wake us
            if (sys::IsCall()) { if (std::system(Mem())) break; }
            if (sys::IsExit()) { break; }
            sys::Acknowledge();
         }
      }
      std::exit(EXIT_SUCCESS); // no children are coming back
   }
   MFEM_VERIFY(Pid()!=0, "[JIT] Children shall not pass!");
}

/// @brief Finalize JIT, used in the MPI communication singleton.
void Jit::Finalize()
{
   // child and env-ranked have nothing to do
   if (Pid() == 0 || Pid() == getpid()) { return; }
   MFEM_VERIFY(!Get().std_system, "std::system should be used!");
   MFEM_VERIFY(sys::IsAck(), "[JIT] Finalize acknowledgment error!");
   int status;
   sys::Send(sys::EXIT);
   ::waitpid(Pid(), &status, WUNTRACED | WCONTINUED); // wait for child
   MFEM_VERIFY(status == 0, "[JIT] Error with the compiler thread");
   if (::munmap(Mem(), Size()) != 0 || // release shared memory
       ::munmap(Ack(), sizeof(int)) != 0)
   { MFEM_ABORT("[JIT] Finalize memory error!"); }
}

/** @brief Set the archive name to @a name and the path to @a path.
 *  If @a keep is set to false, the cache will be removed by the MPI root.
 *  @param[in] name basename of the JIT cache, set to \c mjit by default,
 *  @param[in] path path of the JIT cache, set to '.' dy default,
 *  @param[in] keep determines if the cache will be removed or not by the MPI
 *  root rank during Jit::Finalize().
 **/
void Jit::Configure(const char *name, const char *path, bool keep)
{
   Get().path = path;
   Get().keep_cache = keep;
   Get().rank = mpi::Rank();

   auto create_full_path = [&](const char *ext)
   {
      std::string lib = Path();
      lib += std::string("/") + std::string("lib") + name;
      lib += std::string(".") + ext;
      return lib;
   };

   Get().lib_ar = create_full_path("a");
   if (mpi::Root() && !io::Exists(Lib_ar())) // if Lib_ar does not exist
   {
      MFEM_VERIFY(std::ofstream(Lib_ar()), "[JIT] Error creating " << Lib_ar());
      std::remove(Lib_ar()); // try to touch and remove
   }
   mpi::Sync();

   const char *so_ext = "" MFEM_SO_EXT; // declared on the compile line
   Get().lib_so = create_full_path(so_ext);
}

void* Jit::Lookup(const size_t hash, const char *name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *incp, const char *source, const char *symbol)
{
   dl::Error(false); // flush dl errors
   mpi::Sync(); // make sure file testing is done at the same time
   void *handle = io::Exists(Lib_so()) ? dl::Open(Lib_so()) : nullptr;
   if (!handle && io::Exists(Lib_ar())) // if .so not found, try archive
   {
      int status = EXIT_SUCCESS;
      if (mpi::Root())
      {
         io::FileLock ar_lock(Lib_ar(), "ak");
         io::FileLock so_lock(Lib_so(), "ok");
         io::Command() << cxx << link << "-shared" << "-o" << Lib_so()
                       << Xprefix() << Lib_ar() << Xpostfix()
                       << Xlinker() + std::string("-rpath,") + Path() << libs;
         status = sys::Call();
      }
      mpi::Sync(status);
      handle = dl::Open(Lib_so());
      if (!handle) // happens when Lib_so is removed in the meantime
      { return Lookup(hash, name, cxx, flags, link, libs, incp, source, symbol); }
      MFEM_VERIFY(handle, "[JIT] Error " << Lib_ar() << " => " << Lib_so());
   }

   auto WorldCompile = [&]() // but only root does the compilation
   {
      // each compilation process adds there id to the hash,
      // this is used to handle parallel compilations of the same source
      auto id = std::string("_") + std::to_string(mpi::Bcast(getpid()));
      auto so = Jit::ToString(hash, id.c_str());
      /*
      * Root lock::ck: [w-w-w-w-w-w-w-w-w-w-w-w-w-w-w-w-w]
      *            cc:  |----|Close  Delete
      *      cc => co:       |------|         Delete
      *      lock::ak:               [x-x-x-x-x-x-x-x-x-x]
      *      ar += co:                  |----|
      * (ar+co) => so:                       |---|             Delete
      *      lock::ok:                           |x-x-x|
      *  so => Lib_so:                             |--|
      */
      std::function<int(const char *)> RootCompile = [&](const char *so)
      {
         auto install = [](const char *in, const char *out)
         {
            io::Command() << "install" << Xbackup() << in << out;
            MFEM_VERIFY(sys::Call(Debug()?"install":nullptr) == EXIT_SUCCESS,
                        "[JIT] install error: " << in << " => " << out);
         };
         io::FileLock cc_lock(Jit::ToString(hash), "ck", false);
         if (cc_lock)
         {
            // Create source file: source => cc
            auto cc = Jit::ToString(hash, ".cc"); // input source
            {
               std::ofstream cc_file(cc); // open the source file
               MFEM_VERIFY(cc_file.good(), "[JIT] Source file error!");
               cc_file << source;
               cc_file.close();
            }
            // Compilation: cc => co
            auto co = Jit::ToString(hash, ".co"); // output object
            {
               MFEM_VERIFY(io::Exists(MFEM_SOURCE_DIR "/mfem.hpp") ||
                           io::Exists(MFEM_INSTALL_DIR "/include/mfem/mfem.hpp"),
                           "[JIT] Could not find any MFEM header!");
               std::string mfem_inst_inc_dir(MFEM_INSTALL_DIR "/include/mfem");
               io::Command() << cxx << flags << (Verbose() ? "-v" : "")
                             << "-I" << MFEM_SOURCE_DIR
                             << "-I" << mfem_inst_inc_dir
                             << "-I" << mfem_inst_inc_dir + "/" + incp
                             << Includes().c_str()
#ifdef MFEM_USE_CUDA
                             // nvcc option to embed relocatable device code
                             << "--relocatable-device-code=true"
#endif
                             << "-c" << "-o" << co << cc;
               if (sys::Call(name)) { return EXIT_FAILURE; }
               if (!Debug()) { std::remove(cc.c_str()); }
            }
            // Update archive: ar += co
            io::FileLock ar_lock(Lib_ar(), "ak");
            {
               io::Command() << ("" MFEM_AR) << "-r" << Lib_ar() << co; // v
               if (sys::Call(Debug() ? name : nullptr)) { return EXIT_FAILURE; }
               if (!Debug()) { std::remove(co.c_str()); }
            }
            // Create temporary shared library: (ar + co) => so
            {
               // macos warns dynamic_lookup may not work with chained fixups
               // to avoid, use both MFEM_SOURCE/INSTALL_DIR directly
               io::Command() << cxx << link << "-o" << so
                             << "-shared"
                             << "-L" MFEM_SOURCE_DIR
                             << "-L" MFEM_INSTALL_DIR " -lmfem"
                             << Xprefix() << Lib_ar() << Xpostfix()
                             << Xlinker() + std::string("-rpath,") + Path()
                             << libs;
               if (sys::Call(Debug() ? name : nullptr)) { return EXIT_FAILURE; }
            }
            // Install temporary shared library: so => Lib_so
            io::FileLock so_lock(Lib_so(), "ok");
            install(so, Lib_so());
         }
         else // avoid duplicate compilation
         {
            cc_lock.Wait();
            // if removed, rerun the compilation
            if (!io::Exists(Lib_so())) { return RootCompile(so); }
            io::FileLock so_lock(Lib_so(), "ok");
            install(Lib_so(), so);
         }
         return EXIT_SUCCESS;
      };
      const int status = mpi::Root() ? RootCompile(so.c_str()) : EXIT_SUCCESS;
      MFEM_VERIFY(status == EXIT_SUCCESS, "[JIT] RootCompile error!");
      mpi::Sync(status); // all ranks verify the status
      std::string symbol_path(Path() + "/");
      handle = dl::Open((symbol_path + so).c_str()); // opens symbol
      mpi::Sync();
      MFEM_VERIFY(handle, "[JIT] Error creating handle!");
      if (mpi::Root()) { std::remove(so.c_str()); }
   }; // WorldCompile

   // no cache => launch compilation
   if (!handle) { WorldCompile(); }
   MFEM_VERIFY(handle, "[JIT] No handle created!");
   void *kernel = dl::Sym(handle, symbol); // symbol lookup

   // no symbol => launch compilation & update kernel symbol
   if (!kernel) { WorldCompile(); kernel = dl::Sym(handle, symbol); }
   MFEM_VERIFY(kernel, "[JIT] No kernel found!");
   return kernel;
}

std::string Jit::Xlinker()   { return "" MFEM_XLINKER; }
std::string Jit::Xcompiler() { return "" MFEM_XCOMPILER; }
std::string Jit::Xprefix()   { return "" MFEM_SO_PREFIX;  }
std::string Jit::Xpostfix()  { return "" MFEM_SO_POSTFIX; }
std::string Jit::Xbackup()   { return "" MFEM_INSTALL_BACKUP; }

bool Jit::Debug() { return !!std::getenv("MFEM_JIT_DEBUG"); }
bool Jit::Verbose() { return !!std::getenv("MFEM_JIT_VERBOSE"); }

std::string Jit::Includes()
{
   std::string incs;
   for (auto inc: Get().includes) { incs += "-include \"" + inc + "\" "; }
   return incs;
}

// Initialisation of the shared memory between the MPI root and the
// thread that will laucnh the 'system' commands
void Jit::SysInit()
{
   MFEM_VERIFY(!Jit::StdSystem(), "std::system should be used!");
   constexpr int prot = PROT_READ | PROT_WRITE;
   constexpr int flags = MAP_SHARED | MAP_ANONYMOUS;
   Jit::Get().size = (uintptr_t) sysconf(_SC_PAGE_SIZE);
   Jit::Get().s_ack = (int*) ::mmap(nullptr, sizeof(int), prot, flags, -1, 0);
   Jit::Get().s_mem = (char*) ::mmap(nullptr, Jit::Get().size, prot, flags, -1, 0);
   sys::Write(sys::ACK); // initialize state
}

} // namespace mfem

#endif // MFEM_USE_JIT