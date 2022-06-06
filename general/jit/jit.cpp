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
#include "../globals.hpp"
#include "../error.hpp"
#include "jit.hpp"

#include <string>
#include <fstream>
#include <thread> // sleep_for
#include <chrono> // milliseconds

#include <cstring> // strlen
#include <cstdlib> // exit, system
#include <dlfcn.h> // dlopen/dlsym, not available on Windows
#include <signal.h> // signals
#include <unistd.h> // fork
#include <sys/file.h> // flock
#include <sys/wait.h> // waitpid

#if !(defined(__linux__) || defined(__APPLE__))
#error mmap(2) implementation as defined in POSIX.1-2001 not supported.
#else
#include <sys/mman.h> // mmap
#endif

namespace mfem
{

namespace jit
{

namespace io
{

class FileLock // need to be 'named' to live during the scope
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
      if (check) { MFEM_VERIFY(ret != -1, "fcntl error");}
      return check ? ret : (ret != -1);
   }

public:
   FileLock(std::string name, const char *ext, bool now = true):
      s_name(name + "." + ext),
      f_name(s_name.c_str()),
      lock(f_name),
      fd(::open(f_name, O_RDWR))
   {
      MFEM_VERIFY(lock.good() && fd > 0, "[FileLock] " << f_name << " error!");
      if (now) { FCntl(F_SETLKW, F_WRLCK, true); } // wait if locked
   }

   operator bool() { return FCntl(F_SETLK, F_WRLCK, false); }

   ~FileLock() // unlock, close and remove
   {
      FCntl(F_SETLK, F_UNLCK, true);
      ::close(fd);
      std::remove(f_name);
      MFEM_VERIFY(!std::fstream(f_name), "[~FileLock] " << f_name << " error!");
   }

   void Wait() const
   {
      while (std::fstream(f_name))
      { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
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

class System // System singleton object
{
   pid_t pid; // of child process
   int *s_ack, rank; // shared status, must be able to store one MPI rank
   char *s_mem; // shared memory to store the command for the system call
   uintptr_t size; // of the s_mem shared memory
   std::string path {"."};
   std::string lib_ar {"libmjit.a"}, lib_so {"./libmjit.so"};
   bool keep = true; // keep lib_ar

   struct Command
   {
      std::ostringstream cmd {};
      Command& operator<<(const char *c) { cmd << c << ' '; return *this; }
      Command& operator<<(const std::string &s) { return *this << s.c_str(); }
      operator const char *()
      {
         std::ostringstream cmd_mv = std::move(cmd);
         static thread_local std::string sl_cmd;
         sl_cmd = cmd_mv.str();
         cmd.clear(); cmd.str(""); // flush for next command
         return sl_cmd.c_str();
      }
   } command;

   template <typename OP> static void Ack(int xx) // spinlock
   { while (OP()(*Ack(), xx)) { Sleep(); } }
   static void AckEQ(int xx = ACK) { Ack<std::equal_to<int>>(xx); }
   static void AckNE(int xx = ACK) { Ack<std::not_equal_to<int>>(xx); }
   static constexpr int ACK = ~0, CALL = 0x3243F6A8, EXIT = 0x9e3779b9;

   static int Read() { return *Ack(); }
   static int Write(int xx) { return *Get().s_ack = xx; }
   static void Acknowledge() { Write(ACK); }
   static void Send(int xx) { AckNE(Write(xx)); } // blocks until != xx
   static void Wait(bool EQ = true) { EQ ? AckEQ() : AckNE(); }

   static bool IsCall() { return Read() == CALL; }
   static bool IsExit() { return Read() == EXIT; }
   static bool IsAck() { return Read() == ACK; }
   static void Sleep(int ms = 200)
   { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

   // Ask the parent to launch a system call using the prepared command
   static int Call(const char *name = nullptr, const char *command = Command())
   {
      MFEM_VERIFY(mpi::Root(), "[JIT] Only MPI root should launch commands!");
      if (name) { MFEM_WARNING("[" << name << "] " << command); }
      // In serial mode, just call std::system
      if (!mpi::IsInitialized()) { return std::system(command); }
      // Otherwise, write the command to the child process
      MFEM_VERIFY((1+std::strlen(command))<Size(), "[JIT] Command length error!");
      std::memcpy(Mem(), command, std::strlen(command) + 1);
      Send(CALL); // call std::system through the child process
      Wait(false); // wait for the acknowledgment after compilation
      return EXIT_SUCCESS;
   }

   static System singleton;
   static System& Get() { return singleton; }

   ~System() // can't use mpi::Root
   {
      if (!Get().keep && (Get().rank==0) && std::fstream(Lib_ar()))
      {
         io::FileLock ar_lock(Lib_ar(), "ak");
         std::remove(Lib_ar());

      }
      if ((Get().rank==0) && std::fstream(Lib_so()))
      {
         io::FileLock so_lock(Lib_so(), "ok");
         std::remove(Lib_so());
      }
   }

   static pid_t Pid() { return Get().pid; }
   static int* Ack() { return Get().s_ack; }
   static char* Mem() { return Get().s_mem; }
   static uintptr_t Size() { return Get().size; }
   static Command& Command() { return Get().command; }

   static void MmapInit()
   {
      constexpr int prot = PROT_READ | PROT_WRITE;
      constexpr int flags = MAP_SHARED | MAP_ANONYMOUS;
      Get().size = (uintptr_t) sysconf(_SC_PAGE_SIZE);
      Get().s_ack = (int*) ::mmap(nullptr, sizeof(int), prot, flags, -1, 0);
      Get().s_mem = (char*) ::mmap(nullptr, Get().size, prot, flags, -1, 0);
      Write(ACK); // initialize state
   }

public:
   static void Init(int *argc, char ***argv)
   {
      MFEM_CONTRACT_VAR(argc);
      MFEM_CONTRACT_VAR(argv);
      MFEM_VERIFY(!mpi::IsInitialized(), "MPI should not be initialized yet!");
      const int env_rank = mpi::EnvRank(); // first env rank is sought for
      if (env_rank >= 0)
      {
         if (env_rank == 0) { MmapInit(); } // if set, only root will use mmap
         if (env_rank > 0) // other ranks only MPI_Init
         {
#ifdef MFEM_USE_MPI
            ::MPI_Init(argc, argv);
#endif
            Get().pid = getpid(); // set ourself to be not null for finalize
            return;
         }
      }
      else { MmapInit(); } // everyone gets ready

      if ((Get().pid = ::fork()) != 0)
      {
#ifdef MFEM_USE_MPI
         ::MPI_Init(argc, argv);
#endif
         Write(mpi::Rank()); // inform the child about our rank
         Wait(false); // wait for the child to acknowledge
      }
      else
      {
         MFEM_VERIFY(Pid()==0, "[JIT] Child pid error!");
         MFEM_VERIFY(IsAck(), "[JIT] Child initialize state error!");
         Wait(); // wait for parent's rank
         const int rank = Read(); // Save the rank
         Acknowledge();
         if (rank == 0) // only root is kept for system calls
         {
            while (true)
            {
               Wait(); // waiting for the root to wake us
               if (IsCall()) { if (std::system(Mem())) break; }
               if (IsExit()) { break;}
               Acknowledge();
            }
         }
         std::exit(EXIT_SUCCESS); // no children are coming back
      }
      MFEM_VERIFY(Pid()!=0, "Children shall not pass!");
   }

   static void Configure(const char *name, const char *path, bool keep)
   {
      Get().path = path;
      Get().keep = keep;
      Get().rank = mpi::Rank();

      auto full_path = [&](const char *ext)
      {
         std::string lib = std::string(Get().path);
         lib += std::string("/") + std::string("lib") + name;
         lib += std::string(".") + ext;
         return lib;
      };

      Get().lib_ar = full_path("a");
      if (!std::fstream(Get().lib_ar))
      {
         MFEM_VERIFY(std::ofstream(Get().lib_ar),
                     "Error Could not create " << Get().lib_ar);
         std::remove(Get().lib_ar.c_str());
      }

#ifdef __APPLE__
      const char *so_ext = "dylib";
#else
      const char *so_ext = "so";
#endif
      Get().lib_so = full_path(so_ext);
   }

   static void Finalize()
   {
      // child and env-ranked have nothing to do
      if (Pid()==0 || Pid()==getpid()) { return; }
      MFEM_VERIFY(IsAck(), "[JIT] Finalize acknowledgment error!");
      int status;
      Send(EXIT);
      ::waitpid(Pid(), &status, WUNTRACED | WCONTINUED); // wait for child
      MFEM_VERIFY(status == 0, "[JIT] Error with the compiler thread");
      if (::munmap(Mem(), Size()) != 0 || // release shared memory
          ::munmap(Ack(), sizeof(int)) != 0)
      { MFEM_ABORT("[JIT] Finalize memory error!"); }
   }

   struct CompilerOptions
   {
      virtual std::string Compiler() { return ""; }
      virtual std::string Pic() { return "-fPIC"; }
      virtual std::string Pipe() { return "-pipe"; }
      virtual std::string Device() { return ""; }
      virtual std::string Linker() { return "-Wl,"; }
   };

   struct NvccOptions: CompilerOptions
   {
      std::string Compiler() override { return "-Xcompiler="; }
      std::string Pic() override { return Xcompiler() + "-fPIC"; }
      std::string Pipe() override { return ""; } // not supported
      std::string Device() override { return "--device-c"; }
      std::string Linker() override { return "-Xlinker="; }
   };

   struct LinkerOptions
   {
      virtual std::string Backup() { return "--backup=none"; }
      virtual std::string Prefix() { return Xlinker() + "--whole-archive"; }
      virtual std::string Postfix() { return Xlinker() + "--no-whole-archive"; }
   };

   struct DarwinOptions: public LinkerOptions
   {
      std::string Backup() override { return ""; }
      std::string Prefix() override { return "-all_load"; }
      std::string Postfix() override { return ""; }
   };

#ifdef MFEM_USE_CUDA
   NvccOptions cxx;
#else
   CompilerOptions cxx;
#endif

#ifdef __APPLE__
   DarwinOptions ar;
#else
   LinkerOptions ar;
#endif

   static const char *Lib_ar() { return Get().lib_ar.c_str(); }
   static const char *Lib_so() { return Get().lib_so.c_str(); }
   static std::string Xpic() { return Get().cxx.Pic(); }
   static std::string Xpipe() { return Get().cxx.Pipe(); }
   static std::string Xdevice() { return Get().cxx.Device(); }
   static std::string Xlinker() { return Get().cxx.Linker(); }
   static std::string Xcompiler() { return Get().cxx.Compiler(); }
   static std::string ARprefix() { return Get().ar.Prefix();  }
   static std::string ARpostfix() { return Get().ar.Postfix(); }
   static std::string ARbackup() { return Get().ar.Backup(); }

   static const char* DLerror(bool show = true) noexcept
   {
      const char* last_error = dlerror();
#ifndef MFEM_DEBUG
      MFEM_CONTRACT_VAR(show);
#else
      if (show && last_error) { MFEM_WARNING("[JIT] " << last_error); }
      MFEM_VERIFY(!dlerror(), "[JIT] Should result in NULL being returned!");
#endif
      return last_error;
   }

   static void* DLsym(void *handle, const char *name) noexcept
   {
      void *sym = ::dlsym(handle, name);
      DLerror();
      return sym;
   }

   static void *DLopen(const char *path)
   {
      void *handle = ::dlopen(path, RTLD_LAZY | RTLD_LOCAL);
      DLerror();
      return handle;
   }

   static void* Lookup(const size_t hash, const char *name, const char *cxx,
                       const char *flags, const char *link, const char *libs,
                       const char *source, const char *symbol)
   {
      DLerror(false); // flush dl errors

      void *handle = std::fstream(Lib_so()) ? DLopen(Lib_so()) : nullptr;
      if (!handle && std::fstream(Lib_ar())) // if .so not found, try archive
      {
         for (int status = EXIT_SUCCESS; !handle; status = EXIT_SUCCESS) // timeout ?
         {
            if (mpi::Root())
            {
               io::FileLock so_lock(Lib_so(), "ok");
               Command() << cxx << link << "-shared" << "-o" << Lib_so()
                         << ARprefix() << Lib_ar() << ARpostfix()
                         << Xlinker() + "-rpath,." << libs;
               status = Call();
            }
            mpi::Sync(status);
            handle = DLopen(Lib_so()); // can be removed in the meantime
         }
         MFEM_VERIFY(handle, "[JIT] Error " << Lib_so() << " from " << Lib_ar());
      }

      auto WorldCompile = [&]() // but only root compiles
      {
         auto tid = std::string("_") + std::to_string(mpi::Bcast(getpid()));
         auto tmp = Jit::ToString(hash, tid.c_str());
         /**
         * Root        ck: [w-w-w-w-w-w-w-w-w-w-w-w-w-w-w-w-w]
         *             cc:  |----|Close  Delete
         *       cc => co:       |------|         Delete
         *             ak:               [x-x-x-x-x-x-x-x-x-x]
         *       ar += co:                  |----|
         * (ar+co) => tmp:                       |---|             Delete
         *             ok:                           |x-x-x|
         *      tmp => so:                             |--|
         *---------------------------------------------------------
         * Lock so => tmp:                                   |---| Delete
         **/
         std::function<int(const char *)> RootCompile = [&](const char *tmp)
         {
            auto install = [](const char *in, const char *out)
            {
               Command() << "install" << ARbackup() << in << out;
               MFEM_VERIFY(Call() == EXIT_SUCCESS,
                           "[JIT] install error: " << in << " => " << out);
            };
            io::FileLock cc_lock(Jit::ToString(hash), "ck", false);
            if (cc_lock)
            {
               // Write kernel source file
               auto cc = Jit::ToString(hash, ".cc"); // input source
               std::ofstream source_file(cc); // open the source file
               MFEM_VERIFY(source_file.good(), "[JIT] Source file error!");
               source_file << source;
               source_file.close();
               // Compilation: cc => co
               const char *bin_mfem_hpp = MFEM_INSTALL_DIR "/include/mfem/mfem.hpp";
               const auto bin_mfem_file = std::fstream(bin_mfem_hpp);
               const char *src_mfem_hpp = MFEM_SOURCE_DIR "/mfem.hpp";
               const auto src_mfem_file = std::fstream(src_mfem_hpp);
               MFEM_VERIFY(bin_mfem_file||src_mfem_file, "MFEM header needed!");
               auto co = Jit::ToString(hash, ".co"); // output object
               Command() << cxx << flags
                         << "-I" << MFEM_INSTALL_DIR "/include/mfem"
                         << "-I" << MFEM_SOURCE_DIR
                         << Xdevice() << Xpic() << Xpipe()
                         << "-c" << "-o" << co << cc
                         << (std::getenv("MFEM_JIT_VERBOSE") ? "-v" : "");
               if (Call(name)) { return EXIT_FAILURE; }
               std::remove(cc.c_str());
               // Update archive: ar += co
               io::FileLock ar_lock(Lib_ar(), "ak");
               Command() << "ar -rv" << Lib_ar() << co;
               if (Call()) { return EXIT_FAILURE; }
               std::remove(co.c_str());
               // Create temporary shared library: (ar + co) => tmp
               Command() << cxx << link << "-shared" << "-o" << tmp
                         << ARprefix() << Lib_ar() << ARpostfix()
                         << Xlinker() + "-rpath,." << libs;
               if (Call(name)) { return EXIT_FAILURE; }
               // Install temporary shared library: tmp => so
               io::FileLock so_lock(Lib_so(), "ok");
               install(tmp, Lib_so());
            }
            else // avoid duplicate compilation
            {
               cc_lock.Wait();
               if (!std::fstream(Lib_so())) { RootCompile(tmp);} // if removed
               install(Lib_so(), tmp);
            }
            return EXIT_SUCCESS;
         };
         const int status = mpi::Root() ? RootCompile(tmp.c_str()) : EXIT_SUCCESS;
         MFEM_VERIFY(status == EXIT_SUCCESS, "!EXIT_SUCCESS");
         mpi::Sync(status); // all ranks verify the status
         std::string symbol_path("./");
         handle = DLopen((symbol_path + tmp).c_str()); // opens symbol
         mpi::Sync();
         MFEM_VERIFY(handle, "[JIT] Error creating handle:" << ::dlerror());
         if (mpi::Root()) { std::remove(tmp.c_str()); }
      }; // WorldCompile

      // no cache => launch compilation
      if (!handle) { WorldCompile(); }
      MFEM_VERIFY(handle, "[JIT] No handle could be created!");
      void *kernel = DLsym(handle, symbol); // symbol lookup

      // no symbol => launch compilation & update kernel symbol
      if (!kernel) { WorldCompile(); kernel = DLsym(handle, symbol); }
      MFEM_VERIFY(kernel, "[JIT] No kernel could be found!");
      return kernel;
   }
};
System System::singleton {}; // Initialize the unique System context.

} // namespace jit

using namespace jit;

void Jit::Init(int *argc, char ***argv) { System::Init(argc, argv); }

void Jit::Configure(const char *name, const char *path, bool keep)
{
   System::Configure(name, path, keep);
}

void Jit::Finalize() { System::Finalize(); }

void* Jit::Lookup(const size_t hash, const char *name, const char *cxx,
                  const char *flags, const char *link, const char *libs,
                  const char *source, const char *symbol)
{
   return System::Lookup(hash, name, cxx, flags, link, libs, source, symbol);
}

} // namespace mfem

#endif // MFEM_USE_JIT
