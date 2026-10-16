#include <stdexcept>
#include <type_traits>

#include <memory>

#include <comm.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
#include <nccl.h>
#include <cuda_runtime.h>
#endif

namespace miniFE {

namespace {

bool g_initialized_mpi = false;

#ifdef HAVE_KOKKOS_COMM
std::unique_ptr<CommContext> g_comm;
#endif

#if defined(HAVE_MPI)
int get_local_rank(MPI_Comm comm, int my_rank)
{
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, my_rank, MPI_INFO_NULL, &node_comm);

  int node_rank = 0;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_free(&node_comm);
  return node_rank;
}
#endif

#if defined(KOKKOSCOMM_ENABLE_NCCL)
void check_cuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(what);
  }
}

void check_nccl(ncclResult_t err, const char* what)
{
  if (err != ncclSuccess) {
    throw std::runtime_error(what);
  }
}
#endif

} // namespace

void initialize_mpi(int argc, char** argv, int& numprocs, int& myproc)
{
#ifdef HAVE_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(&argc, &argv);
    g_initialized_mpi = true;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#else
  (void)argc;
  (void)argv;
  numprocs = 1;
  myproc = 0;
#endif
}

#ifdef HAVE_KOKKOS_COMM
CommBootstrap prepare_comm_bootstrap(const Parameters& params, int numprocs, int myproc)
{
  (void)params;
  CommBootstrap bootstrap;

#if defined(KOKKOSCOMM_ENABLE_NCCL) && defined(HAVE_MPI)
  if constexpr (std::is_same_v<KokkosComm::DefaultCommunicationSpace, KokkosComm::Experimental::NcclSpace>) {
    int local_rank = get_local_rank(MPI_COMM_WORLD, myproc);
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "miniFE: failed to query CUDA devices for NCCL bootstrap");
    if (device_count < 1) {
      throw std::runtime_error("miniFE: NCCL default communication requires at least one CUDA device per node");
    }

    int device = local_rank % device_count;
    check_cuda(cudaSetDevice(device), "miniFE: failed to select CUDA device for NCCL bootstrap");

    ncclUniqueId id{};
    if (myproc == 0) {
      check_nccl(ncclGetUniqueId(&id), "miniFE: failed to create NCCL unique id");
    }

    MPI_Bcast(&id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
    check_nccl(ncclCommInitRank(&bootstrap.raw_comm, numprocs, id, myproc),
               "miniFE: failed to initialize NCCL communicator");
  }
#endif

  return bootstrap;
}

CommContext build_comm(CommBootstrap&& bootstrap)
{
  auto exec = Kokkos::DefaultExecutionSpace{};

#if defined(KOKKOSCOMM_ENABLE_NCCL)
  if constexpr (std::is_same_v<KokkosComm::DefaultCommunicationSpace, KokkosComm::Experimental::NcclSpace>) {
    return CommContext{KokkosComm::Communicator<>::from_raw(bootstrap.raw_comm, exec), std::move(bootstrap)};
  }
#endif

  return CommContext{KokkosComm::Communicator<>::from_raw(MPI_COMM_WORLD, exec), std::move(bootstrap)};
}

void initialize_comm(const Parameters& params, int numprocs, int myproc)
{
  auto bootstrap = prepare_comm_bootstrap(params, numprocs, myproc);
  g_comm = std::make_unique<CommContext>(build_comm(std::move(bootstrap)));
}

void finalize_comm()
{
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  if constexpr (std::is_same_v<KokkosComm::DefaultCommunicationSpace, KokkosComm::Experimental::NcclSpace>) {
    if (g_comm && g_comm->bootstrap.raw_comm != nullptr) {
      ncclCommDestroy(g_comm->bootstrap.raw_comm);
      g_comm->bootstrap.raw_comm = nullptr;
    }
  }
#endif

  g_comm.reset();
}

CommContext& get_comm()
{
  if (!g_comm) {
    throw std::runtime_error("miniFE: KokkosComm context requested before initialization");
  }

  return *g_comm;
}
#endif

void finalize_mpi()
{
#ifdef HAVE_MPI
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (g_initialized_mpi && !finalized) {
    MPI_Finalize();
  }
#endif
}

int num_ranks()
{
#ifdef HAVE_KOKKOS_COMM
  return get_comm().size();
#else
  return 1;
#endif
}

int this_rank()
{
#ifdef HAVE_KOKKOS_COMM
  return get_comm().rank();
#else
  return 0;
#endif
}

} // namespace miniFE
