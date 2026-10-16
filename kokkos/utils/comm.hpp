#ifndef _comm_hpp_
#define _comm_hpp_

#include <algorithm>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include <Parameters.hpp>

namespace miniFE {

#ifdef HAVE_KOKKOS_COMM
struct CommBootstrap {
#if defined(KOKKOSCOMM_ENABLE_NCCL)
  KokkosComm::DefaultCommunicationSpace::communicator_type raw_comm = nullptr;
#endif
};

struct CommContext {
  using comm_type = KokkosComm::Communicator<>;
  using request_type = KokkosComm::Request<>;

  comm_type comm;
  CommBootstrap bootstrap;

  int rank() const { return comm.rank(); }
  int size() const { return comm.size(); }
};

void initialize_comm(const Parameters& params, int numprocs, int myproc);

void finalize_comm();

CommContext& get_comm();

#endif

void initialize_mpi(int argc, char** argv, int& numprocs, int& myproc);

void finalize_mpi();

int num_ranks();

int this_rank();

namespace detail {

template<typename ValueType>
auto make_unmanaged_host_view(const ValueType* data, std::size_t count)
{
  return Kokkos::View<const ValueType*, Kokkos::LayoutLeft,
                      typename Kokkos::DefaultHostExecutionSpace::memory_space,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, count);
}

template<typename ValueType>
auto make_unmanaged_host_view(ValueType* data, std::size_t count)
{
  return Kokkos::View<ValueType*, Kokkos::LayoutLeft,
                      typename Kokkos::DefaultHostExecutionSpace::memory_space,
                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data, count);
}

} // namespace detail

template<typename ValueType>
void broadcast(ValueType* data, std::size_t count, int root = 0)
{
  if (count == 0 || num_ranks() == 1) {
    return;
  }

#ifdef HAVE_KOKKOS_COMM
  auto view = detail::make_unmanaged_host_view(data, count);
  KokkosComm::Experimental::broadcast(get_comm().comm, view, root).wait();
#else
  (void)data;
  (void)root;
#endif
}

template<typename ValueType>
void allgather(const ValueType* send_data,
               std::size_t send_count,
               ValueType* recv_data)
{
  if (send_count == 0) {
    return;
  }

  if (num_ranks() == 1) {
    std::copy(send_data, send_data + send_count, recv_data);
    return;
  }

#ifdef HAVE_KOKKOS_COMM
  auto send_view = detail::make_unmanaged_host_view(send_data, send_count);
  auto recv_view = detail::make_unmanaged_host_view(recv_data, send_count * num_ranks());
  KokkosComm::Experimental::allgather(get_comm().comm, send_view, recv_view).wait();
#endif
}

template<typename ValueType, typename ReductionOp>
void allreduce(const ValueType* send_data,
               ValueType* recv_data,
               std::size_t count,
               ReductionOp op)
{
  if (count == 0) {
    return;
  }

  if (num_ranks() == 1) {
    std::copy(send_data, send_data + count, recv_data);
    return;
  }

#ifdef HAVE_KOKKOS_COMM
  auto send_view = detail::make_unmanaged_host_view(send_data, count);
  auto recv_view = detail::make_unmanaged_host_view(recv_data, count);
  KokkosComm::Experimental::allreduce(get_comm().comm, send_view, recv_view, op).wait();
#else
  (void)op;
#endif
}

template<typename ValueType, typename ReductionOp>
ValueType allreduce_scalar(const ValueType& value, ReductionOp op)
{
  if (num_ranks() == 1) {
    return value;
  }

  ValueType result{};
#ifdef HAVE_KOKKOS_COMM
  Kokkos::View<const ValueType, typename Kokkos::DefaultHostExecutionSpace::memory_space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>> send_host_view(&value);
  Kokkos::View<ValueType, typename Kokkos::DefaultHostExecutionSpace::memory_space,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>> recv_host_view(&result);
  KokkosComm::Experimental::allreduce(get_comm().comm, send_host_view, recv_host_view, op).wait();
#else
  (void)op;
#endif
  return result;
}

#ifdef HAVE_KOKKOS_COMM
inline void wait_all(std::vector<CommContext::request_type>& requests)
{
  if (requests.empty()) {
    return;
  }

  KokkosComm::wait_all(std::span<CommContext::request_type>(requests.data(), requests.size()));
}
#endif

inline void sync_ranks()
{
  int token = 0;
  int unused = 0;
  allreduce(&token, &unused, 1, KokkosComm::Sum{});
}

} // namespace miniFE

#endif
