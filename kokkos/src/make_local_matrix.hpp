#ifndef _make_local_matrix_hpp_
#define _make_local_matrix_hpp_

//@HEADER
// ************************************************************************
//
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#include <Kokkos_Subview.hpp>

#include <comm.hpp>

namespace miniFE {

template<typename MatrixType>
void make_local_matrix(MatrixType& A)
{
#ifdef HAVE_KOKKOS_COMM
  const int numprocs = num_ranks();
  const int myproc = this_rank();

  if (numprocs < 2) {
    A.num_cols = A.rows.size();
    A.has_local_indices = true;
    return;
  }

  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::ScalarType Scalar;

  std::map<GlobalOrdinal, GlobalOrdinal> externals;
  LocalOrdinal num_external = 0;

  const size_t local_nrow = A.rows.size();
  const GlobalOrdinal start_row = local_nrow > 0 ? A.rows[0] : -1;
  const GlobalOrdinal stop_row = local_nrow > 0 ? A.rows[local_nrow - 1] : -1;

  Kokkos::vector<GlobalOrdinal, Kokkos::DefaultExecutionSpace>& external_index = A.external_index;

  for (size_t i = 0; i < A.rows.size(); ++i) {
    GlobalOrdinal* Acols = NULL;
    Scalar* Acoefs = NULL;
    size_t row_len = 0;
    A.get_row_pointers(A.rows[i], row_len, Acols, Acoefs);
    (void)Acoefs;
    for (size_t j = 0; j < row_len; ++j) {
      GlobalOrdinal cur_ind = Acols[j];
      if (start_row <= cur_ind && cur_ind <= stop_row) {
        Acols[j] -= start_row;
      } else {
        if (externals.find(cur_ind) == externals.end()) {
          externals[cur_ind] = num_external++;
          external_index.push_back(cur_ind);
        }
        Acols[j] = -(Acols[j] + 1);
      }
    }
  }

  std::vector<GlobalOrdinal> global_index_offsets(numprocs, 0);
  std::vector<GlobalOrdinal> start_rows(numprocs, 0);
  start_rows[myproc] = start_row;
  allreduce(start_rows.data(), global_index_offsets.data(), static_cast<std::size_t>(numprocs), KokkosComm::Sum{});

  Kokkos::vector<int, Kokkos::DefaultExecutionSpace> external_processor(num_external);
  for (LocalOrdinal i = 0; i < num_external; ++i) {
    const GlobalOrdinal cur_ind = external_index[i];
    for (int j = numprocs - 1; j >= 0; --j) {
      if (global_index_offsets[j] <= cur_ind && global_index_offsets[j] >= 0) {
        external_processor[i] = j;
        break;
      }
    }
  }

  size_t count = local_nrow;
  Kokkos::vector<GlobalOrdinal, Kokkos::DefaultExecutionSpace>& external_local_index = A.external_local_index;
  external_local_index.on_host();
  external_local_index.assign(num_external, -1);

  for (LocalOrdinal i = 0; i < num_external; ++i) {
    if (external_local_index[i] == -1) {
      external_local_index[i] = count++;
      for (LocalOrdinal j = i + 1; j < num_external; ++j) {
        if (external_processor[j] == external_processor[i]) {
          external_local_index[j] = count++;
        }
      }
    }
  }

  for (size_t i = 0; i < local_nrow; ++i) {
    GlobalOrdinal* Acols = NULL;
    Scalar* Acoefs = NULL;
    size_t row_len = 0;
    A.get_row_pointers(A.rows[i], row_len, Acols, Acoefs);
    (void)Acoefs;

    for (size_t j = 0; j < row_len; ++j) {
      if (Acols[j] < 0) {
        const GlobalOrdinal cur_ind = -Acols[j] - 1;
        Acols[j] = external_local_index[externals[cur_ind]];
      }
    }
  }

  Kokkos::vector<int, Kokkos::DefaultExecutionSpace> new_external_processor(num_external, 0);
  for (LocalOrdinal i = 0; i < num_external; ++i) {
    new_external_processor[external_local_index[i] - local_nrow] = external_processor[i];
  }

  std::vector<int> recv_list;
  std::vector<int> recv_lengths;
  recv_list.reserve(num_external);
  recv_lengths.reserve(num_external);
  if (num_external > 0) {
    recv_list.push_back(new_external_processor[0]);
    recv_lengths.push_back(1);
    for (LocalOrdinal i = 1; i < num_external; ++i) {
      if (new_external_processor[i - 1] != new_external_processor[i]) {
        recv_list.push_back(new_external_processor[i]);
        recv_lengths.push_back(1);
      } else {
        ++recv_lengths.back();
      }
    }
  }

  std::vector<int> send_counts(numprocs, 0);
  for (size_t i = 0; i < recv_list.size(); ++i) {
    send_counts[recv_list[i]] = recv_lengths[i];
  }

  std::vector<int> all_send_counts(numprocs * numprocs, 0);
  allgather(send_counts.data(), static_cast<std::size_t>(numprocs), all_send_counts.data());

  std::vector<int> send_list;
  std::vector<int> send_lengths_host;
  int total_to_be_sent = 0;
  for (int proc = 0; proc < numprocs; ++proc) {
    const int count_from_proc = all_send_counts[proc * numprocs + myproc];
    if (count_from_proc > 0) {
      send_list.push_back(proc);
      send_lengths_host.push_back(count_from_proc);
      total_to_be_sent += count_from_proc;
    }
  }

  std::vector<int> combined_neighbors = recv_list;
  for (int peer : send_list) {
    if (std::find(combined_neighbors.begin(), combined_neighbors.end(), peer) == combined_neighbors.end()) {
      combined_neighbors.push_back(peer);
    }
  }

  std::sort(combined_neighbors.begin(), combined_neighbors.end());

  Kokkos::vector<int, Kokkos::DefaultExecutionSpace>& neighbors = A.neighbors;
  Kokkos::vector<int, Kokkos::DefaultExecutionSpace>& recv_length = A.recv_length;
  Kokkos::vector<int, Kokkos::DefaultExecutionSpace>& send_length = A.send_length;
  neighbors.assign(combined_neighbors.size(), 0);
  recv_length.assign(combined_neighbors.size(), 0);
  send_length.assign(combined_neighbors.size(), 0);

  for (size_t i = 0; i < combined_neighbors.size(); ++i) {
    neighbors[i] = combined_neighbors[i];

    const auto recv_it = std::find(recv_list.begin(), recv_list.end(), combined_neighbors[i]);
    if (recv_it != recv_list.end()) {
      recv_length[i] = recv_lengths[std::distance(recv_list.begin(), recv_it)];
    }

    const auto send_it = std::find(send_list.begin(), send_list.end(), combined_neighbors[i]);
    if (send_it != send_list.end()) {
      send_length[i] = send_lengths_host[std::distance(send_list.begin(), send_it)];
    }
  }

  A.elements_to_send.assign(total_to_be_sent, 0);
  A.send_buffer.assign(total_to_be_sent, 0);

  Kokkos::vector<GlobalOrdinal, Kokkos::DefaultExecutionSpace> new_external(num_external);
  for (LocalOrdinal i = 0; i < num_external; ++i) {
    new_external[external_local_index[i] - local_nrow] = external_index[i];
  }

  std::vector<int> recv_offsets(combined_neighbors.size(), 0);
  std::vector<int> send_offsets(combined_neighbors.size(), 0);
  for (size_t i = 1; i < combined_neighbors.size(); ++i) {
    recv_offsets[i] = recv_offsets[i - 1] + recv_length[i - 1];
    send_offsets[i] = send_offsets[i - 1] + send_length[i - 1];
  }

  std::vector<CommContext::request_type> requests;
  requests.reserve(combined_neighbors.size() * 2);

  for (size_t i = 0; i < combined_neighbors.size(); ++i) {
    if (send_length[i] > 0) {
      auto recv_view = Kokkos::subview(A.elements_to_send.d_view,
                                       std::make_pair(send_offsets[i], send_offsets[i] + send_length[i]));
      requests.push_back(KokkosComm::recv(get_comm().comm, recv_view, neighbors[i]));
    }
  }

  for (size_t i = 0; i < combined_neighbors.size(); ++i) {
    if (recv_length[i] > 0) {
      auto send_view = Kokkos::subview(new_external.d_view,
                                       std::make_pair(recv_offsets[i], recv_offsets[i] + recv_length[i]));
      requests.push_back(KokkosComm::send(get_comm().comm, send_view, neighbors[i]));
    }
  }

  wait_all(requests);

  for (GlobalOrdinal i = 0; i < total_to_be_sent; ++i) {
    A.elements_to_send[i] -= start_row;
  }

  A.num_cols = local_nrow + num_external;
  A.has_local_indices = true;
#else
  A.num_cols = A.rows.size();
  A.has_local_indices = true;
#endif
}

} //namespace miniFE

#endif
