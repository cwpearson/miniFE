#ifndef _exchange_externals_hpp_
#define _exchange_externals_hpp_

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

#include <cstdlib>
#include <iostream>
#include <vector>

#include <Kokkos_Subview.hpp>

#include <comm.hpp>
#include <outstream.hpp>
#include <Minife_Kokkos_Vector.hpp>

namespace miniFE {

template<typename MatrixType,
         typename VectorType>
struct
exchange_externals_functor {
	typedef device_device_type device_type;
	Kokkos::vector<typename VectorType::ScalarType,Kokkos::DefaultExecutionSpace> _x;
    Kokkos::vector<typename MatrixType::GlobalOrdinalType,Kokkos::DefaultExecutionSpace> _elements_to_send;
	Kokkos::vector<typename MatrixType::ScalarType,Kokkos::DefaultExecutionSpace>        _send_buffer;

	exchange_externals_functor(const MatrixType& A,
			const VectorType& x):_x(x.coefs),_elements_to_send(A.elements_to_send),_send_buffer(A.send_buffer) {

	}
	KOKKOS_INLINE_FUNCTION
	void operator() (int i) const {
		_send_buffer.d_view(i) = _x.d_view(_elements_to_send.d_view(i));
	}
};

template<typename MatrixType,
         typename VectorType>
void
exchange_externals(MatrixType& A,
                   VectorType& x)
{
#ifdef HAVE_KOKKOS_COMM
#ifdef MINIFE_DEBUG
  std::ostream& os = outstream();
  os << "entering exchange_externals\n";
#endif

  if (num_ranks() < 2) return;

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;

  const int local_nrow = A.rows.size();
  const int num_neighbors = A.neighbors.size();
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& recv_length = A.recv_length;
  const Kokkos::vector<LocalOrdinal,Kokkos::DefaultExecutionSpace>& send_length = A.send_length;
  const Kokkos::vector<int,Kokkos::DefaultExecutionSpace>& neighbors = A.neighbors;

  std::vector<CommContext::request_type> requests;
  requests.reserve(num_neighbors * 2);

  int recv_offset = local_nrow;
  for (int i = 0; i < num_neighbors; ++i) {
    const int n_recv = recv_length[i];
    if (n_recv == 0) {
      continue;
    }

    auto recv_view = Kokkos::subview(x.coefs.d_view, std::make_pair(recv_offset, recv_offset + n_recv));
    requests.push_back(KokkosComm::recv(get_comm().comm, recv_view, neighbors[i]));
    recv_offset += n_recv;
  }

#ifdef MINIFE_DEBUG
  os << "launched recvs\n";
#endif

  const size_t total_to_be_sent = A.elements_to_send.size();
  exchange_externals_functor<MatrixType, VectorType> f(A, x);
  Kokkos::parallel_for("exchange_externals", total_to_be_sent, f);

  int send_offset = 0;
  for (int i = 0; i < num_neighbors; ++i) {
    const int n_send = send_length[i];
    if (n_send == 0) {
      continue;
    }

    auto send_view = Kokkos::subview(A.send_buffer.d_view, std::make_pair(send_offset, send_offset + n_send));
    requests.push_back(KokkosComm::send(get_comm().comm, send_view, neighbors[i]));
    send_offset += n_send;
  }

#ifdef MINIFE_DEBUG
  os << "send to " << num_neighbors << std::endl;
#endif

  wait_all(requests);
  x.coefs.template modify<typename decltype(x.coefs)::t_dev::device_type>();

#ifdef MINIFE_DEBUG
  os << "leaving exchange_externals" << std::endl;
#endif
#else
  (void)A;
  (void)x;
#endif
}

} //namespace miniFE

#endif
