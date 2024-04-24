# CMake Options for miniFE/kokkos

You will probably need to provide `-DKokkos_ROOT=path/to/kokkos-install`.

Other options are:

* `-DMINIFE_ENABLE_MPI`:
  * `=ON`: (default) require and enable MPI
  * `=OFF`: disable MPI support
* `-DMINIFE_SCALAR=double`: (default) Define `MINIFE_SCALAR=double` in source files
* `-DMINIFE_SCALAR=float`: Define `MINIFE_SCALAR=float` in source files