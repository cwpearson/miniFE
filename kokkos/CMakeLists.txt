cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # same as Kokkos 4.3

option(MINIFE_ENABLE_MPI "Enable MPI support" ON)

project(miniFE LANGUAGES CXX)

if(NOT DEFINED MINIFE_SCALAR)
    set(MINIFE_SCALAR "double") # or float
endif()
if(NOT DEFINED MINIFE_LOCAL_ORDINAL)
    set(MINIFE_LOCAL_ORDINAL "int") # only int seems to work
endif()
if(NOT DEFINED MINIFE_GLOBAL_ORDINAL)
    set(MINIFE_GLOBAL_ORDINAL "int") # only int seems to work
endif()

find_package(Kokkos REQUIRED)
if(MINIFE_ENABLE_MPI)
  find_package(MPI REQUIRED)
else()
  find_package(MPI) # okay if we can't find MPI if the user didn't ask for it
endif()

# set MINIFE_SRCS and MINIFE_INCL_DIRS
add_subdirectory(src)
add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(kokkos/linalg/src)
add_subdirectory(utils)

add_executable(miniFE.kokkos ${MINIFE_SRCS})
target_link_libraries(miniFE.kokkos Kokkos::kokkos)
foreach(INCL_DIR ${MINIFE_INCL_DIRS})
  target_include_directories(miniFE.kokkos PRIVATE ${INCL_DIR})
endforeach()

target_compile_definitions(miniFE.kokkos PRIVATE MINIFE_SCALAR=${MINIFE_SCALAR})
target_compile_definitions(miniFE.kokkos PRIVATE MINIFE_LOCAL_ORDINAL=${MINIFE_LOCAL_ORDINAL})
target_compile_definitions(miniFE.kokkos PRIVATE MINIFE_GLOBAL_ORDINAL=${MINIFE_GLOBAL_ORDINAL})
message(STATUS "MINIFE_SCALAR=${MINIFE_SCALAR}")
message(STATUS "MINIFE_LOCAL_ORDINAL=${MINIFE_LOCAL_ORDINAL}")
message(STATUS "MINIFE_GLOBAL_ORDINAL=${MINIFE_GLOBAL_ORDINAL}")

if (MPI_FOUND AND MINIFE_ENABLE_MPI)
  message(STATUS "MPI_VERSION =                 ${MPI_VERSION}")
  message(STATUS "MPI_CXX_COMPILER =            ${MPI_CXX_COMPILER}")
  message(STATUS "MPI_CXX_COMPILE_OPTIONS =     ${MPI_CXX_COMPILE_OPTIONS}")
  message(STATUS "MPI_CXX_COMPILE_DEFINITIONS = ${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "MPI_CXX_INCLUDE_DIRS =        ${MPI_CXX_INCLUDE_DIRS}")
  message(STATUS "MPI_CXX_LINK_FLAGS =          ${MPI_CXX_LINK_FLAGS}")
  message(STATUS "MPI_CXX_LIBRARIES =           ${MPI_CXX_LIBRARIES}")
  target_compile_definitions(miniFE.kokkos PRIVATE HAVE_MPI)
  target_link_libraries(miniFE.kokkos MPI::MPI_CXX)
endif()