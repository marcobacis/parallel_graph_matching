cmake_minimum_required (VERSION 2.8.11)
project (graph_matching)

#set(BOOST_ROOT /usr/local/boost-1.66.0)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Wall -ggdb -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -ggdb --std=c++0x -O3")

#Boost Library includes and flags
find_package(Boost 1.56.0 REQUIRED COMPONENTS system)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

#MPI Library include/flags
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})



#Executables
add_executable(test_nsd test_nsd.cpp nsd.cpp match.cpp)
target_link_libraries(test_nsd ${MPI_LIBRARIES})
target_link_libraries(test_nsd ${Boost_LIBRARIES})

#Sequential test (both similarity and auction, no MPI)
add_executable(test_sequential test_sequential.cpp nsd.cpp match.cpp)
target_link_libraries(test_sequential ${MPI_LIBRARIES})
target_link_libraries(test_sequential ${Boost_LIBRARIES})

#if(MPI_COMPILE_FLAGS)
#  set_target_properties(match PROPERTIES
#    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()

#if(MPI_LINK_FLAGS)
#  set_target_properties(match PROPERTIES
#    LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
