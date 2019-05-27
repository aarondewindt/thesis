cmake_minimum_required(VERSION 3.13)
project(f16)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonInterp 3.6 REQUIRED)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
        ${CMAKE_CURRENT_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

# Add external libraries
include_directories(extern_libs/eigen-git-mirror)
add_subdirectory(extern_libs/random) # path to the 'random' library root
link_libraries(effolkronium_random) # add include path to a compiler

find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

add_subdirectory(extern_libs/msgpack11)
link_libraries(msgpack11)

# Add spikes and the main library.
add_subdirectory(f16/cpp)
add_subdirectory(spike)
add_subdirectory(test)
