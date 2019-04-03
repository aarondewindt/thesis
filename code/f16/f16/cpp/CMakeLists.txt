
add_library(libf16
        rand.h
        grid_tile_coding.h
        agent.h
        enviroment.cpp enviroment.h)

target_compile_options(libf16 PUBLIC -march=native)

set_source_files_properties(cpp_init.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(cpp_init cpp_init.pyx)
target_link_libraries(cpp_init libf16)
target_include_directories(cpp_init PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(cpp_init PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
