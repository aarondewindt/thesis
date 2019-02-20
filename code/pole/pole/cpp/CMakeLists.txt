
add_library(libpole
        pole.cpp pole.h
        season.cpp season.h
        rand.h
        agent.h
        pid_agent.cpp pid_agent.h
        poly_rl_agent.cpp poly_rl_agent.h
        table_rl_agent.cpp table_rl_agent.h
        grid_tile_coding.h
        tile_coding_agent.cpp tile_coding_agent.h
        )

target_compile_options(libpole PUBLIC -march=native)

set_source_files_properties(cpp_init.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(cpp_init cpp_init.pyx)
target_link_libraries(cpp_init libpole)
target_include_directories(cpp_init PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(cpp_init PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
