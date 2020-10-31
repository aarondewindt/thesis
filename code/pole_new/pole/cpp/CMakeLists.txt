
add_library(pole INTERFACE)
target_include_directories(pole INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pole_main
        grid_tile_coding.h
        main.cpp
        table_3d.h
        rand.h
        types.h
        agent_base.h
        pid_agent.h
        table_agent.h
        environment.h
        spancpy.h
        grid_tile_coding_agent.h)

set_source_files_properties(__init__.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(__init__ __init__.pyx)
target_compile_options(__init__ PUBLIC -march=native)
target_link_libraries(__init__ pole)
#target_include_directories(__init__ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(__init__ PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
