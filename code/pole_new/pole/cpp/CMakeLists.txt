
add_executable(pole
        main.cpp
        rand.h
        types.h
        agent_base.h
        pid_agent.h
        table_agent.h
        environment.h)

target_compile_options(pole PUBLIC -march=native)


set_source_files_properties(__init__.pyx PROPERTIES CYTHON_IS_CXX TRUE)
cython_add_module(__init__ __init__.pyx)
target_compile_options(__init__ PUBLIC -march=native)
target_include_directories(__init__ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(__init__ PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        )
