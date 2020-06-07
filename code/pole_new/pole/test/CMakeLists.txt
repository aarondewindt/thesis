add_executable(pole_test
        match_span.h
        test_main.cpp
        test_table_3d.cpp)

target_link_libraries(pole_test Catch2::Catch2)
target_link_libraries(pole_test pole)
