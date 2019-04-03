
add_executable(cpp_tests
        test_main.cpp
        test_factorial.cpp
        test_grid_tile_coding.cpp
        match_pointer_array.h
        )

target_link_libraries(libf16)
target_include_directories(cpp_tests PUBLIC ${CMAKE_SOURCE_DIR}/f16/cpp)
