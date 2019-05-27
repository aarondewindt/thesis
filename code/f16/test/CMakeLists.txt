
add_executable(cpp_tests
        test_main.cpp
        test_factorial.cpp
        test_grid_tile_coding.cpp
        test_environment.cpp
        match_pointer_array.h
        )

## Add msgpack11 to the
target_include_directories(cpp_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../extern_libs/msgpack11)

target_link_libraries(cpp_tests libf16)
target_include_directories(cpp_tests PUBLIC ${CMAKE_SOURCE_DIR}/f16/cpp)
