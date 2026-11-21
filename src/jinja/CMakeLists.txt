# Standalone Jinja2-compatible template engine + chat helpers (no third-party deps).
add_library(qv_jinja STATIC value.cpp parse.cpp builtins.cpp eval.cpp chat.cpp)
target_include_directories(qv_jinja PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_compile_features(qv_jinja PUBLIC cxx_std_20)

add_executable(jinja_render ${CMAKE_SOURCE_DIR}/tools/jinja_render.cpp)
target_link_libraries(jinja_render PRIVATE qv_jinja)

add_executable(test_jinja ${CMAKE_SOURCE_DIR}/tests/test_jinja.cpp)
target_link_libraries(test_jinja PRIVATE qv_jinja qv_text)
target_compile_definitions(test_jinja PRIVATE QV_JINJA_TEST_DIR="${CMAKE_SOURCE_DIR}/tests/jinja")
add_test(NAME jinja COMMAND test_jinja)
