# GBNF grammar parser + grammar-constrained sampling + JSON-schema -> GBNF (no third-party deps).
add_library(qv_grammar STATIC grammar.cpp json_schema.cpp tool_grammar.cpp)
target_include_directories(qv_grammar PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_compile_features(qv_grammar PUBLIC cxx_std_20)

add_executable(test_grammar ${CMAKE_SOURCE_DIR}/tests/test_grammar.cpp)
target_link_libraries(test_grammar PRIVATE qv_grammar qv_text)
target_compile_definitions(test_grammar PRIVATE QV_QWEN_TOKENIZER="/workspace/q-volta-models/Qwen3.8-27B/tokenizer.json")
add_test(NAME grammar COMMAND test_grammar)
