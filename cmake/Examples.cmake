add_executable (xor
    examples/xor.cpp
    )
set_target_properties(xor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (xor ${WEED_LIBS})

set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (xor PUBLIC ${EXAMPLE_COMPILE_OPTS})
