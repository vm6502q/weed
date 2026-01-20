add_executable (quantum_associative_memory
    examples/quantum_associative_memory.cpp
    )
set_target_properties(quantum_associative_memory PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (quantum_associative_memory ${QRACK_LIBS})

set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (quantum_associative_memory PUBLIC ${EXAMPLE_COMPILE_OPTS})
