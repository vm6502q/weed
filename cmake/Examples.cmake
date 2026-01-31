add_executable (nor
    examples/nor.cpp
    )
set_target_properties(nor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (nor ${WEED_LIBS})

set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (nor PUBLIC ${EXAMPLE_COMPILE_OPTS})
