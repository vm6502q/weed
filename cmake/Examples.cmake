set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)

add_executable (xor
    examples/xor.cpp
    )
set_target_properties(xor PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (xor PUBLIC weed)
target_compile_options (xor PUBLIC ${EXAMPLE_COMPILE_OPTS})

add_executable (heart_attack
    examples/heart_attack.cpp
    )
set_target_properties(heart_attack PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (heart_attack PUBLIC weed)
target_compile_options (heart_attack PUBLIC ${EXAMPLE_COMPILE_OPTS})

configure_file(examples/data/Heart_Attack_Data_Set.csv examples/data/Heart_Attack_Data_Set.csv COPYONLY)
configure_file(examples/data/LICENSE.txt examples/data/LICENSE.txt COPYONLY)

if (QRACK_LIB)
    add_executable (xor_qrack
        examples/xor_qrack.cpp
    )
    set_target_properties(xor_qrack PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
    )
    target_link_libraries (xor_qrack
        PRIVATE weed
        PRIVATE ${QRACK_LIB}
    )
    target_compile_options (xor_qrack PUBLIC ${EXAMPLE_COMPILE_OPTS})
endif (QRACK_LIB)
