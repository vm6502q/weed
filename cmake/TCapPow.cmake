set(TCAPPOW "5" CACHE STRING "Log-2 of maximum qubit capacity of a single Tensor (must be at least 5, equivalent to >= 32 qubits)")

if (TCAPPOW LESS 3)
    message(FATAL_ERROR "TCAPPOW must be at least 3, equivalent to >= 8 qubits!")
endif (TCAPPOW LESS 3)

if (TCAPPOW GREATER 7)
    message(FATAL_ERROR "TCAPPOW must be less than 8, equivalent to <= 128 qubits!")
endif (TCAPPOW GREATER 7)
