set(VECCAPPOW "7" CACHE STRING "Log2 of maximum qubit capacity of a single QInterface (must be at least 5, equivalent to >= 32 qubits)")

if (VECCAPPOW LESS 5)
    message(FATAL_ERROR "VECCAPPOW must be at least 5, equivalent to >= 32 qubits!")
endif (VECCAPPOW LESS 5)

if (VECCAPPOW LESS UINTPOW)
    message(FATAL_ERROR "VECCAPPOW must be greater than or equal to UINTPOW!")
endif (VECCAPPOW LESS UINTPOW)

if (VECCAPPOW GREATER 6)
    target_sources(weed PRIVATE
        src/common/big_integer.cpp
        )
endif (VECCAPPOW GREATER 6)
