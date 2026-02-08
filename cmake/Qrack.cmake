set(QRACK_DIR "/usr/local/lib/qrack")
set(QRACK_INCLUDE "/usr/local/include")

find_library(QRACK_LIB
    NAMES qrack libqrack
    PATHS "${QRACK_DIR}"
    NO_DEFAULT_PATH)

if (QRACK_LIB)
    message(STATUS "Found Qrack library: ${QRACK_LIB}")
    target_link_directories(weed PUBLIC ${QRACK_DIR})
    target_include_directories(weed PUBLIC ${QRACK_INCLUDE})
    target_compile_definitions (weed PUBLIC QRACK_AVAILABLE=1)
    target_link_libraries (weed PUBLIC ${QRACK_LIB})
    target_sources (weed PRIVATE
        src/modules/qrack_neuron.cpp
        src/modules/qrack_neuron_layer.cpp
        )
endif (QRACK_LIB)
