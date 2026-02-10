cmake_minimum_required (VERSION 3.10)
project (Weed VERSION 0.1.0 DESCRIPTION "Minimalist AI/ML inference and backprogation in the style of Qrack" LANGUAGES CXX)

# Installation commands
include (GNUInstallDirs)

include ("cmake/Coverage.cmake")
include ("cmake/Format.cmake")
# PSTRIDEPOW relates to parallel for loops:
include ("cmake/Pthread.cmake")

include_directories ("include" "include/common" "${CMAKE_BINARY_DIR}/include/common")

set(CMAKE_WARN_DEPRECATED FALSE)

option (ENABLE_EXAMPLES "Include usage examples" ON)
option (ENABLE_TESTS "Include tests and benchmarks" ON)
if (PACK_DEBIAN)
    set(ENABLE_EXAMPLES OFF)
endif (PACK_DEBIAN)

# Declare the library
add_library (weed STATIC
    src/common/functions.cpp
    src/common/parallel_for.cpp
    src/modules/dropout.cpp
    src/modules/embedding.cpp
    src/modules/gru.cpp
    src/modules/layernorm.cpp
    src/modules/linear.cpp
    src/modules/lstm.cpp
    src/modules/migrate_cpu.cpp
    src/modules/migrate_gpu.cpp
    src/modules/module.cpp
    src/modules/multihead_attention.cpp
    src/modules/positional_encoding.cpp
    src/modules/sequential.cpp
    src/modules/transformer_encoder_layer.cpp
    src/ops/abs.cpp
    src/ops/clamp.cpp
    src/ops/commuting.cpp
    src/ops/copy_broadcast.cpp
    src/ops/div.cpp
    src/ops/embedding.cpp
    src/ops/in_place.cpp
    src/ops/matmul.cpp
    src/ops/reduce.cpp
    src/ops/real_extremum.cpp
    src/ops/real_unary.cpp
    src/ops/sub.cpp
    src/ops/sum.cpp
    src/ops/pow.cpp
    src/ops/util.cpp
    src/storage/cpu_complex_storage.cpp
    src/storage/cpu_int_storage.cpp
    src/storage/cpu_real_storage.cpp
    src/storage/sparse_cpu_complex_storage.cpp
    src/storage/sparse_cpu_real_storage.cpp
    src/storage/storage.cpp
    src/tensors/parameter.cpp
    src/tensors/tensor.cpp
    src/tensors/symbol_tensor.cpp
    )
set_property(TARGET weed PROPERTY POSITION_INDEPENDENT_CODE ON)

set(WEED_LIBS weed)
if (WEED_ENABLE_PTHREAD)
    if (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
        set(WEED_LIBS ${WEED_LIBS} atomic)
    endif (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
    if (NOT ANDROID AND NOT MSVC)
	set(WEED_LIBS ${WEED_LIBS} pthread)
    endif (NOT ANDROID AND NOT MSVC)
endif (WEED_ENABLE_PTHREAD)

if (ENABLE_TESTS AND NOT PACK_DEBIAN)
    add_executable (unittest
        test/test_main.cpp
        test/tests.cpp
        )

    add_executable (benchmarks
        test/test_main.cpp
        test/benchmarks.cpp
        )

    target_link_libraries (unittest weed)
    target_link_libraries (benchmarks weed)
endif (ENABLE_TESTS AND NOT PACK_DEBIAN)

# Included after the library and other modules have been declared
option (WEED_ENABLE_OPENCL "Use OpenCL optimizations" ON)
option (WEED_ENABLE_SNUCL "Use SnuCL framework for clusters" OFF)
include ("cmake/CppStd.cmake")
include ("cmake/EnvVars.cmake")
include ("cmake/FpMath.cmake")
include ("cmake/Pstridepow.cmake")
include ("cmake/TCapPow.cmake")
include ("cmake/Qrack.cmake")
if (ENABLE_EXAMPLES)
    include ("cmake/Examples.cmake")
endif (ENABLE_EXAMPLES)
include ("cmake/OpenCL.cmake" )

message ("C++ standard year is (at least): ${WEED_CPP_STD}")
message ("Tensor capacity power is: ${WEED_TCAPPOW}")
message ("Floating-point capacity power is: ${WEED_FPPOW}")
message ("CPU parallelism is: ${WEED_ENABLE_PTHREAD}")
message ("CPU work item stride power is: ${WEED_PSTRIDEPOW}")
message ("OpenCL (v2.0) out-of-order queue is: ${WEED_ENABLE_OOO_OCL}")
message ("Environment variable usage is: ${WEED_ENABLE_ENV_VARS}")

if (WEED_FPPOW GREATER 6)
    target_link_libraries(weed PUBLIC quadmath)
endif (WEED_FPPOW GREATER 6)

if (MSVC)
    if (WEED_CPP_STD GREATER_EQUAL 23)
        set(CPP_STD_OPT /std:c++23)
    elseif (WEED_CPP_STD GREATER_EQUAL 20)
        set(CPP_STD_OPT /std:c++20)
    elseif (WEED_CPP_STD GREATER_EQUAL 17)
        set(CPP_STD_OPT /std:c++17)
    elseif (WEED_CPP_STD GREATER_EQUAL 14)
        set(CPP_STD_OPT /std:c++14)
    else ()
        set(CPP_STD_OPT "")
    endif ()
else (MSVC)
    if (WEED_CPP_STD GREATER_EQUAL 23)
        set(CPP_STD_OPT -std=c++23)
    elseif (WEED_CPP_STD GREATER_EQUAL 20)
        set(CPP_STD_OPT -std=c++20)
    elseif (WEED_CPP_STD GREATER_EQUAL 17)
        set(CPP_STD_OPT -std=c++17)
    elseif (WEED_CPP_STD GREATER_EQUAL 14)
        set(CPP_STD_OPT -std=c++14)
    else ()
        set(CPP_STD_OPT -std=c++11)
    endif ()
endif (MSVC)

if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
 
        set(WEED_COMPILE_FLAGS_PART /O2 ${CPP_STD_OPT} /Wall)
    else (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(WEED_COMPILE_FLAGS_PART ${CPP_STD_OPT} /Wall)
    endif (CMAKE_BUILD_TYPE STREQUAL "Release")
else (MSVC)
    set(WEED_COMPILE_FLAGS_PART -O3 ${CPP_STD_OPT})
    if (EMSCRIPTEN AND WEED_ENABLE_PTHREAD)
        set (WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -pthread)
    endif (EMSCRIPTEN AND WEED_ENABLE_PTHREAD)
    set(WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -Wall)
    if (EMSCRIPTEN OR NOT APPLE OR NOT WEED_ENABLE_OPENCL)
        set(WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -Werror)
    endif (EMSCRIPTEN OR NOT APPLE OR NOT WEED_ENABLE_OPENCL)
endif(MSVC)

set(WEED_COMPILE_OPTS ${WEED_COMPILE_FLAGS_PART})
set(TEST_COMPILE_OPTS ${WEED_COMPILE_FLAGS_PART})

if (FPPOW LESS 5)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -mfp16-format=ieee)
        set(TEST_COMPILE_OPTS ${TEST_COMPILE_OPTS} -mfp16-format=ieee)
    endif ()
endif (FPPOW LESS 5)

include_directories(include/common)

if (ENABLE_TESTS AND NOT PACK_DEBIAN)
    target_include_directories (unittest PUBLIC test)
    target_include_directories (benchmarks PUBLIC test)
endif (ENABLE_TESTS AND NOT PACK_DEBIAN)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

if (MSVC)
target_compile_options (weed PUBLIC ${WEED_COMPILE_OPTS})
else (MSVC)
target_compile_options (weed PUBLIC ${WEED_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
endif (MSVC)

if (ENABLE_TESTS AND NOT PACK_DEBIAN)
    target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
endif (ENABLE_TESTS AND NOT PACK_DEBIAN)

add_library (weed_shared SHARED src/shared_api.cpp)
set_property(TARGET weed_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries (weed_shared weed)
target_compile_options (weed_shared PUBLIC ${WEED_COMPILE_OPTS})
set_target_properties (weed_shared PROPERTIES VERSION ${PROJECT_VERSION})

# Declare the OCL precompilation executable
add_executable (weed_cl_precompile
    src/weed_cl_precompile.cpp
    )
target_link_libraries (weed_cl_precompile weed)
target_compile_options (weed_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
install(TARGETS weed_cl_precompile RUNTIME DESTINATION bin BUNDLE DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set_target_properties (weed PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    ${CMAKE_BINARY_DIR}/include/common/config.h
    include/common/half.hpp
    include/common/oclapi.hpp
    include/common/oclengine.hpp
    include/common/parallel_for.hpp
    include/common/rapidcsv.h
    include/common/serializer.hpp
    include/common/weed_functions.hpp
    include/common/weed_types.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/weed/common
    )

# Install weed library headers
install (FILES
    include/shared_api.hpp
    include/autograd/adam.hpp
    include/autograd/bci_loss.hpp
    include/autograd/mse_loss.hpp
    include/autograd/node.hpp
    include/autograd/sgd.hpp
    include/autograd/zero_grad.hpp
    include/devices/gpu_device.hpp
    include/devices/pool_item.hpp
    include/devices/queue_item.hpp
    include/enums/device_tag.hpp
    include/enums/dtype.hpp
    include/enums/module_type.hpp
    include/enums/storage_type.hpp
    include/enums/quantum_function_type.hpp
    include/modules/dropout.hpp
    include/modules/embedding.hpp
    include/modules/gru.hpp
    include/modules/linear.hpp
    include/modules/logsoftmax.hpp
    include/modules/lstm.hpp
    include/modules/max.hpp
    include/modules/mean.hpp
    include/modules/mean_center.hpp
    include/modules/migrate_cpu.hpp
    include/modules/migrate_gpu.hpp
    include/modules/min.hpp
    include/modules/module.hpp
    include/modules/multihead_attention.hpp
    include/modules/positional_encoding.hpp
    include/modules/relu.hpp
    include/modules/qrack_neuron.hpp
    include/modules/qrack_neuron_layer.hpp
    include/modules/sequential.hpp
    include/modules/sigmoid.hpp
    include/modules/softmax.hpp
    include/modules/tanh.hpp
    include/modules/transformer_encoder_layer.hpp
    include/ops/abs.hpp
    include/ops/clamp.hpp
    include/ops/commuting.hpp
    include/ops/copy_broadcast.hpp
    include/ops/div.hpp
    include/ops/embedding.hpp
    include/ops/in_place.hpp
    include/ops/matmul.hpp
    include/ops/pow.hpp
    include/ops/reduce.hpp
    include/ops/real_extremum.hpp
    include/ops/real_unary.hpp
    include/ops/sub.hpp
    include/ops/sum.hpp
    include/ops/util.hpp
    include/storage/all_storage.hpp
    include/storage/cpu_complex_storage.hpp
    include/storage/cpu_real_storage.hpp
    include/storage/cpu_storage.hpp
    include/storage/gpu_complex_storage.hpp
    include/storage/gpu_real_storage.hpp
    include/storage/gpu_storage.hpp
    include/storage/sparse_cpu_complex_storage.hpp
    include/storage/sparse_cpu_real_storage.hpp
    include/storage/sparse_cpu_storage.hpp
    include/storage/storage.hpp
    include/storage/typed_storage.hpp
    include/tensors/base_tensor.hpp
    include/tensors/complex_scalar.hpp
    include/tensors/complex_tensor.hpp
    include/tensors/parameter.hpp
    include/tensors/real_scalar.hpp
    include/tensors/real_tensor.hpp
    include/tensors/scalar.hpp
    include/tensors/symbol_tensor.hpp
    include/tensors/tensor.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/weed
    )

# Install the archive
install (TARGETS weed
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/weed
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
install (TARGETS weed_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/weed
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE
        )

# Install the pkg-config file
configure_file (libweed.pc.in libweed.pc @ONLY)
install (FILES ${CMAKE_BINARY_DIR}/libweed.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
install (FILES ${CMAKE_SOURCE_DIR}/debian/copyright DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/weed)
if (NOT MSVC)
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/changelog.gz"
        COMMAND gzip -cn9 "${CMAKE_SOURCE_DIR}/debian/changelog" > "${CMAKE_BINARY_DIR}/changelog.gz"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS "${CMAKE_SOURCE_DIR}/debian/changelog"
        COMMENT "Compressing changelog"
    )
    add_custom_target(changelog ALL DEPENDS "${CMAKE_BINARY_DIR}/changelog.gz")
    install(FILES "${CMAKE_BINARY_DIR}/changelog.gz"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/weed"
    )
    add_custom_command(
        OUTPUT "${CMAKE_BINARY_DIR}/weed_cl_precompile.1.gz"
        COMMAND gzip -cn9 "${CMAKE_SOURCE_DIR}/debian/docs/weed_cl_precompile.1" > "${CMAKE_BINARY_DIR}/weed_cl_precompile.1.gz"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS "${CMAKE_SOURCE_DIR}/debian/docs/weed_cl_precompile.1"
        COMMENT "Compressing weed_cl_precompile.1"
    )
    add_custom_target(weed_cl_precompile.1 ALL DEPENDS "${CMAKE_BINARY_DIR}/weed_cl_precompile.1.gz")
    install(FILES "${CMAKE_BINARY_DIR}/weed_cl_precompile.1.gz"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1"
    )
endif (NOT MSVC)

include(InstallRequiredSystemLibraries)
set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    )
set(CPACK_PACKAGE_NAME "libweed")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.md")
file(COPY LICENSE.md DESTINATION ${CMAKE_BINARY_DIR})
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
file(COPY README.md DESTINATION ${CMAKE_BINARY_DIR})
set(CPACK_PACKAGE_VERSION_MAJOR "${Weed_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${Weed_VERSION_MINOR}")
set(CPACK_PACKAGE_VENDOR "vm6502q")
set(CPACK_PACKAGE_CONTACT "stranoj@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "vm6502q/weed minimalist AI/ML inference and backprogation library (in the style of Qrack)")
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "vm6502q/weed minimalist AI/ML inference and backprogation library (in the style of Qrack)")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_THREADS 0)

if (PACK_DEBIAN)
    file(COPY cmake DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY CMakeLists.txt DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY libweed.pc.in DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY include DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY src DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY test DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY examples DESTINATION ${CMAKE_BINARY_DIR})

    set(CPACK_SOURCE_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Daniel Strano <stranoj@gmail.com>")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/vm6502q/weed")

    file(READ debian/changelog DEBIAN_CHANGELOG)
    string(REPLACE ") " ";" VERSION_LIST ${DEBIAN_CHANGELOG})
    list(GET VERSION_LIST 1 DEBIAN_CHANGELOG)
    string(REPLACE " " ";" VERSION_LIST ${DEBIAN_CHANGELOG})
    list(GET VERSION_LIST 0 DEBIAN_DISTRO_SERIES)
    message("Debian distroseries: ${DEBIAN_DISTRO_SERIES}")

    if (DEBHELPER_COMPAT_VERSION)
        message("debhelper-compat version override: ${DEBHELPER_COMPAT_VERSION}")
    else (DEBHELPER_COMPAT_VERSION)
        set(DEBHELPER_COMPAT_VERSION "13")
        if (DEBIAN_DISTRO_SERIES STREQUAL "bionic")
            set(DEBHELPER_COMPAT_VERSION "11")
        endif (DEBIAN_DISTRO_SERIES STREQUAL "bionic")
        if (DEBIAN_DISTRO_SERIES STREQUAL "focal")
            set(DEBHELPER_COMPAT_VERSION "12")
        endif (DEBIAN_DISTRO_SERIES STREQUAL "focal")
    endif (DEBHELPER_COMPAT_VERSION)
    if (DEBHELPER_COMPAT_VERSION STREQUAL "13")
        set(WEED_OPENCL_V3 ON)
        set(CMAKE_INSTALL_LIBDIR "/usr/lib")
    else (DEBHELPER_COMPAT_VERSION STREQUAL "13")
        set(WEED_OPENCL_V3 OFF)
    endif (DEBHELPER_COMPAT_VERSION STREQUAL "13")

    if (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "libboost-dev, ")
    else (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
    endif (FPPOW GREATER 6)

    if (WEED_ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "${CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS}opencl-headers, ocl-icd-opencl-dev, xxd")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "ocl-icd-opencl-dev")
    else (WEED_ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    endif (WEED_ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))

    set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
    set(CPACK_DEB_COMPONENT_INSTALL YES)
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_BINARY_DIR}/debian/rules;${CMAKE_BINARY_DIR}/debian/triggers")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
    set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)


    if (ENABLE_MAKE_CLEAN_OVERRIDE)
        configure_file (makefile.in makefile @ONLY)
    endif (ENABLE_MAKE_CLEAN_OVERRIDE)
    configure_file (debian/control.in debian/control @ONLY)
    configure_file (debian/libweed.dirs.in debian/${CPACK_PACKAGE_NAME}.dirs @ONLY)
    configure_file (debian/libweed.install.in debian/${CPACK_PACKAGE_NAME}.install @ONLY)
    configure_file (debian/libweed-dev.dirs.in debian/${CPACK_PACKAGE_NAME}-dev.dirs @ONLY)
    configure_file (debian/libweed-dev.install.in debian/${CPACK_PACKAGE_NAME}-dev.install @ONLY)
    configure_file (debian/libweed-docs.docs.in debian/${CPACK_PACKAGE_NAME}-docs.docs @ONLY)
    file(COPY debian DESTINATION ${CMAKE_BINARY_DIR})

    file(COPY makefile.in DESTINATION ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION})
    if (NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}")
        file(GLOB ARCHIVE_FILES ${CMAKE_BINARY_DIR}/*)
        file(COPY ${ARCHIVE_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION})
        execute_process(
            COMMAND make clean-cmake
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}
            )
        execute_process(
            COMMAND tar -C ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION} -cJf ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}_${CMAKE_PROJECT_VERSION}.orig.tar.xz .
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}
            )
    endif (NOT "${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/${CPACK_PACKAGE_NAME}-${CMAKE_PROJECT_VERSION}")
else (PACK_DEBIAN)
    set(CPACK_SOURCE_GENERATOR "TGZ")
endif (PACK_DEBIAN)
include(CPack)

configure_file(include/common/config.h.in include/common/config.h @ONLY)
