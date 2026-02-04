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
option (ENABLE_INTRINSICS "Use x86_64 intrinsics" ON)
if (PACK_DEBIAN)
    set(ENABLE_EXAMPLES OFF)
endif (PACK_DEBIAN)

# Declare the library
add_library (weed STATIC
    src/common/functions.cpp
    src/common/parallel_for.cpp
    src/modules/dropout.cpp
    src/modules/embedding.cpp
    src/modules/linear.cpp
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
    src/tensors/tensor.cpp
    )
set_property(TARGET weed PROPERTY POSITION_INDEPENDENT_CODE ON)

set(WEED_LIBS weed)
if (ENABLE_PTHREAD)
    if (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
        set(WEED_LIBS ${WEED_LIBS} atomic)
    endif (ANDROID OR CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv" OR (NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
    if (NOT ANDROID AND NOT MSVC)
	set(WEED_LIBS ${WEED_LIBS} pthread)
    endif (NOT ANDROID AND NOT MSVC)
endif (ENABLE_PTHREAD)

if (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)
    add_executable (unittest
        test/test_main.cpp
        test/tests.cpp
        )

    add_executable (benchmarks
        test/test_main.cpp
        test/benchmarks.cpp
        )

    target_link_libraries (unittest ${WEED_LIBS})
    target_link_libraries (benchmarks ${WEED_LIBS})
endif (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)

# Included after the library and other modules have been declared
option (ENABLE_OPENCL "Use OpenCL optimizations" ON)
option (ENABLE_SNUCL "Use SnuCL framework for clusters" OFF)
include ("cmake/Boost.cmake")
include ("cmake/Complex_x2.cmake")
include ("cmake/CppStd.cmake")
include ("cmake/EnvVars.cmake")
include ("cmake/FpMath.cmake")
include ("cmake/Pstridepow.cmake")
include ("cmake/TCapPow.cmake")
if (ENABLE_EXAMPLES)
    include ("cmake/Examples.cmake")
endif (ENABLE_EXAMPLES)
include ("cmake/OpenCL.cmake" )
include ("cmake/CUDA.cmake")

message ("C++ standard year is (at least): ${CPP_STD}")
message ("Tensor capacity power is: ${TCAPPOW}")
message ("Floating-point capacity power is: ${FPPOW}")
message ("CPU parallelism is: ${ENABLE_PTHREAD}")
message ("CPU work item stride power is: ${PSTRIDEPOW}")
message ("Async CPU is: ${ENABLE_ASYNC}")
message ("OpenCL (v2.0) out-of-order queue is: ${ENABLE_OOO_OCL}")
message ("Environment variable usage is: ${ENABLE_ENV_VARS}")

if (ENABLE_OPENCL AND ENABLE_CUDA)
  message(FATAL_ERROR "You cannot enable both OpenCL and CUDA! Please choose one (or turn both off)." )
endif (ENABLE_OPENCL AND ENABLE_CUDA)

if (FPPOW GREATER 6)
    target_link_libraries(weed PUBLIC quadmath)
endif (FPPOW GREATER 6)

if (ENABLE_COMPLEX_X2)
    set(COMPLEX_X2_MACRO "(!(defined(__GNUC__) || defined(__MINGW32__)) || ((FPPOW == 5) && __SSE__) || ((FPPOW == 6) && __SSE2__))")
else (ENABLE_COMPLEX_X2)
    set(COMPLEX_X2_MACRO "0")
endif (ENABLE_COMPLEX_X2)

if (ENABLE_SSE3)
    set(SSE3_MACRO "(!(defined(__GNUC__) || defined(__MINGW32__)) || __SSE3__)")
else (ENABLE_SSE3)
    set(SSE3_MACRO "0")
endif (ENABLE_SSE3)

if (MSVC)
    if (CPP_STD GREATER_EQUAL 23)
        set(WEED_CPP_STD_OPT /std:c++23)
    elseif (CPP_STD GREATER_EQUAL 20)
        set(WEED_CPP_STD_OPT /std:c++20)
    elseif (CPP_STD GREATER_EQUAL 17)
        set(WEED_CPP_STD_OPT /std:c++17)
    elseif (CPP_STD GREATER_EQUAL 14)
        set(WEED_CPP_STD_OPT /std:c++14)
    else ()
        set(WEED_CPP_STD_OPT "")
    endif ()
else (MSVC)
    if (CPP_STD GREATER_EQUAL 23)
        set(WEED_CPP_STD_OPT -std=c++23)
    elseif (CPP_STD GREATER_EQUAL 20)
        set(WEED_CPP_STD_OPT -std=c++20)
    elseif (CPP_STD GREATER_EQUAL 17)
        set(WEED_CPP_STD_OPT -std=c++17)
    elseif (CPP_STD GREATER_EQUAL 14)
        set(WEED_CPP_STD_OPT -std=c++14)
    else ()
        set(WEED_CPP_STD_OPT -std=c++11)
    endif ()
endif (MSVC)

if (TCAPPOW GREATER 6)
    set(WEED_CUDA_COMPILE_OPTS -O3 -use_fast_math -Xcompiler -fpermissive ${WEED_CPP_STD_OPT} --ptxas-options -O3,)
else (TCAPPOW GREATER 6)
    set(WEED_CUDA_COMPILE_OPTS -O3 -use_fast_math -Werror all-warnings ${WEED_CPP_STD_OPT} --ptxas-options -O3,)
endif (TCAPPOW GREATER 6)

if (ENABLE_GVIRTUS)
    set(WEED_CUDA_COMPILE_OPTS --cudart=shared ${WEED_CUDA_COMPILE_OPTS})
endif (ENABLE_GVIRTUS)

if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
 
        set(WEED_COMPILE_FLAGS_PART /O2 ${WEED_CPP_STD_OPT} /Wall)
    else (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(WEED_COMPILE_FLAGS_PART ${WEED_CPP_STD_OPT} /Wall)
    endif (CMAKE_BUILD_TYPE STREQUAL "Release")
else (MSVC)
    set(WEED_COMPILE_FLAGS_PART -O3 ${WEED_CPP_STD_OPT})
    if (EMSCRIPTEN AND ENABLE_PTHREAD)
        set (WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -pthread)
    endif (EMSCRIPTEN AND ENABLE_PTHREAD)
    set(WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -Wall)
    if (EMSCRIPTEN OR NOT APPLE OR NOT ENABLE_OPENCL)
        set(WEED_COMPILE_FLAGS_PART ${WEED_COMPILE_FLAGS_PART} -Werror)
    endif (EMSCRIPTEN OR NOT APPLE OR NOT ENABLE_OPENCL)
endif(MSVC)

set(WEED_COMPILE_OPTS ${WEED_COMPILE_FLAGS_PART})
set(TEST_COMPILE_OPTS ${WEED_COMPILE_FLAGS_PART})

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "i.86|amd64|x86_64"))
    set(ENABLE_INTRINSICS OFF)
    set(ENABLE_COMPLEX_X2 OFF)
endif (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "i.86|amd64|x86_64"))

if (ENABLE_INTRINSICS)
    include(CheckCXXCompilerFlag)
    if (ENABLE_SSE3)
        check_cxx_compiler_flag(-msse3 COMPILER_SUPPORTS_SSE3_FLAG)
        if (COMPILER_SUPPORTS_SSE3_FLAG)
            set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -msse3)
        else (COMPILER_SUPPORTS_SSE3_FLAG)
            set(ENABLE_SSE3 OFF)
        endif (COMPILER_SUPPORTS_SSE3_FLAG)
    endif (ENABLE_SSE3)

    if (ENABLE_AVX AND FPPOW GREATER 5)
        check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX_FLAG)
        if (COMPILER_SUPPORTS_AVX_FLAG)
            set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -mavx)
        else (COMPILER_SUPPORTS_AVX_FLAG)
            set(ENABLE_AVX OFF)
        endif (COMPILER_SUPPORTS_AVX_FLAG)
    endif (ENABLE_AVX AND FPPOW GREATER 5)

    if (ENABLE_FMA)
        check_cxx_compiler_flag(-mfma COMPILER_SUPPORTS_FMA_FLAG)
        if (COMPILER_SUPPORTS_FMA_FLAG)
            set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -mfma)
        else (COMPILER_SUPPORTS_FMA_FLAG)
            set(ENABLE_FMA OFF)
        endif (COMPILER_SUPPORTS_FMA_FLAG)
    endif (ENABLE_FMA)
endif (ENABLE_INTRINSICS)

message ("Complex (x2) SIMD support is: ${ENABLE_COMPLEX_X2}")
message ("SSE3.0 support is: ${ENABLE_SSE3}")
message ("FMA support is: ${ENABLE_FMA}")

if (FPPOW LESS 5)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -mfp16-format=ieee)
        set(TEST_COMPILE_OPTS ${TEST_COMPILE_OPTS} -mfp16-format=ieee)
    endif ()
endif (FPPOW LESS 5)

if (EMSCRIPTEN AND ENABLE_COMPLEX_X2)
    check_cxx_compiler_flag(-msimd128 COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -msimd128)
    endif (COMPILER_SUPPORTS_FLAG)
    check_cxx_compiler_flag(-msse COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -msse)
    endif (COMPILER_SUPPORTS_FLAG)
    check_cxx_compiler_flag(-msse2 COMPILER_SUPPORTS_FLAG)
    if (COMPILER_SUPPORTS_FLAG)
        set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -msse2)
    endif (COMPILER_SUPPORTS_FLAG)
endif (EMSCRIPTEN AND ENABLE_COMPLEX_X2)

if (ENABLE_EMIT_LLVM)
    set(WEED_COMPILE_OPTS ${WEED_COMPILE_OPTS} -emit-llvm)
endif (ENABLE_EMIT_LLVM)

include_directories(include/common)

if (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)
    target_include_directories (unittest PUBLIC test)
endif (NOT ENABLE_EMIT_LLVM AND NOT PACK_DEBIAN)

if (APPLE)
    set(TEST_COMPILE_OPTS -Wno-inconsistent-missing-override)
endif (APPLE)

if (ENABLE_CUDA)
    target_compile_options (weed PUBLIC
        "$<$<COMPILE_LANGUAGE:C>:${WEED_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${WEED_COMPILE_OPTS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${WEED_CUDA_COMPILE_OPTS}>"
        -DCATCH_CONFIG_FAST_COMPILE
        )
else (ENABLE_CUDA)
    if (MSVC)
        target_compile_options (weed PUBLIC ${WEED_COMPILE_OPTS})
    else (MSVC)
        target_compile_options (weed PUBLIC ${WEED_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
    endif (MSVC)
endif (ENABLE_CUDA)
if (NOT PACK_DEBIAN)
    target_compile_options (unittest PUBLIC ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
endif (NOT PACK_DEBIAN)

# Declare the OCL precompilation executable
add_executable (weed_cl_precompile
    src/weed_cl_precompile.cpp
    )
target_link_libraries (weed_cl_precompile ${WEED_LIBS})
target_compile_options (weed_cl_precompile PUBLIC ${TEST_COMPILE_OPTS})
install(TARGETS weed_cl_precompile RUNTIME DESTINATION bin BUNDLE DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

set_target_properties (weed PROPERTIES
    VERSION ${PROJECT_VERSION}
    )

# Install common headers
install (FILES
    ${CMAKE_BINARY_DIR}/include/common/config.h
    include/common/weed_functions.hpp
    include/common/weed_types.hpp
    include/common/complex16x2simd.hpp
    include/common/complex8x2simd.hpp
    include/common/oclapi.hpp
    include/common/oclengine.hpp
    include/common/parallel_for.hpp
    include/common/half.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/weed/common
    )

# Install weed library headers
install (FILES
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
    include/modules/dropout.hpp
    include/modules/embedding.hpp
    include/modules/linear.hpp
    include/modules/module.hpp
    include/modules/relu.hpp
    include/modules/sequential.hpp
    include/modules/sigmoid.hpp
    include/modules/tanh.hpp
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
        set(OPENCL_V3 ON)
        set(CMAKE_INSTALL_LIBDIR "/usr/lib")
    else (DEBHELPER_COMPAT_VERSION STREQUAL "13")
        set(OPENCL_V3 OFF)
    endif (DEBHELPER_COMPAT_VERSION STREQUAL "13")

    if (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "libboost-dev, ")
    else (FPPOW GREATER 6)
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
    endif (FPPOW GREATER 6)

    if (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "${CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS}opencl-headers, ocl-icd-opencl-dev, xxd")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "ocl-icd-opencl-dev")
    else (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))
        set(CPACK_DEBIAN_PACKAGE_BUILD_DEPENDS "")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
    endif (ENABLE_OPENCL AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^ppc"))

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
