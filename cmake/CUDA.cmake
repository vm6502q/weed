option (ENABLE_CUDA "Build CUDA-based QEngine type" OFF)
option (ENABLE_GVIRTUS "Link against GVirtuS for distributed CUDA" OFF)

if (ENABLE_OPENCL)
    set(ENABLE_CUDA OFF)
endif ()

if (ENABLE_CUDA)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package(CUDA)
        if (NOT CUDA_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        find_package (CUDAToolkit)
        if (NOT CUDAToolkit_FOUND)
            set(ENABLE_CUDA OFF)
        endif ()
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
endif()

message ("CUDA Support is: ${ENABLE_CUDA}")

if (ENABLE_CUDA)
    enable_language(CUDA)
    target_compile_definitions(weed PUBLIC ENABLE_CUDA=1)
    if ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (weed PUBLIC ${PROJECT_BINARY_DIR} ${CUDA_INCLUDE_DIRS})
        target_compile_options (weed PUBLIC ${CUDA_COMPILATION_OPTIONS})
        set(WEED_CUDA_LIBRARIES ${CUDA_LIBRARIES})
    else ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))
        target_include_directories (weed PUBLIC ${PROJECT_BINARY_DIR} ${CUDAToolkit_INCLUDE_DIRS})
        target_compile_options (weed PUBLIC ${CUDAToolkit_COMPILATION_OPTIONS})
        set(WEED_CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
    endif ((CMAKE_MAJOR_VERSION LESS 4) AND (CMAKE_MINOR_VERSION LESS 27))

    if (NOT DEFINED WEED_CUDA_ARCHITECTURES)
        # See https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures#answer-68223399
        if (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            include(FindCUDA/select_compute_arch)
            CUDA_DETECT_INSTALLED_GPUS(WEED_CUDA_ARCHITECTURES)
            string(STRIP "${WEED_CUDA_ARCHITECTURES}" WEED_CUDA_ARCHITECTURES)
            string(REPLACE " " ";" WEED_CUDA_ARCHITECTURES "${WEED_CUDA_ARCHITECTURES}")
            string(REPLACE "." "" WEED_CUDA_ARCHITECTURES "${WEED_CUDA_ARCHITECTURES}")
        else (${CMAKE_VERSION} VERSION_LESS "3.24.0")
            set(WEED_CUDA_ARCHITECTURES native)
        endif (${CMAKE_VERSION} VERSION_LESS "3.24.0")
    endif (NOT DEFINED WEED_CUDA_ARCHITECTURES)

    message("WEED_CUDA_ARCHITECTURES: ${WEED_CUDA_ARCHITECTURES}")

    target_link_libraries (weed ${WEED_CUDA_LIBRARIES})
    set_target_properties(weed PROPERTIES CUDA_ARCHITECTURES "${WEED_CUDA_ARCHITECTURES}")
    
    # Add the CUDA objects to the library
    target_sources (weed PRIVATE
        src/common/cudaengine.cu
        src/common/qengine.cu
        src/devices/gpu_device.cpp
        src/storage/gpu_complex_storage.cpp
        src/storage/gpu_int_storage.cpp
        src/storage/gpu_real_storage.cpp
        )

endif(ENABLE_CUDA)
