CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

SET(WITH_CUDA ON CACHE BOOL "Build CUDA packages")

IF (WITH_CUDA)
  IF(APPLE)
    # work around for mac os x bug:
    # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
    if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
    endif()
  ENDIF()
  FIND_PACKAGE(CUDA 4.0)
ENDIF (WITH_CUDA)

IF (CUDA_FOUND)

  SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
  SET(CUDA_PROPAGATE_HOST_FLAGS ON)
  SET(CUDA_HOST_COMPILATION_CPP ON)

  LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")

  # bug on Apple
  IF(APPLE)
    LINK_DIRECTORIES("/usr/local/cuda/lib/")
  ENDIF(APPLE)

  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
  INCLUDE_DIRECTORIES(${Torch_SOURCE_INCLUDES})
  INCLUDE_DIRECTORIES("lib/THC")

  # libraries
  ADD_SUBDIRECTORY(lib)

  # packages
  ADD_SUBDIRECTORY(pkg)
  
ELSE (CUDA_FOUND)

  MESSAGE(STATUS "Disabling CUDA (CUDA 4.0 required, and not found)")
  MESSAGE(STATUS "If CUDA 4.0 is installed, then specify CUDA_TOOLKIT_ROOT_DIR")
  
ENDIF (CUDA_FOUND)



IF (WIN32 AND NOT CYGWIN)
  SET(INSTALL_CUDA_LIBRARIES "OFF"
    CACHE BOOL "Copy the required CUDA DLLs into the Torch install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(Install_Required_Library ln)
    get_filename_component(libpath ${ln} PATH)
    get_filename_component(libname ${ln} NAME_WE)
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      file(GLOB libdlls "${libpath}/../../bin/${libname}64*.dll")
	else()
      file(GLOB libdlls "${libpath}/../../bin/${libname}32*.dll")
	endif()
    install(PROGRAMS ${libdlls} DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")
ENDMACRO(Install_Required_Library libname)

IF (CUDA_FOUND AND INSTALL_CUDA_LIBRARIES)
  IF (CUDA_CUDART_LIBRARY)
    Install_Required_Library(${CUDA_CUDART_LIBRARY})
  ENDIF()
  IF (CUDA_cublas_LIBRARY)
    Install_Required_Library(${CUDA_cublas_LIBRARY})
  ENDIF()
  IF (CUDA_curand_LIBRARY)
    Install_Required_Library(${CUDA_curand_LIBRARY})
  ENDIF()
  IF (CUDA_cusparse_LIBRARY)
    Install_Required_Library(${CUDA_cusparse_LIBRARY})
  ENDIF()
ENDIF()
