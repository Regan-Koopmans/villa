# --- VC dependencies ----------------------------------------------------------
include(FetchContent)

option(VC_BUILD_JSON "Build in-source JSON library" OFF)
option(VC_BUILD_Z5   "Build (vendor) z5 header-only library" ON)
option(VC_BUILD_FMT "Build in-source fmt library" OFF)
option(VC_BUILD_EIGEN "Build in-source Eigen library" OFF)
option(VC_BUILD_CERES "Build in-source ceres-solver library" OFF)
option(VC_BUILD_XTENSOR "Build in-source xtensor library" OFF)
option(VC_BUILD_SPDLOG "Build in-source spdlog library" OFF)

#find_package(CURL REQUIRED)
#find_package(OpenSSL REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(glog REQUIRED)

# Try a preinstalled z5 first, unless the user explicitly forces vendoring.
if (VC_BUILD_Z5)
    find_package(z5 CONFIG QUIET)
    if (z5_FOUND)
        message(STATUS "Using preinstalled z5 at: ${z5_DIR} (set VC_BUILD_Z5=OFF to force this; keep ON to try vendoring).")
        set(VC_BUILD_Z5 OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (NOT VC_BUILD_Z5)
    # Use a system / previously installed z5
    find_package(z5 CONFIG REQUIRED)
else()
    # Vendoring path: fetch z5 and add it as a subdir.
    # z5 defines options; set them in the cache *before* adding the subproject.
    set(BUILD_Z5PY OFF CACHE BOOL "Disable Python bits for z5" FORCE)
    set(WITH_BLOSC ON  CACHE BOOL "Enable Blosc in z5"        FORCE)

    # On CMake ≥4, compatibility with <3.5 was removed. Setting this floor
    # avoids errors if z5 asks for 3.1 in its CMakeLists.
    if (NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()

    # FetchContent: prefer MakeAvailable over deprecated Populate/add_subdirectory
    FetchContent_Declare(
        z5
        GIT_REPOSITORY https://github.com/constantinpape/z5.git
        GIT_TAG        ee2081bb974fe0d0d702538400c31c38b09f1629
    )
    FetchContent_MakeAvailable(z5)
endif()

# ---- Qt (apps / utils) -------------------------------------------------------
if ((VC_BUILD_APPS OR VC_BUILD_UTILS) AND VC_BUILD_GUI)
    find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    # Guard old qt cmake helper on distros with Qt < 6.3
    if(NOT DEFINED qt_generate_deploy_app_script)
        message(WARNING "WARNING qt_generate_deploy_app_script MISSING!")
        function(qt_generate_deploy_app_script)
        endfunction()
    endif()
endif()

# ---- CUDA sparse toggle ------------------------------------------------------
option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

# ---- Eigen -------------------------------------------------------------------
if (VC_BUILD_EIGEN)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.1
    )
    FetchContent_GetProperties(Eigen3)
    if(NOT Eigen3_POPULATED)
      FetchContent_Populate(Eigen3)
      add_subdirectory(${eigen3_SOURCE_DIR} ${eigen3_BINARY_DIR})
    endif()
else()
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()

if (CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING
        "Configuring a Debug build. Eigen performance will be degraded. "
        "Consider RelWithDebInfo for symbols, or Release for max performance.")
endif()

# ---- Ceres -------------------------------------------------------------------
if (VC_BUILD_CERES)
    FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver.git
        GIT_TAG 2.2.0
    )
    FetchContent_MakeAvailable(ceres)
else()
    find_package(Ceres REQUIRED)
endif()

# ---- OpenCV ------------------------------------------------------------------
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
endif()

# ---- OpenMP ------------------------------------------------------------------
if (VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    find_package(OpenMP REQUIRED)
    set(XTENSOR_USE_OPENMP 1)
else()
    message(STATUS "OpenMP support disabled")
    set(XTENSOR_USE_OPENMP 0)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
endif()

# ---- xtensor/xsimd toggle used by your code ---------------------------------

if (VC_BUILD_XTENSOR)
    set(XTENSOR_USE_XSIMD 1)
    FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG 0.8.1
    )
    FetchContent_MakeAvailable(xtl)

    FetchContent_Declare(
        xsimd
        GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
        GIT_TAG 13.2.0
    )
    FetchContent_MakeAvailable(xsimd)

    FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG 0.27.1
    )
    FetchContent_MakeAvailable(xtensor)
else()
    set(XTENSOR_USE_XSIMD 1)
    find_package(xtensor REQUIRED)
    get_target_property(XTENSOR_INCLUDES xtensor INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "xtensor include dirs: ${XTENSOR_INCLUDES}")
endif()

# ---- fmt ---------------------------------------------------------------------

if (VC_BUILD_FMT)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.1.0
    )
    FetchContent_MakeAvailable(fmt)
else()
    find_package(fmt REQUIRED)
endif()

# ---- spdlog ------------------------------------------------------------------

if (VC_BUILD_SPDLOG)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.4.2
    )
    FetchContent_MakeAvailable(spdlog)
    if(VC_INSTALL_LIBS)
        install(
                TARGETS spdlog
                COMPONENT "Libraries"
                EXPORT "${targets_export_name}"
                ARCHIVE DESTINATION "lib"
                LIBRARY DESTINATION "lib"
                INCLUDES DESTINATION "${include_install_dir}"
                RUNTIME DESTINATION "bin"
        )
    endif()
else()
    find_package(spdlog 1.4.2 CONFIG REQUIRED)
endif()

# ---- nlohmann/json -----------------------------------------------------------
if (VC_BUILD_JSON)
    FetchContent_Declare(
        json
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
    )
    FetchContent_GetProperties(json)
    if (NOT json_POPULATED)
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install   ON  CACHE INTERNAL "")
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    find_package(nlohmann_json 3.9.1 REQUIRED)
endif()

# ---- Boost (apps/utils only) -------------------------------------------------
if (VC_BUILD_APPS OR VC_BUILD_UTILS)
    find_package(Boost 1.58 REQUIRED COMPONENTS program_options)
endif()

# ---- PaStiX ------------------------------------------------------------------
if (VC_WITH_PASTIX)
  find_package(PaStiX REQUIRED)
  message(STATUS "PaStiX found: ${PASTIX_LIBRARY}")
  if (NOT TARGET vc3d_pastix)
    add_library(vc3d_pastix INTERFACE)
    target_link_libraries(vc3d_pastix INTERFACE PaStiX::PaStiX)
    target_compile_definitions(vc3d_pastix INTERFACE VC_HAVE_PASTIX=1)
  endif()
endif()
