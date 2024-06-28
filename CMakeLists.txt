cmake_minimum_required(VERSION 3.21)

# Project #
# ####### #

# Set the project name and language
project(
    EmbeddedLapack
    VERSION 0.0.1
    DESCRIPTION "Embedded Lapack C library using f2c"
    HOMEPAGE_URL "https://github.com/FancySoftBot/EmbeddedLapack"
    LANGUAGES C)

# Build Type #
# ########## #

# Set the possible values of build type
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "Instrument")

# Configure Options #
# ################# #

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    option(ENABLE_LINK_TIME_OPTIMIZATION "Enables link time optimization, if available" ON)
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Instrument")
    option(ENABLE_ASAN "Enable Address, Undefined Behaviour, and Memory Sanitizers" ON)
    option(ENABLE_TSAN "Enable Thread Sanitizer" OFF)
    if (INSTRUMENT_ASAN AND INSTRUMENT_TSAN)
        error("Only INSTRUMENT_ASAN or INSTRUMENT_TSAN can be set on ON, not both")
    endif ()
endif ()

# C/C++ standard #
# ############## #

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (NOT DEFINED CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Check Link-Time Optimization #
# ############################ #

if (ENABLE_LINK_TIME_OPTIMIZATION)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT
        LTO_SUPPORTED
        OUTPUT
        error)
    if (LTO_SUPPORTED)
        message(STATUS "IPO / LTO enabled")
    else ()
        message(ERROR "IPO / LTO not supported: <${error}>")
    endif ()
endif ()

# EmbeddedLapack library #
##########################

add_subdirectory(EmbeddedLapack)
