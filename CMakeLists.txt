cmake_minimum_required(VERSION 3.20)
project(gurobi_cpp_dsl 
    VERSION 1.0.0
    DESCRIPTION "Modern C++20 DSL for Gurobi optimization"
    LANGUAGES CXX
)

# ============================================================================
# Options
# ============================================================================
option(DSL_BUILD_TESTS "Build test suite" ON)
option(DSL_BUILD_EXAMPLES "Build example programs" OFF)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Find Gurobi
# ============================================================================
if(DEFINED ENV{GUROBI_HOME})
    set(GUROBI_HOME $ENV{GUROBI_HOME})
else()
    if(WIN32)
        file(GLOB GUROBI_DIRS "C:/gurobi*")
    else()
        file(GLOB GUROBI_DIRS "/opt/gurobi*" "$ENV{HOME}/gurobi*")
    endif()
    if(GUROBI_DIRS)
        list(GET GUROBI_DIRS -1 GUROBI_HOME)
    endif()
endif()

find_path(GUROBI_INCLUDE_DIR
    NAMES gurobi_c++.h
    HINTS 
        ${GUROBI_HOME}/include
        ${GUROBI_HOME}/linux64/include
        ${GUROBI_HOME}/win64/include
        ${GUROBI_HOME}/mac64/include
)

find_library(GUROBI_LIBRARY
    NAMES gurobi gurobi100 gurobi110 gurobi120 gurobi95
    HINTS 
        ${GUROBI_HOME}/lib
        ${GUROBI_HOME}/linux64/lib
        ${GUROBI_HOME}/win64/lib
        ${GUROBI_HOME}/mac64/lib
)

find_library(GUROBI_CXX_LIBRARY
    NAMES gurobi_c++ gurobi_c++md2019 gurobi_c++md2022
    HINTS 
        ${GUROBI_HOME}/lib
        ${GUROBI_HOME}/linux64/lib
        ${GUROBI_HOME}/win64/lib
        ${GUROBI_HOME}/mac64/lib
)

if(NOT GUROBI_INCLUDE_DIR OR NOT GUROBI_LIBRARY OR NOT GUROBI_CXX_LIBRARY)
    message(FATAL_ERROR 
        "Gurobi not found. Please set GUROBI_HOME environment variable.\n"
        "  GUROBI_HOME=${GUROBI_HOME}\n"
        "  GUROBI_INCLUDE_DIR=${GUROBI_INCLUDE_DIR}\n"
        "  GUROBI_LIBRARY=${GUROBI_LIBRARY}\n"
        "  GUROBI_CXX_LIBRARY=${GUROBI_CXX_LIBRARY}"
    )
endif()

message(STATUS "Found Gurobi: ${GUROBI_HOME}")

# ============================================================================
# DSL Library (Header-Only Interface)
# ============================================================================
add_library(gurobi_dsl INTERFACE)
add_library(gurobi::dsl ALIAS gurobi_dsl)

target_include_directories(gurobi_dsl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${GUROBI_INCLUDE_DIR}
)

target_link_libraries(gurobi_dsl INTERFACE
    ${GUROBI_CXX_LIBRARY}
    ${GUROBI_LIBRARY}
)

target_compile_features(gurobi_dsl INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(gurobi_dsl INTERFACE /W4 /permissive-)
else()
    target_compile_options(gurobi_dsl INTERFACE -Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Tests
# ============================================================================
if(DSL_BUILD_TESTS)
    enable_testing()
    
    add_library(catch2_main STATIC tests/catch_amalgamated.cpp)
    target_include_directories(catch2_main PUBLIC tests)
    
    file(GLOB TEST_SOURCES "tests/test_*.cpp")
    add_executable(dsl_tests ${TEST_SOURCES})
    target_link_libraries(dsl_tests PRIVATE gurobi_dsl catch2_main)
    
    add_test(NAME DSLTests COMMAND dsl_tests)
endif()

# ============================================================================
# Examples
# ============================================================================
if(DSL_BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE gurobi_dsl)
    endforeach()
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS gurobi_dsl
    EXPORT gurobi_dsl-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/gurobi_dsl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT gurobi_dsl-targets
    FILE gurobi_dsl-config.cmake
    NAMESPACE gurobi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gurobi_dsl
)
