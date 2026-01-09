CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
project(Satellites)

add_executable(parallel parallel.c)


# Here is an example syntax how to add compiler options to your build process
# See the project work document on compiler flag syntax on Linux and Windows


target_compile_options(parallel PRIVATE /fp:fast )
target_compile_options(parallel PRIVATE /arch:AVX2 )
target_compile_options(parallel PRIVATE /Qvec-report:2)



# requires OpenCL_headers and OpenCL.lib statically linked library

set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/OpenCL-Headers")
set(OpenCL_LIBRARY "${CMAKE_SOURCE_DIR}/OpenCL.lib")
find_package(OpenCL REQUIRED)
target_include_directories(parallel PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(parallel ${OpenCL_LIBRARIES})

add_custom_command(
    TARGET parallel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/parallel.cl"
    $<TARGET_FILE_DIR:parallel>
    VERBATIM)


# Find and link SDL2
if (WIN32)
    set(SDL2_DIR "${CMAKE_SOURCE_DIR}/SDL2-2.32.10/cmake")
endif()
find_package(SDL2 REQUIRED)
target_include_directories(parallel PRIVATE ${SDL2_INCLUDE_DIR})
target_link_libraries(parallel ${SDL2_LIBRARIES})

if (WIN32)
# copy the sdl2 .dll file to the same folder as the executable
add_custom_command(
    TARGET parallel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:SDL2::SDL2>
    $<TARGET_FILE_DIR:parallel>
    VERBATIM)
else()
    # Math library shouldn't be linked on Windows, but must be linked on Linux
    target_link_libraries(parallel m)
endif()
