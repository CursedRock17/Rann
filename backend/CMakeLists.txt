cmake_minimum_version(VERSION 3.23)
project(Rann VERSION 1.0)

add_subdirectory(include)
add_subdirectory(src)

if(NOT CMAKE_CXX_VERSION)
    set(CMAKE_CXX_VERSION 20)
    set(CMAKE_CXX_VERISON_REQUIRED TRUE)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Werror -O3)

target_link_libraries(
${PROJECT_NAME} PUBLIC
)

target_include_directories(
${PROJECT_NAME} PUBLIC
"${PROJECT_BINARY_DIR}"
"${PROJECT_SOURCE_DIR}/include"
"${PROJECT_SOURCE_DIR}/src"
)
