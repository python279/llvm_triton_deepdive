cmake_minimum_required(VERSION 3.20)
project(chapter08_build_add LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(build_add build_add.cpp)
target_link_libraries(build_add PRIVATE
    MLIRIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSupport
    LLVMSupport)
