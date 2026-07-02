cmake_minimum_required(VERSION 3.20)
# LLVM 的 FindLibEdit 会 try_compile 一个 .c 文件，需要同时启用 C
project(chapter05_examples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_plugin_libs
    core support demangle remarks passes)
llvm_map_components_to_libnames(llvm_tool_libs
    core support demangle remarks passes analysis irreader)

function(add_chapter05_pass_plugin target source)
    add_library(${target} MODULE ${source})
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        COMPILE_FLAGS "${LLVM_COMPILE_FLAGS}"
        LINK_FLAGS "${LLVM_LINK_FLAGS}"
    )
    target_link_libraries(${target} PRIVATE ${llvm_plugin_libs})
endfunction()

add_chapter05_pass_plugin(CountInstructionsPass CountInstructionsPass.cpp)
add_chapter05_pass_plugin(CountFunctionsPass CountFunctionsPass.cpp)
add_chapter05_pass_plugin(MyPass MyPass.cpp)
add_chapter05_pass_plugin(InsertPrintfPass InsertPrintfPass.cpp)
add_chapter05_pass_plugin(CountAddPass CountAddPass.cpp)
add_chapter05_pass_plugin(AddToSubPass AddToSubPass.cpp)

add_executable(run_pipeline RunPipelineTool.cpp)
target_link_libraries(run_pipeline PRIVATE ${llvm_tool_libs})
