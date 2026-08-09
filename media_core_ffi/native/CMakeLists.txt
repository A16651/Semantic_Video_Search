cmake_minimum_required(VERSION 3.14)
project(media_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(media_core SHARED media_core.cpp)

if(MSVC)
    target_compile_options(media_core PRIVATE /EHsc /W3)
endif()
