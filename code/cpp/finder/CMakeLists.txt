cmake_minimum_required(VERSION 2.8)
PROJECT( finder )
FIND_PACKAGE(OpenCV REQUIRED )
add_executable (finder finder.cpp finder.h )
target_link_libraries (finder  ${OpenCV_LIBS})

#include_directories ("${PROJECT_SOURCE_DIR}/HogFinder")
#add_subdirectory (HogFinder)
#set (EXTRA_LIBS ${EXTRA_LIBS} HogFinder)
#target_link_libraries (finder  ${EXTRA_LIBS})


