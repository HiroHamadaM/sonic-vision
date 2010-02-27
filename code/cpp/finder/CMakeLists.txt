cmake_minimum_required(VERSION 2.8)
PROJECT( finder )
FIND_PACKAGE(OpenCV REQUIRED )
add_executable (finder main.cpp hand.cpp skin.cpp tools.cpp finder.cpp tools.h skin.h hand.h finder.h settings.h)
target_link_libraries (finder  ${OpenCV_LIBS})

#include_directories ("${PROJECT_SOURCE_DIR}/HogFinder")
#add_subdirectory (HogFinder)
#set (EXTRA_LIBS ${EXTRA_LIBS} HogFinder)
#target_link_libraries (finder  ${EXTRA_LIBS})


