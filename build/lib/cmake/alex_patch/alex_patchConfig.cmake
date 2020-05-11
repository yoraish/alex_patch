# Config file for the module example
# It defines the following variables:
# ${PROJECT_NAME}_INCLUDE_DIR  - Location of header files
# ${PROJECT_NAME}_INCLUDE_DIRS - All include directories needed to use ${PROJECT_NAME}
# ${PROJECT_NAME}_LIBRARY      - ${PROJECT_NAME} library
# ${PROJECT_NAME}_LIBRARIES    - ${PROJECT_NAME} library and all dependent libraries
# ${PROJECT_NAME}_DEFINITIONS  - Compiler definitions as semicolon separated list

find_library(alex_patch_LIBRARY alex_patch
  PATHS /usr/local/lib
  NO_DEFAULT_PATH
  )

set(alex_patch_LIBRARIES
  ${alex_patch_LIBRARY}
  ${OpenCV_LIBS})

find_path(alex_patch_INCLUDE_DIR alex_patch/alex_patch.hpp
  PATHS /usr/local/include
  NO_DEFAULT_PATH
  )

set(alex_patch_INCLUDE_DIRS
  ${alex_patch_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS})

set(alex_patch_DEFINITIONS "-std=c++11")
