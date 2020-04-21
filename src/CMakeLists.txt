# Build main library.
set(ALEX_PATCH_SRCS
  alex_patch/alex_patch.cpp 
  alex_patch/alex_patch.hpp)

add_library(${PROJECT_NAME} SHARED ${ALEX_PATCH_SRCS})

# Link the library
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

# Set properties
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build an executable then link with the library
add_executable(alex_patch_eval main.cpp )
target_link_libraries(alex_patch_eval "${PROJECT_NAME}")
set_target_properties(alex_patch_eval PROPERTIES
    LINKER_LANGUAGE CXX COMPILE_FLAGS "-std=c++14 -Wall")

# Install these
install(TARGETS ${PROJECT_NAME} alex_patch_eval
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY ${PROJECT_NAME} DESTINATION include
  FILES_MATCHING PATTERN "*.h")