cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(alex_patch)

find_package(Torch REQUIRED)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(alex_patch src/main.cpp src/alex_patch.cpp src/alex_patch.hpp)
target_link_libraries(alex_patch "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET alex_patch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET alex_patch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:alex_patch>)
endif (MSVC)