
cmake_minimum_required(VERSION 3.8)

project( scanArray )

#export OpenCV_DIR=/usr/local/opt/opencv3
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/opt/opencv3/share/OpenCV)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} )

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW4_files main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(HW4 ${HW4_files} ${hdr} ${img} ${cu})


