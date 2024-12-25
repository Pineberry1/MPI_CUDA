# Set the project name
project(MPIProject)

# Find MPI package
find_package(MPI REQUIRED)

add_executable(laba lab-a.c)
target_include_directories(laba PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(laba ${MPI_LIBRARIES})

add_executable(labb lab-b.c)
target_include_directories(labb PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(labb ${MPI_LIBRARIES})

add_executable(labc1 lab-c-1.c)
target_include_directories(labc1 PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(labc1 ${MPI_LIBRARIES})

add_executable(labc2 lab-c-2.c)
target_include_directories(labc2 PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(labc2 ${MPI_LIBRARIES})

add_executable(labd lab-d.cpp)
target_include_directories(labd PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(labd ${MPI_LIBRARIES})
