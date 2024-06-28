
add_subdirectory(src)

add_executable(LinalgMain
    src/main.c)

target_link_libraries(LinalgMain
    EmbeddedLapack::EmbeddedLapack)
