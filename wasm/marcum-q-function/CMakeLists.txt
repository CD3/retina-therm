cmake_minimum_required(VERSION 3.20)

project(SpecialFunctions)

find_package(Boost REQUIRED)
add_executable( marcum_q marcum_q.cpp)

# this command would build the module
# emcc ./marcum_q.cpp -o marcum_q.wasm --no-entry -I/path/to/boost/include
# so we need cmake to use `.wasm` for the target suffix and add the `--no-entry` option.
set_target_properties( marcum_q PROPERTIES SUFFIX ".wasm")
target_link_libraries(marcum_q Boost::headers "--no-entry")
