# The C++ SDK over the C interface: `cpp/stringzilla/*.hpp` include the C headers `stringzilla::abi` carries, so a
# consumer linking `stringzilla::cpp` gets the static library its wrappers dispatch through.
add_library(stringzilla_cpp INTERFACE)
add_library(stringzilla::cpp ALIAS stringzilla_cpp)
set_target_properties(stringzilla_cpp PROPERTIES EXPORT_NAME cpp)
target_include_directories(
    stringzilla_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(stringzilla_cpp INTERFACE stringzilla::abi stringzilla::static)
target_compile_features(stringzilla_cpp INTERFACE cxx_std_20)

# The smoke test of the SDK, against the compiled kernels a `stringzilla::cpp` consumer links, never the header-only
# engine.
if (STRINGZILLA_BUILD_TEST)
    add_executable(stringzilla_test_cpp test.cpp)
    target_link_libraries(stringzilla_test_cpp PRIVATE stringzilla::cpp stringzilla_warnings)
    add_test(NAME stringzilla_test_cpp COMMAND stringzilla_test_cpp)
endif ()
