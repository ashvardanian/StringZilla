# Guards on `stringzilla::types`: the target gains no define and no link beyond `stringzilla::abi`, every container header
# compiles alone, and a probe over all of them neither defines nor references a symbol the library defines. Run with
# `-P`, this file is the `stringzilla_types_exports` test itself. Beside them, every tier header compiles as the first
# include of a header-only unit.

if (CMAKE_SCRIPT_MODE_FILE)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${STRINGZILLA_BINARY_DIR}" --config "${STRINGZILLA_CONFIG}" --target
                stringzilla_types_exports stringzilla_static COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${STRINGZILLA_NM}" -g -j --defined-only "${STRINGZILLA_LIBRARY}"
        OUTPUT_FILE "${STRINGZILLA_BINARY_DIR}/stringzilla_types/library.symbols" COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND "${STRINGZILLA_NM}" -g -j "${STRINGZILLA_PROBE}"
        OUTPUT_FILE "${STRINGZILLA_BINARY_DIR}/stringzilla_types/probe.symbols" COMMAND_ERROR_IS_FATAL ANY
    )
    file(STRINGS "${STRINGZILLA_BINARY_DIR}/stringzilla_types/library.symbols" library_symbols_)
    file(STRINGS "${STRINGZILLA_BINARY_DIR}/stringzilla_types/probe.symbols" probe_symbols_)
    set(clashes_)
    foreach (symbol_ IN LISTS probe_symbols_)
        if (symbol_ IN_LIST library_symbols_)
            list(APPEND clashes_ "${symbol_}")
        endif ()
    endforeach ()
    if (clashes_)
        message(FATAL_ERROR "stringzilla::types headers define or reference library symbols: ${clashes_}")
    endif ()
    return()
endif ()

function (stringzilla_check_types_target)
    get_target_property(definitions_ stringzilla_types INTERFACE_COMPILE_DEFINITIONS)
    get_target_property(links_ stringzilla_types INTERFACE_LINK_LIBRARIES)
    if (definitions_ OR NOT links_ STREQUAL "stringzilla_abi")
        message(
            FATAL_ERROR "stringzilla::types defines nothing and links only stringzilla::abi, not `${definitions_}` and "
                        "`${links_}`"
        )
    endif ()
endfunction ()
cmake_language(DEFER CALL stringzilla_check_types_target)

if (STRINGZILLA_BUILD_TEST)
    file(
        GLOB stringzilla_types_headers_
        RELATIVE "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/include/stringzilla/*.hpp" "${PROJECT_SOURCE_DIR}/include/stringzilla/small_string.h"
    )
    set(stringzilla_types_units_)
    set(stringzilla_types_probe_ "")
    foreach (header_ IN LISTS stringzilla_types_headers_)
        file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/stringzilla_types/${header_}.cpp" CONTENT
             "#include \"${header_}\"\n"
        )
        list(APPEND stringzilla_types_units_ "${PROJECT_BINARY_DIR}/stringzilla_types/${header_}.cpp")
        string(APPEND stringzilla_types_probe_ "#include \"${header_}\"\n")
    endforeach ()
    file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/stringzilla_types/exports.cpp" CONTENT "${stringzilla_types_probe_}")

    # Built by the tests rather than by default, so a failure reads as a failed guard.
    add_library(stringzilla_types_standalone OBJECT EXCLUDE_FROM_ALL ${stringzilla_types_units_})
    target_link_libraries(stringzilla_types_standalone PRIVATE stringzilla_types stringzilla_warnings)
    target_compile_features(stringzilla_types_standalone PRIVATE cxx_std_20)
    add_test(NAME stringzilla_types_standalone COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --config
                                                       $<CONFIG> --target stringzilla_types_standalone
    )

    # Every inline function is emitted, so the references in its body reach the symbol table.
    add_library(stringzilla_types_exports STATIC EXCLUDE_FROM_ALL "${PROJECT_BINARY_DIR}/stringzilla_types/exports.cpp")
    target_link_libraries(stringzilla_types_exports PRIVATE stringzilla_types)
    target_compile_features(stringzilla_types_exports PRIVATE cxx_std_20)
    target_compile_options(
        stringzilla_types_exports PRIVATE "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-O0;-femit-all-decls>"
                                          "$<$<CXX_COMPILER_ID:GNU>:-O0;-fkeep-inline-functions>"
    )

    # One C99 and one C++20 unit per tier, tables aside, with every capability the toolchain compiles, as a header-only
    # consumer may include any tier before its hub.
    file(
        GLOB stringzilla_tiers_headers_
        RELATIVE "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/include/stringzilla/*/*.h"
    )
    set(stringzilla_tiers_units_)
    foreach (header_ IN LISTS stringzilla_tiers_headers_)
        if (header_ MATCHES "/tables\\.h$")
            continue()
        endif ()
        foreach (extension_ IN ITEMS c cpp)
            file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/stringzilla_tiers/${header_}.${extension_}" CONTENT
                 "#include \"${header_}\"\n"
            )
            list(APPEND stringzilla_tiers_units_ "${PROJECT_BINARY_DIR}/stringzilla_tiers/${header_}.${extension_}")
        endforeach ()
    endforeach ()
    add_library(stringzilla_tiers_first OBJECT EXCLUDE_FROM_ALL ${stringzilla_tiers_units_})
    target_link_libraries(
        stringzilla_tiers_first PRIVATE stringzilla_cpu_capabilities_compiled stringzilla_header stringzilla_warnings
    )
    set_target_properties(
        stringzilla_tiers_first
        PROPERTIES C_STANDARD 99
                   C_STANDARD_REQUIRED ON
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
    )
    add_test(NAME stringzilla_tiers_first COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --config $<CONFIG>
                                                  --target stringzilla_tiers_first
    )

    # Each builds in the one tree, which two concurrent builds would race over.
    set_tests_properties(
        stringzilla_types_standalone stringzilla_tiers_first PROPERTIES RESOURCE_LOCK stringzilla_build_tree
    )
    if (CMAKE_NM)
        add_test(
            NAME stringzilla_types_exports
            COMMAND
                "${CMAKE_COMMAND}" -D "STRINGZILLA_BINARY_DIR=${PROJECT_BINARY_DIR}" -D "STRINGZILLA_CONFIG=$<CONFIG>"
                -D "STRINGZILLA_NM=${CMAKE_NM}" -D "STRINGZILLA_LIBRARY=$<TARGET_FILE:stringzilla_static>" -D
                "STRINGZILLA_PROBE=$<TARGET_FILE:stringzilla_types_exports>" -P "${CMAKE_CURRENT_LIST_FILE}"
        )
        set_tests_properties(stringzilla_types_exports PROPERTIES RESOURCE_LOCK stringzilla_build_tree)
    endif ()
endif ()
