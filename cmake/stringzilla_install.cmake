# Install rules, the CMake package files, and the Swift artifact.

# The static library as SwiftPM links it, beside the headers and their module map: an XCFramework on Apple platforms,
# and elsewhere an SE-0482 artifact bundle. Each build adds its target to the one in STRINGZILLA_SWIFT_DIRECTORY, so a
# release fills one artifact from a build per target.
if (STRINGZILLA_BUILD_SWIFT)
    set(STRINGZILLA_SWIFT_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        CACHE PATH "Where the Swift artifact lands"
    )
    if (APPLE)
        # The SDK names the XCFramework library an archive joins, and Mac Catalyst builds with the macOS one.
        get_filename_component(stringzilla_swift_sysroot_name_ "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${stringzilla_swift_sysroot_name_}" stringzilla_swift_sysroot_lowercase_)
        string(REGEX REPLACE "[0-9.]*\\.sdk$" "" stringzilla_swift_sdk_ "${stringzilla_swift_sysroot_lowercase_}")
        if (CMAKE_C_FLAGS MATCHES "-macabi")
            set(stringzilla_swift_sdk_ maccatalyst)
        elseif (NOT stringzilla_swift_sdk_)
            set(stringzilla_swift_sdk_ macosx)
        endif ()
        set(stringzilla_swift_target_ -D "STRINGZILLA_SWIFT_SDK=${stringzilla_swift_sdk_}" -D
                                      "STRINGZILLA_SWIFT_ARCHITECTURE=${CMAKE_SYSTEM_PROCESSOR}"
        )
    else ()
        # SwiftPM matches a variant by the exact triple it builds for, spelled as Swift spells it.
        if (stringzilla_arch_x8664_)
            set(stringzilla_swift_arch_ x86_64)
        elseif (stringzilla_arch_arm64_)
            set(stringzilla_swift_arch_ aarch64)
        else ()
            set(stringzilla_swift_arch_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(stringzilla_swift_triple_)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-linux-gnu")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-linux-android${ANDROID_NATIVE_API_LEVEL}")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-windows-msvc")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "WASI")
            set(stringzilla_swift_triple_ "wasm32-unknown-wasip1")
        endif ()
        set(STRINGZILLA_SWIFT_TRIPLES
            "${stringzilla_swift_triple_}"
            CACHE STRING "Swift target triples the artifact bundle serves"
        )
        if (NOT STRINGZILLA_SWIFT_TRIPLES)
            message(FATAL_ERROR "STRINGZILLA_BUILD_SWIFT needs STRINGZILLA_SWIFT_TRIPLES on ${CMAKE_SYSTEM_NAME}")
        endif ()
        set(stringzilla_swift_target_ -D "STRINGZILLA_SWIFT_TRIPLES=$<JOIN:${STRINGZILLA_SWIFT_TRIPLES},,>" -D
                                      "STRINGZILLA_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    add_custom_target(
        stringzilla_swift ALL
        COMMAND
            "${CMAKE_COMMAND}" -D "STRINGZILLA_SWIFT_ARCHIVE=$<TARGET_FILE:stringzilla_static>" -D
            "STRINGZILLA_SWIFT_HEADERS=${PROJECT_SOURCE_DIR}/include" -D
            "STRINGZILLA_SWIFT_DIRECTORY=${STRINGZILLA_SWIFT_DIRECTORY}" ${stringzilla_swift_target_} -P
            "${PROJECT_SOURCE_DIR}/cmake/stringzilla_swift_artifact.cmake"
        DEPENDS stringzilla_static
        VERBATIM
    )
endif ()

if (STRINGZILLA_INSTALL)
    include(CMakePackageConfigHelpers)

    # Every installed library belongs to an `EXPORT` set - a `.so` with no imported target leaves consumers
    # hand-rolling `find_library`.
    # Destinations come from the `GNUInstallDirs` defaults, so each call names only its target and export set.
    install(TARGETS stringzilla_header stringzilla_static EXPORT stringzillaTargets)
    if (TARGET stringzilla_shared)
        install(TARGETS stringzilla_shared EXPORT stringzillaTargets)
    endif ()

    install(
        EXPORT stringzillaTargets
        FILE stringzillaTargets.cmake
        NAMESPACE stringzilla::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stringzillaConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )

    # SwiftPM reads `module.modulemap` from the XCFramework or artifact bundle `stringzilla_swift` packs. At an installed
    # include root it is a hazard: Clang treats whichever one it finds there as authoritative for the whole prefix.
    install(
        DIRECTORY ${stringzilla_include_build_dir_}
        DESTINATION ${stringzilla_include_install_dir_}
        PATTERN "module.modulemap" EXCLUDE
    )
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/src)

    # A `.pc` for the consumers CMake never reaches - cgo among them. Spelling the prefix against `${pcfiledir}` keeps it
    # correct after `cmake --install --prefix` moves the tree. `GNUInstallDirs` returns a relative path unless a distro
    # forced an absolute one, and only the former may follow a prefix. No `Libs.private`: the shared library is C and
    # links nothing but libc, so a static caller inherits no closure to name.
    if (TARGET stringzilla_shared)
        file(RELATIVE_PATH stringzilla_pc_relative_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
             "${CMAKE_INSTALL_PREFIX}"
        )
        string(REGEX REPLACE "/+$" "" stringzilla_pc_trimmed_prefix_ "${stringzilla_pc_relative_prefix_}")
        set(stringzilla_pc_prefix_ "\${pcfiledir}/${stringzilla_pc_trimmed_prefix_}")

        if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
            set(stringzilla_pc_libdir_ "${CMAKE_INSTALL_LIBDIR}")
        else ()
            set(stringzilla_pc_libdir_ "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
        endif ()
        if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
            set(stringzilla_pc_includedir_ "${CMAKE_INSTALL_INCLUDEDIR}")
        else ()
            set(stringzilla_pc_includedir_ "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
        endif ()

        configure_file(cmake/stringzilla.pc.in "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" @ONLY)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    endif ()

endif ()
