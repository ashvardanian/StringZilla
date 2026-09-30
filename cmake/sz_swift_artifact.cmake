# Adds one build's `stringzilla_static` to the Swift artifact in STRINGZILLA_SWIFT_DIRECTORY, keeping what earlier builds
# added, so a release fills one artifact from one CMake build per target. Runs as `cmake -P`, from `stringzilla_swift`.
#
#   STRINGZILLA_SWIFT_ARCHIVE       the static library
#   STRINGZILLA_SWIFT_HEADERS       the public headers, with the module map
#   STRINGZILLA_SWIFT_DIRECTORY     where `StringZillaC.xcframework` or `StringZillaC.artifactbundle` lands
#   STRINGZILLA_SWIFT_SDK           on Apple platforms, the SDK the archive targets, like `iphonesimulator`: the
#                                   XCFramework gets one library per SDK, fattened with `lipo`
#   STRINGZILLA_SWIFT_ARCHITECTURE  on Apple platforms, the archive's one architecture
#   STRINGZILLA_SWIFT_TRIPLES       elsewhere, the comma-separated Swift triples the archive serves
#   STRINGZILLA_SWIFT_VERSION       elsewhere, the library's version
get_filename_component(archive_name "${STRINGZILLA_SWIFT_ARCHIVE}" NAME)

if (STRINGZILLA_SWIFT_SDK)
    set(slices "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.slices")
    file(MAKE_DIRECTORY "${slices}/${STRINGZILLA_SWIFT_SDK}/${STRINGZILLA_SWIFT_ARCHITECTURE}")
    file(COPY_FILE "${STRINGZILLA_SWIFT_ARCHIVE}"
         "${slices}/${STRINGZILLA_SWIFT_SDK}/${STRINGZILLA_SWIFT_ARCHITECTURE}/${archive_name}"
    )

    set(libraries)
    file(
        GLOB sdks
        LIST_DIRECTORIES true
        "${slices}/*"
    )
    foreach (sdk IN LISTS sdks)
        file(GLOB architectures "${sdk}/*/${archive_name}")
        execute_process(
            COMMAND lipo -create ${architectures} -output "${sdk}/${archive_name}" COMMAND_ERROR_IS_FATAL ANY
        )
        list(APPEND libraries -library "${sdk}/${archive_name}" -headers "${STRINGZILLA_SWIFT_HEADERS}")
    endforeach ()
    file(REMOVE_RECURSE "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.xcframework")
    execute_process(
        COMMAND xcodebuild -create-xcframework ${libraries} -output
                "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.xcframework" COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET
    )
    return()
endif ()

# One directory per variant, named by its first triple, beside one `include/` every variant shares.
set(bundle "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.artifactbundle")
string(REPLACE "," ";" triples "${STRINGZILLA_SWIFT_TRIPLES}")
list(GET triples 0 variant)
file(MAKE_DIRECTORY "${bundle}/${variant}")
file(COPY_FILE "${STRINGZILLA_SWIFT_ARCHIVE}" "${bundle}/${variant}/${archive_name}")
file(REMOVE_RECURSE "${bundle}/include")
file(COPY "${STRINGZILLA_SWIFT_HEADERS}/" DESTINATION "${bundle}/include")

set(entry [[{"path": "", "supportedTriples": [], "staticLibraryMetadata": {"headerPaths": ["include"],
    "moduleMapPath": "include/module.modulemap"}}]]
)
string(JSON entry SET "${entry}" path "\"${variant}/${archive_name}\"")
foreach (triple IN LISTS triples)
    string(JSON count LENGTH "${entry}" supportedTriples)
    string(
        JSON
        entry
        SET
        "${entry}"
        supportedTriples
        ${count}
        "\"${triple}\""
    )
endforeach ()

if (EXISTS "${bundle}/info.json")
    file(READ "${bundle}/info.json" info)
else ()
    set(info [[{"schemaVersion": "1.0", "artifacts": {"StringZillaC": {"type": "staticLibrary", "variants": []}}}]])
endif ()
string(
    JSON
    info
    SET
    "${info}"
    artifacts
    StringZillaC
    version
    "\"${STRINGZILLA_SWIFT_VERSION}\""
)
# A rebuild of the same variant replaces it.
string(JSON count LENGTH "${info}" artifacts StringZillaC variants)
while (count GREATER 0)
    math(EXPR count "${count} - 1")
    string(
        JSON
        path
        GET
        "${info}"
        artifacts
        StringZillaC
        variants
        ${count}
        path
    )
    if (path STREQUAL "${variant}/${archive_name}")
        string(
            JSON
            info
            REMOVE
            "${info}"
            artifacts
            StringZillaC
            variants
            ${count}
        )
    endif ()
endwhile ()
string(JSON count LENGTH "${info}" artifacts StringZillaC variants)
string(
    JSON
    info
    SET
    "${info}"
    artifacts
    StringZillaC
    variants
    ${count}
    "${entry}"
)
file(WRITE "${bundle}/info.json" "${info}\n")
