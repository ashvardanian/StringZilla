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
get_filename_component(archive_name_ "${STRINGZILLA_SWIFT_ARCHIVE}" NAME)

if (STRINGZILLA_SWIFT_SDK)
    set(slices_ "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.slices")
    file(MAKE_DIRECTORY "${slices_}/${STRINGZILLA_SWIFT_SDK}/${STRINGZILLA_SWIFT_ARCHITECTURE}")
    file(COPY_FILE "${STRINGZILLA_SWIFT_ARCHIVE}"
         "${slices_}/${STRINGZILLA_SWIFT_SDK}/${STRINGZILLA_SWIFT_ARCHITECTURE}/${archive_name_}"
    )

    set(libraries_)
    file(
        GLOB sdks_
        LIST_DIRECTORIES true
        "${slices_}/*"
    )
    foreach (sdk_ IN LISTS sdks_)
        file(GLOB architectures_ "${sdk_}/*/${archive_name_}")
        execute_process(
            COMMAND lipo -create ${architectures_} -output "${sdk_}/${archive_name_}" COMMAND_ERROR_IS_FATAL ANY
        )
        list(APPEND libraries_ -library "${sdk_}/${archive_name_}" -headers "${STRINGZILLA_SWIFT_HEADERS}")
    endforeach ()
    file(REMOVE_RECURSE "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.xcframework")
    execute_process(
        COMMAND xcodebuild -create-xcframework ${libraries_} -output
                "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.xcframework" COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET
    )
    return()
endif ()

# One directory per variant, named by its first triple, beside one `include/` every variant shares.
set(bundle_ "${STRINGZILLA_SWIFT_DIRECTORY}/StringZillaC.artifactbundle")
string(REPLACE "," ";" triples_ "${STRINGZILLA_SWIFT_TRIPLES}")
list(GET triples_ 0 variant_)
file(MAKE_DIRECTORY "${bundle_}/${variant_}")
file(COPY_FILE "${STRINGZILLA_SWIFT_ARCHIVE}" "${bundle_}/${variant_}/${archive_name_}")
file(REMOVE_RECURSE "${bundle_}/include")
file(COPY "${STRINGZILLA_SWIFT_HEADERS}/" DESTINATION "${bundle_}/include")

set(entry_ [[{"path": "", "supportedTriples": [], "staticLibraryMetadata": {"headerPaths": ["include"],
    "moduleMapPath": "include/module.modulemap"}}]]
)
string(JSON entry_ SET "${entry_}" path "\"${variant_}/${archive_name_}\"")
foreach (triple_ IN LISTS triples_)
    string(JSON count_ LENGTH "${entry_}" supportedTriples)
    string(
        JSON
        entry_
        SET
        "${entry_}"
        supportedTriples
        ${count_}
        "\"${triple_}\""
    )
endforeach ()

if (EXISTS "${bundle_}/info.json")
    file(READ "${bundle_}/info.json" info_)
else ()
    set(info_ [[{"schemaVersion": "1.0", "artifacts": {"StringZillaC": {"type": "staticLibrary", "variants": []}}}]])
endif ()
string(
    JSON
    info_
    SET
    "${info_}"
    artifacts
    StringZillaC
    version
    "\"${STRINGZILLA_SWIFT_VERSION}\""
)
# A rebuild of the same variant replaces it.
string(JSON count_ LENGTH "${info_}" artifacts StringZillaC variants)
while (count_ GREATER 0)
    math(EXPR count_ "${count_} - 1")
    string(
        JSON
        path_
        GET
        "${info_}"
        artifacts
        StringZillaC
        variants
        ${count_}
        path
    )
    if (path_ STREQUAL "${variant_}/${archive_name_}")
        string(
            JSON
            info_
            REMOVE
            "${info_}"
            artifacts
            StringZillaC
            variants
            ${count_}
        )
    endif ()
endwhile ()
string(JSON count_ LENGTH "${info_}" artifacts StringZillaC variants)
string(
    JSON
    info_
    SET
    "${info_}"
    artifacts
    StringZillaC
    variants
    ${count_}
    "${entry_}"
)
file(WRITE "${bundle_}/info.json" "${info_}\n")
