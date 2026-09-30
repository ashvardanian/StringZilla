// swift-tools-version:6.4
import PackageDescription

// CMake builds and probes the C library, as for every other binding, and SwiftPM links what it
// produced: an XCFramework on Apple platforms and an artifact bundle elsewhere. A checkout points
// `STRINGZILLA_SWIFT_ARTIFACT` at the one `cmake --build --preset swift` wrote, relative to this
// directory; everyone else downloads the release's, whose checksums the release workflow writes.
let release = "https://github.com/ashvardanian/StringZilla/releases/download/v5.1.2"
let appleChecksum = "0000000000000000000000000000000000000000000000000000000000000000"
let portableChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let library: [Target]
let libraryDependencies: [Target.Dependency]
if let artifact = Context.environment["STRINGZILLA_SWIFT_ARTIFACT"] {
    library = [.binaryTarget(name: "StringZillaC", path: artifact)]
    libraryDependencies = ["StringZillaC"]
}
else {
    library = [
        .binaryTarget(
            name: "StringZillaCApple",
            url: "\(release)/StringZillaC.xcframework.zip",
            checksum: appleChecksum
        ),
        .binaryTarget(
            name: "StringZillaCPortable",
            url: "\(release)/StringZillaC.artifactbundle.zip",
            checksum: portableChecksum
        ),
    ]
    libraryDependencies = [
        .target(
            name: "StringZillaCApple",
            condition: .when(platforms: [.macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS])
        ),
        .target(name: "StringZillaCPortable", condition: .when(platforms: [.linux, .android, .windows, .wasi])),
    ]
}

let package = Package(
    name: "StringZilla",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
        .watchOS(.v9),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "StringZilla", targets: ["StringZilla"])
    ],
    targets: library + [
        .target(
            name: "StringZilla",
            dependencies: libraryDependencies,
            path: "swift",
            exclude: ["Test.swift", "README.md"],
            sources: ["StringProtocol+StringZilla.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZilla"],
            path: "swift",
            exclude: ["StringProtocol+StringZilla.swift", "README.md"],
            sources: ["Test.swift"]
        ),
    ]
)
