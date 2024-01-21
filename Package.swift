// swift-tools-version:5.3

import PackageDescription

let package = Package(
    name: "StringZilla",
    products: [
        .library(
            name: "StringZilla",
            targets: ["StringZilla", "StringZillaC"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "StringZillaC",
            dependencies: [],
            path: "swift",
            sources: ["SwiftPackageManager.c"],
            publicHeadersPath: "./include/",
            cSettings: [
                .headerSearchPath("./include/"),
                .define("SZ_PUBLIC", to: ""),
                .define("SZ_DEBUG", to: "0")
            ]
        ),
        .target(
            name: "StringZilla",
            dependencies: ["StringZillaC"],
            path: "swift",
            exclude: ["Test.swift", "SwiftPackageManager.c"],
            sources: ["StringProtocol+StringZilla.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZilla"],
            path: "swift",
            exclude: ["StringProtocol+StringZilla.swift", "SwiftPackageManager.c"],
            sources: ["Test.swift"]
        )
    ],
    cLanguageStandard: CLanguageStandard.c99,
    cxxLanguageStandard: CXXLanguageStandard.cxx14
)
