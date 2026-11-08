#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/stringzilla_docs_<version>.tar.gz
set -euo pipefail
version=$(cat VERSION)
name=stringzilla_docs_$version out=build/stringzilla_docs_$version
# Rustdoc's JSON is unstable, and `rust-version` needs 1.100, which reaches stable on 2026-11-12.
toolchain=${STRINGZILLA_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" build/docs-scratch
mkdir -p "$out"/{doxygen,doxygen-java,doxygen-csharp,headers,python,rust,swift,javascript,go} \
    build/docs-scratch/{library,stub/stringzilla}

rm -rf build/xml build/xml-java build/xml-csharp && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
# The bindings' passes keep the C pass's settings, but links to the JDK and the base class library stay unresolved.
(cd docs && { cat conf.dox && printf '%s\n' 'INPUT = ../java' 'FILE_PATTERNS = *.java' \
    'EXCLUDE_PATTERNS = */src/test/*' 'XML_OUTPUT = xml-java' 'WARN_AS_ERROR = NO'; } | doxygen -)
cp -r build/xml-java/. "$out/doxygen-java"
(cd docs && { cat conf.dox && printf '%s\n' 'INPUT = ../csharp' 'FILE_PATTERNS = *.cs' \
    'EXCLUDE_PATTERNS = *Tests.cs' 'XML_OUTPUT = xml-csharp' 'WARN_AS_ERROR = NO'; } | doxygen -)
cp -r build/xml-csharp/. "$out/doxygen-csharp"
git ls-files ':(glob)c/stringzilla/**/*.h' ':(glob)include/stringzilla/**/*.h' ':(glob)include/stringzilla/**/*.hpp' \
    ':(glob)cpp/stringzilla/**/*.hpp' ':(glob)java/src/main/**/*.java' 'csharp/*.cs' ':!csharp/*Tests.cs' |
    xargs cp --parents -t "$out/headers"
# griffe loads packages, not a lone stub, and `python/stringzilla/` holds the extension's C sources.
cp python/stringzilla.pyi build/docs-scratch/stub/stringzilla/__init__.pyi
griffe dump stringzilla --search build/docs-scratch/stub --no-inspection --docstyle google --full \
    --output "$out/python/stringzilla.json"
cp build/docs-scratch/stub/stringzilla/__init__.pyi "$out/python/"
STRINGZILLA_LIBRARY_DIR=$PWD/build/docs-scratch/library RUSTC_BOOTSTRAP=1 \
    cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/stringzilla.json" "$out/rust/"
mapfile -t swift_sources < <(git ls-files 'swift/*.swift' ':!swift/Test.swift')
swiftc -emit-module -emit-module-path build/docs-scratch/StringZilla.swiftmodule -module-name StringZilla \
    -package-name StringZilla -parse-as-library -I c -I include \
    -emit-symbol-graph -emit-symbol-graph-dir "$out/swift" "${swift_sources[@]}"
cp javascript/stringzilla.js javascript/stringzilla.d.ts "$out/javascript/"
cp golang/*.go golang/go.mod "$out/go/" && rm -f "$out"/go/*_test.go
# griffe, rustdoc, swiftc and Doxygen keep absolute paths, which would leak the build machine's directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/python/stringzilla.json" "$out/rust/stringzilla.json" \
    "$out"/swift/*.json "$out"/doxygen/*.xml "$out"/doxygen-java/*.xml "$out"/doxygen-csharp/*.xml
printf '{"project":"stringzilla","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" \
    > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
