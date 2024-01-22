#!/bin/sh

echo $1 > VERSION && 
    sed -i "s/^\(#define STRINGZILLA_VERSION_MAJOR \).*/\1$(echo "$1" | cut -d. -f1)/" ./include/stringzilla/stringzilla.h &&
    sed -i "s/^\(#define STRINGZILLA_VERSION_MINOR \).*/\1$(echo "$1" | cut -d. -f2)/" ./include/stringzilla/stringzilla.h &&
    sed -i "s/^\(#define STRINGZILLA_VERSION_PATCH \).*/\1$(echo "$1" | cut -d. -f3)/" ./include/stringzilla/stringzilla.h &&
    sed -i "s/VERSION [0-9]\+\.[0-9]\+\.[0-9]\+/VERSION $1/" CMakeLists.txt &&
    sed -i "s/version = \".*\"/version = \"$1\"/" Cargo.toml &&
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json
