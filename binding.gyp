{
    "targets": [
        {
            "target_name": "stringzilla",
            "sources": ["javascript/lib.c"],
            "include_dirs": ["include"],
            "cflags": ["-std=c99", "-Wno-unknown-pragmas", "-Wno-maybe-uninitialized"],
        }
    ]
}
