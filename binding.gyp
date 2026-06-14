{
  "targets": [
    {
      "target_name": "stringzilla",
      "sources": [
        "javascript/lib.c",
        "c/stringzilla/runtime.c",
        "c/stringzilla/compare.c",
        "c/stringzilla/memory.c",
        "c/stringzilla/hash.c",
        "c/stringzilla/find.c",
        "c/stringzilla/sort.c",
        "c/stringzilla/intersect.c",
        "c/stringzilla/utf8_norm.c",
        "c/stringzilla/utf8_iterate.c",
        "c/stringzilla/utf8_case_fold.c",
        "c/stringzilla/utf8_case_insensitive.c"
      ],
      "include_dirs": [
        "include",
        "c/stringzilla"
      ],
      "cflags": [
        "-std=c99",
        "-fPIC",
        "-Wno-unknown-pragmas",
        "-Wno-maybe-uninitialized",
        "-Wno-cast-function-type",
        "-Wno-unused-function"
      ],
      "defines": [
        "SZ_DYNAMIC_DISPATCH=1"
      ],
      "conditions": [
        [
          "OS=='linux' or OS=='freebsd'",
          {
            "conditions": [
              [
                "target_arch=='x64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=1",
                    "SZ_USE_GOLDMONT=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=1",
                    "SZ_USE_ICELAKE=1",
                    "SZ_USE_NEON=0",
                    "SZ_USE_NEONAES=0",
                    "SZ_USE_NEONSHA=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0",
                    "SZ_USE_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICELAKE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEONAES=1",
                    "SZ_USE_NEONSHA=1",
                    "SZ_USE_SVE=1",
                    "SZ_USE_SVE2=1",
                    "SZ_USE_SVE2AES=1"
                  ]
                }
              ]
            ]
          }
        ],
        [
          "OS=='mac'",
          {
            "conditions": [
              [
                "target_arch=='x64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=1",
                    "SZ_USE_GOLDMONT=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICELAKE=0",
                    "SZ_USE_NEON=0",
                    "SZ_USE_NEONAES=0",
                    "SZ_USE_NEONSHA=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0",
                    "SZ_USE_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=0",
                    "SZ_USE_GOLDMONT=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICELAKE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEONAES=1",
                    "SZ_USE_NEONSHA=1",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0",
                    "SZ_USE_SVE2AES=0"
                  ]
                }
              ]
            ]
          }
        ],
        [
          "OS=='win'",
          {
            "conditions": [
              [
                "target_arch=='x64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=1",
                    "SZ_USE_GOLDMONT=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=1",
                    "SZ_USE_ICELAKE=1",
                    "SZ_USE_NEON=0",
                    "SZ_USE_NEONAES=0",
                    "SZ_USE_NEONSHA=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0",
                    "SZ_USE_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_WESTMERE=0",
                    "SZ_USE_GOLDMONT=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICELAKE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEONAES=1",
                    "SZ_USE_NEONSHA=1",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0",
                    "SZ_USE_SVE2AES=0"
                  ]
                }
              ]
            ]
          }
        ]
      ]
    }
  ]
}