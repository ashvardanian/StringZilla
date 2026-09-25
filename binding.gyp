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
        "c/stringzilla/cipher.c",
        "c/stringzilla/find.c",
        "c/stringzilla/sort.c",
        "c/stringzilla/intersect.c",
        "c/stringzilla/utf8_runes.c",
        "c/stringzilla/utf8_tokens.c",
        "c/stringzilla/utf8_wordbreaks.c",
        "c/stringzilla/utf8_graphemes.c",
        "c/stringzilla/utf8_sentences.c",
        "c/stringzilla/utf8_linebreaks.c",
        "c/stringzilla/utf8_norm.c",
        "c/stringzilla/utf8_uncased.c",
        "c/stringzilla/utf8_uncased_fold.c"
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
        "STRINGZILLA_RUNTIME_DISPATCH=1"
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
                    "STRINGZILLA_TARGET_WESTMERE=1",
                    "STRINGZILLA_TARGET_GOLDMONT=1",
                    "STRINGZILLA_TARGET_HASWELL=1",
                    "STRINGZILLA_TARGET_SKYLAKE=1",
                    "STRINGZILLA_TARGET_ICELAKE=1",
                    "STRINGZILLA_TARGET_NEON=0",
                    "STRINGZILLA_TARGET_NEONAES=0",
                    "STRINGZILLA_TARGET_NEONSHA=0",
                    "STRINGZILLA_TARGET_SVE=0",
                    "STRINGZILLA_TARGET_SVE2=0",
                    "STRINGZILLA_TARGET_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "STRINGZILLA_TARGET_WESTMERE=0",
                    "STRINGZILLA_TARGET_HASWELL=0",
                    "STRINGZILLA_TARGET_SKYLAKE=0",
                    "STRINGZILLA_TARGET_ICELAKE=0",
                    "STRINGZILLA_TARGET_NEON=1",
                    "STRINGZILLA_TARGET_NEONAES=1",
                    "STRINGZILLA_TARGET_NEONSHA=1",
                    "STRINGZILLA_TARGET_SVE=1",
                    "STRINGZILLA_TARGET_SVE2=1",
                    "STRINGZILLA_TARGET_SVE2AES=1"
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
                    "STRINGZILLA_TARGET_WESTMERE=1",
                    "STRINGZILLA_TARGET_GOLDMONT=1",
                    "STRINGZILLA_TARGET_HASWELL=1",
                    "STRINGZILLA_TARGET_SKYLAKE=0",
                    "STRINGZILLA_TARGET_ICELAKE=0",
                    "STRINGZILLA_TARGET_NEON=0",
                    "STRINGZILLA_TARGET_NEONAES=0",
                    "STRINGZILLA_TARGET_NEONSHA=0",
                    "STRINGZILLA_TARGET_SVE=0",
                    "STRINGZILLA_TARGET_SVE2=0",
                    "STRINGZILLA_TARGET_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "STRINGZILLA_TARGET_WESTMERE=0",
                    "STRINGZILLA_TARGET_GOLDMONT=0",
                    "STRINGZILLA_TARGET_HASWELL=0",
                    "STRINGZILLA_TARGET_SKYLAKE=0",
                    "STRINGZILLA_TARGET_ICELAKE=0",
                    "STRINGZILLA_TARGET_NEON=1",
                    "STRINGZILLA_TARGET_NEONAES=1",
                    "STRINGZILLA_TARGET_NEONSHA=1",
                    "STRINGZILLA_TARGET_SVE=0",
                    "STRINGZILLA_TARGET_SVE2=0",
                    "STRINGZILLA_TARGET_SVE2AES=0"
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
                    "STRINGZILLA_TARGET_WESTMERE=1",
                    "STRINGZILLA_TARGET_GOLDMONT=1",
                    "STRINGZILLA_TARGET_HASWELL=1",
                    "STRINGZILLA_TARGET_SKYLAKE=1",
                    "STRINGZILLA_TARGET_ICELAKE=1",
                    "STRINGZILLA_TARGET_NEON=0",
                    "STRINGZILLA_TARGET_NEONAES=0",
                    "STRINGZILLA_TARGET_NEONSHA=0",
                    "STRINGZILLA_TARGET_SVE=0",
                    "STRINGZILLA_TARGET_SVE2=0",
                    "STRINGZILLA_TARGET_SVE2AES=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "STRINGZILLA_TARGET_WESTMERE=0",
                    "STRINGZILLA_TARGET_GOLDMONT=0",
                    "STRINGZILLA_TARGET_HASWELL=0",
                    "STRINGZILLA_TARGET_SKYLAKE=0",
                    "STRINGZILLA_TARGET_ICELAKE=0",
                    "STRINGZILLA_TARGET_NEON=1",
                    "STRINGZILLA_TARGET_NEONAES=1",
                    "STRINGZILLA_TARGET_NEONSHA=1",
                    "STRINGZILLA_TARGET_SVE=0",
                    "STRINGZILLA_TARGET_SVE2=0",
                    "STRINGZILLA_TARGET_SVE2AES=0"
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