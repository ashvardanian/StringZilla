{
  "targets": [
    {
      "target_name": "stringzilla",
      "sources": [
        "javascript/lib.c",
        "c/stringzilla.c"
      ],
      "include_dirs": [
        "include"
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
                    "SZ_USE_NEHALEM=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=1",
                    "SZ_USE_ICE=1",
                    "SZ_USE_NEON=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_NEHALEM=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEON_AES=1",
                    "SZ_USE_SVE=1",
                    "SZ_USE_SVE2=1",
                    "SZ_USE_SVE2_AES=1"
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
                    "SZ_USE_NEHALEM=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICE=0",
                    "SZ_USE_NEON=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_NEHALEM=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEON_AES=1",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0"
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
                    "SZ_USE_NEHALEM=1",
                    "SZ_USE_HASWELL=1",
                    "SZ_USE_SKYLAKE=1",
                    "SZ_USE_ICE=1",
                    "SZ_USE_NEON=0",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0"
                  ]
                }
              ],
              [
                "target_arch=='arm64'",
                {
                  "defines": [
                    "SZ_USE_NEHALEM=0",
                    "SZ_USE_HASWELL=0",
                    "SZ_USE_SKYLAKE=0",
                    "SZ_USE_ICE=0",
                    "SZ_USE_NEON=1",
                    "SZ_USE_NEON_AES=1",
                    "SZ_USE_SVE=0",
                    "SZ_USE_SVE2=0"
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