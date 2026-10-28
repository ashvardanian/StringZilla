# Options effecting formatting.
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True


# Options affecting comment reflow and formatting.
with section("markup"):
    # Keep comments verbatim: the reflow engine only recognizes `*` bullets, so it merges
    # dash-bulleted lines into run-on prose.
    enable_markup = False


# Commands defined here, so their keywords group like those of built-in commands.
additional_commands = {
    "stringzilla_add_cpu_capability": {"pargs": 2, "kwargs": {"BIT": 1, "GCC_FLAGS": "+", "MSVC_FLAGS": "+"}},
    "stringzilla_add_gpu_capability": {"pargs": 2, "kwargs": {"BIT": 1, "CUDA_ARCHITECTURES": "+"}},
}
