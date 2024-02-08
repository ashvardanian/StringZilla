set(warning_guard "")
if(NOT PROJECT_IS_TOP_LEVEL)
  option(
    stringzilla_INCLUDES_WITH_SYSTEM
    "Use SYSTEM modifier for stringzilla's includes, disabling warnings"
    ON)
  mark_as_advanced(stringzilla_INCLUDES_WITH_SYSTEM)
  if(stringzilla_INCLUDES_WITH_SYSTEM)
    set(warning_guard SYSTEM)
  endif()
endif()
