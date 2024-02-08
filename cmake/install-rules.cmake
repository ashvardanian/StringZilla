include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT stringzilla_Development)

install(
  TARGETS stringzilla stringzilla_c stringzillite
  EXPORT stringzillaTargets
  RUNTIME
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT stringzilla_Runtime
  LIBRARY
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT stringzilla_Runtime
  NAMELINK_COMPONENT stringzilla_Development
  ARCHIVE
  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT stringzilla_Development
  INCLUDES
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

write_basic_package_version_file(
  stringzillaConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)

set(
  stringzilla_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla"
  CACHE STRING "CMake package config location relative to the install prefix")
mark_as_advanced(stringzilla_INSTALL_CMAKEDIR)

install(
  FILES cmake/install-config.cmake
  DESTINATION "${stringzilla_INSTALL_CMAKEDIR}"
  RENAME stringzillaConfig.cmake
  COMPONENT stringzilla_Development)

install(
  FILES "${PROJECT_BINARY_DIR}/stringzillaConfigVersion.cmake"
  DESTINATION "${stringzilla_INSTALL_CMAKEDIR}"
  COMPONENT stringzilla_Development)

install(
  EXPORT stringzillaTargets
  NAMESPACE stringzilla::
  DESTINATION "${stringzilla_INSTALL_CMAKEDIR}"
  COMPONENT stringzilla_Development)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
