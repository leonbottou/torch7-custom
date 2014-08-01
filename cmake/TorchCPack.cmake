SET(CPACK_PACKAGE_CONTACT "ronan [at] collobert [dot] com")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Torch7 machine-learning platform.
 Torch7 provides a Matlab-like environment for state-of-the-art
 machine-learning algorithms. It is easy to use and provides a very
 efficient implementation, thanks to an easy and fast scripting language
 (Lua) and a underlying C implementation.  
 
 Torch7-custom-lb was highly customized to run smoothly on Windows
 with a single installer and high performance convolutions.")

SET(CPACK_PACKAGE_NAME "Torch7-custom-lb")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/COPYRIGHT.txt")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
SET(CPACK_PACKAGE_VERSION_MAJOR 7)
SET(CPACK_PACKAGE_VERSION_MINOR 1)
SET(CPACK_PACKAGE_VERSION_PATCH 1)
SET(CPACK_CREATE_DESKTOP_LINKS wintorch)
SET(CPACK_PACKAGE_EXECUTABLES "wintorch" "WinTorch")
SET(CPACK_NSIS_MENU_LINKS "http://www.torch.ch" "Torch Help")
SET(CPACK_PACKAGE_VENDOR "The Torch Team")
SET(CPACK_INSTALL_COMMANDS "${CMAKE_BUILD_TOOL}")
INCLUDE(InstallRequiredSystemLibraries)
INSTALL(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")
INSTALL(PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

IF( CMAKE_SIZEOF_VOID_P EQUAL 8 )
 SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
ELSE()
 SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES32")
ENDIF()

# Debian related stuff
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libgcc1, libreadline5, libncurses5, libqtgui4 (>= 4.4.0)")
SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

# MacOS X Bundle
SET(CPACK_BUNDLE_PLIST "${CMAKE_BINARY_DIR}/exe/qtlua/qlua/Info.plist")
SET(CPACK_BUNDLE_STARTUP_COMMAND macqlua)
SET(CPACK_BUNDLE_ICON "${CMAKE_SOURCE_DIR}/exe/qtlua/qlua/images/torch.icns")

INCLUDE(CPack)
