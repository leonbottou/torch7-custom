# -*- cmake -*-

INCLUDE_DIRECTORIES(../libz ${CMAKE_CURRENT_BINARY_DIR}/../libz)

SET (src png.c pngerror.c pngget.c pngmem.c pngpread.c pngread.c
         pngrio.c pngrtran.c pngrutil.c pngset.c pngtrans.c pngwio.c
         pngwrite.c pngwtran.c pngwutil.c)

#IF(MSVC)
#  SET(src ${src} scripts/symbols.def)
#ENDIF(MSVC)

ADD_LIBRARY(libpng SHARED ${src})
TARGET_LINK_LIBRARIES(libpng zlib)

IF(UNIX)
  TARGET_LINK_LIBRARIES(libpng m)
ENDIF(UNIX)

# To compile example
# ADD_EXECUTABLE(pngtest pngtest.c)
# TARGET_LINK_LIBRARIES(pngtest png)

INSTALL(TARGETS libpng 
  LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")

INSTALL(FILES png.h pngconf.h pnginfo.h DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}")
