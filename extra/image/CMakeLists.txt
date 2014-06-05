CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(JPEG QUIET)
FIND_PACKAGE(PNG QUIET)


SET(src ppm.c)
ADD_TORCH_PACKAGE(ppm "${src}" "${luasrc}" "Image Processing")
TARGET_LINK_LIBRARIES(ppm luaT TH)

if (NOT JPEG_FOUND)
  ADD_SUBDIRECTORY(libjpeg)
  SET(JPEG_LIBRARIES jpeg6)
  SET(JPEG_INCLUDE_DIR $(CMAKE_CURRENT_SOURCE_DIR)/libjpeg)
  SET(JPEG_FOUND 1)
endif (NOT JPEG_FOUND)

if (NOT PNG_FOUND)
  ADD_SUBDIRECTORY(libz)
  ADD_SUBDIRECTORY(libpng)
  SET(PNG_LIBRARIES png1)
  SET(PNG_INCLUDE_DIR $(CMAKE_CURRENT_SOURCE_DIR)/libpng)
  SET(PNG_FOUND 1)
endif (NOT PNG_FOUND)

if (JPEG_FOUND)
    SET(src jpeg.c)
    include_directories (${JPEG_INCLUDE_DIR})
    ADD_TORCH_PACKAGE(jpeg "${src}" "${luasrc}" "Image Processing")
    TARGET_LINK_LIBRARIES(jpeg luaT TH ${JPEG_LIBRARIES})
else (JPEG_FOUND)
    message ("WARNING: Could not find JPEG libraries, JPEG wrapper will not be installed")
endif (JPEG_FOUND)

if (PNG_FOUND)
    SET(src png.c)
    include_directories (${PNG_INCLUDE_DIR})
    ADD_TORCH_PACKAGE(png "${src}" "${luasrc}" "Image Processing")
    TARGET_LINK_LIBRARIES(png luaT TH ${PNG_LIBRARIES})
else (PNG_FOUND)
    message ("WARNING: Could not find PNG libraries, PNG wrapper will not be installed")
endif (PNG_FOUND)

SET(src image.c)
SET(luasrc init.lua lena.jpg lena.png lena_full.jpg lena_full.png win.ui)

ADD_TORCH_PACKAGE(image "${src}" "${luasrc}" "Image Processing")

TARGET_LINK_LIBRARIES(image luaT TH)
