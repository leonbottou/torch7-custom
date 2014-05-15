# -*- cmake -*-


SET (src 
    jcomapi.c jdapimin.c jdapistd.c jdatasrc.c jdcoefct.c jdcolor.c
    jddctmgr.c jdhuff.c jdinput.c jdmainct.c jdmarker.c jdmaster.c
    jdmerge.c jdphuff.c jdpostct.c jdsample.c jerror.c jidctint.c
    jmemmgr.c jutils.c)


IF(MSVC)
  SET(src ${src} libjpeg.def)
ENDIF(MSVC)

ADD_LIBRARY(libjpeg SHARED ${src})

INSTALL(TARGETS libjpeg 
  LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")

INSTALL(FILES  DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}")
