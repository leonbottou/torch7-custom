# -*- cmake -*-


SET (src 
        jcapimin.c jcapistd.c jccoefct.c jccolor.c jcdctmgr.c jchuff.c 
        jcinit.c jcmainct.c jcmarker.c jcmaster.c jcomapi.c jcparam.c 
        jcphuff.c jcprepct.c jcsample.c jctrans.c jdapimin.c jdapistd.c 
        jdatadst.c jdatasrc.c jdcoefct.c jdcolor.c jddctmgr.c jdhuff.c 
        jdinput.c jdmainct.c jdmarker.c jdmaster.c jdmerge.c jdphuff.c 
        jdpostct.c jdsample.c jdtrans.c jerror.c jfdctflt.c jfdctfst.c 
        jfdctint.c jidctflt.c jidctfst.c jidctint.c jidctred.c jquant1.c 
        jquant2.c jutils.c jmemmgr.c jmemansi.c )


IF(MSVC)
  SET(src ${src} libjpeg.def)
ENDIF(MSVC)

ADD_LIBRARY(jpeg6 SHARED ${src})

INSTALL(TARGETS jpeg6 
  LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")

INSTALL(FILES jpeglib.h jconfig.h jmorecfg.h
           DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}")
