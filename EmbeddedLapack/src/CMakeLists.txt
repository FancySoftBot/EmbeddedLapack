
set(Blas_SOURCE_FILES
    Lapack/Blas/daxpy.c
    Lapack/Blas/dcopy.c
    Lapack/Blas/ddot.c
    Lapack/Blas/dgemm.c
    Lapack/Blas/dgemv.c
    Lapack/Blas/dger.c
    Lapack/Blas/dnrm2.c
    Lapack/Blas/drot.c
    Lapack/Blas/dscal.c
    Lapack/Blas/dswap.c
    Lapack/Blas/dtrmm.c
    Lapack/Blas/dtrmv.c
    Lapack/Blas/dtrsm.c
    Lapack/Blas/idamax.c
    Lapack/Blas/isamax.c
    Lapack/Blas/lsame.c
    Lapack/Blas/saxpy.c
    Lapack/Blas/scopy.c
    Lapack/Blas/sdot.c
    Lapack/Blas/sgemm.c
    Lapack/Blas/sgemv.c
    Lapack/Blas/sger.c
    Lapack/Blas/snrm2.c
    Lapack/Blas/srot.c
    Lapack/Blas/sscal.c
    Lapack/Blas/sswap.c
    Lapack/Blas/strmm.c
    Lapack/Blas/strmv.c
    Lapack/Blas/strsm.c
    Lapack/Blas/xerbla.c)

set(F2c_HEADERS
    Lapack/Include/f2c.h
    Lapack/Include/clapack.h)

set(F2c_SOURCE_FILES
    Lapack/F2c/d_lg10.c
    Lapack/F2c/d_sign.c
    Lapack/F2c/exit_.c
    Lapack/F2c/f77_aloc.c
    Lapack/F2c/i_nint.c
    Lapack/F2c/pow_dd.c
    Lapack/F2c/pow_di.c
    Lapack/F2c/pow_ri.c
    Lapack/F2c/r_lg10.c
    Lapack/F2c/r_sign.c
    Lapack/F2c/s_cat.c
    Lapack/F2c/s_cmp.c
    Lapack/F2c/s_copy.c)

set(Lapack_SOURCE_FILES
    Lapack/Install/dlamch.c
    Lapack/Install/slamch.c
    Lapack/Src/dbdsqr.c
    Lapack/Src/dgebak.c
    Lapack/Src/dgebal.c
    Lapack/Src/dgebd2.c
    Lapack/Src/dgebrd.c
    Lapack/Src/dgeev.c
    Lapack/Src/dgehd2.c
    Lapack/Src/dgehrd.c
    Lapack/Src/dgelq2.c
    Lapack/Src/dgelqf.c
    Lapack/Src/dgeqr2.c
    Lapack/Src/dgeqrf.c
    Lapack/Src/dgesv.c
    Lapack/Src/dgesvd.c
    Lapack/Src/dgetf2.c
    Lapack/Src/dgetrf.c
    Lapack/Src/dgetrs.c
    Lapack/Src/dhseqr.c
    Lapack/Src/disnan.c
    Lapack/Src/dlabad.c
    Lapack/Src/dlabrd.c
    Lapack/Src/dlacpy.c
    Lapack/Src/dladiv.c
    Lapack/Src/dlaexc.c
    Lapack/Src/dlahqr.c
    Lapack/Src/dlahr2.c
    Lapack/Src/dlaisnan.c
    Lapack/Src/dlaln2.c
    Lapack/Src/dlange.c
    Lapack/Src/dlanv2.c
    Lapack/Src/dlapy2.c
    Lapack/Src/dlaqr0.c
    Lapack/Src/dlaqr1.c
    Lapack/Src/dlaqr2.c
    Lapack/Src/dlaqr3.c
    Lapack/Src/dlaqr4.c
    Lapack/Src/dlaqr5.c
    Lapack/Src/dlarf.c
    Lapack/Src/dlarfb.c
    Lapack/Src/dlarfg.c
    Lapack/Src/dlarfp.c
    Lapack/Src/dlarft.c
    Lapack/Src/dlarfx.c
    Lapack/Src/dlartg.c
    Lapack/Src/dlas2.c
    Lapack/Src/dlascl.c
    Lapack/Src/dlaset.c
    Lapack/Src/dlasq1.c
    Lapack/Src/dlasq2.c
    Lapack/Src/dlasq3.c
    Lapack/Src/dlasq4.c
    Lapack/Src/dlasq5.c
    Lapack/Src/dlasq6.c
    Lapack/Src/dlasr.c
    Lapack/Src/dlasrt.c
    Lapack/Src/dlassq.c
    Lapack/Src/dlasv2.c
    Lapack/Src/dlaswp.c
    Lapack/Src/dlasy2.c
    Lapack/Src/dorg2r.c
    Lapack/Src/dorgbr.c
    Lapack/Src/dorghr.c
    Lapack/Src/dorgl2.c
    Lapack/Src/dorglq.c
    Lapack/Src/dorgqr.c
    Lapack/Src/dorm2r.c
    Lapack/Src/dormbr.c
    Lapack/Src/dormhr.c
    Lapack/Src/dorml2.c
    Lapack/Src/dormlq.c
    Lapack/Src/dormqr.c
    Lapack/Src/dtrevc.c
    Lapack/Src/dtrexc.c
    Lapack/Src/ieeeck.c
    Lapack/Src/iladlc.c
    Lapack/Src/iladlr.c
    Lapack/Src/ilaenv.c
    Lapack/Src/ilaslc.c
    Lapack/Src/ilaslr.c
    Lapack/Src/iparmq.c
    Lapack/Src/sgebak.c
    Lapack/Src/sgebal.c
    Lapack/Src/sgeev.c
    Lapack/Src/sgehd2.c
    Lapack/Src/sgehrd.c
    Lapack/Src/sgeqr2.c
    Lapack/Src/sgetf2.c
    Lapack/Src/sgetrf.c
    Lapack/Src/shseqr.c
    Lapack/Src/sisnan.c
    Lapack/Src/slabad.c
    Lapack/Src/slacpy.c
    Lapack/Src/sladiv.c
    Lapack/Src/slaexc.c
    Lapack/Src/slahqr.c
    Lapack/Src/slahr2.c
    Lapack/Src/slaisnan.c
    Lapack/Src/slaln2.c
    Lapack/Src/slange.c
    Lapack/Src/slanv2.c
    Lapack/Src/slapy2.c
    Lapack/Src/slaqr0.c
    Lapack/Src/slaqr1.c
    Lapack/Src/slaqr2.c
    Lapack/Src/slaqr3.c
    Lapack/Src/slaqr4.c
    Lapack/Src/slaqr5.c
    Lapack/Src/slarf.c
    Lapack/Src/slarfb.c
    Lapack/Src/slarfg.c
    Lapack/Src/slarfp.c
    Lapack/Src/slarft.c
    Lapack/Src/slarfx.c
    Lapack/Src/slartg.c
    Lapack/Src/slascl.c
    Lapack/Src/slaset.c
    Lapack/Src/slassq.c
    Lapack/Src/slaswp.c
    Lapack/Src/slasy2.c
    Lapack/Src/sorg2r.c
    Lapack/Src/sorghr.c
    Lapack/Src/sorgqr.c
    Lapack/Src/sorm2r.c
    Lapack/Src/sormhr.c
    Lapack/Src/sormqr.c
    Lapack/Src/strevc.c
    Lapack/Src/strexc.c)

set(EmbeddedLapack_HEADERS
    ${F2c_HEADERS})

set(EmbeddedLapack_SOURCE_FILES
    ${Blas_SOURCE_FILES}
    ${F2c_SOURCE_FILES}
    ${Lapack_SOURCE_FILES})

add_library(EmbeddedLapack STATIC
    ${EmbeddedLapack_HEADERS}
    ${EmbeddedLapack_SOURCE_FILES})

# library version
add_library(EmbeddedLapack::EmbeddedLapack ALIAS EmbeddedLapack)
set_target_properties(
    EmbeddedLapack
    PROPERTIES VERSION ${PROJECT_VERSION})

# Link-time optimization
if (LTO_SUPPORTED)
    set_target_properties(EmbeddedLapack
        PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

# standard
target_compile_features(
    EmbeddedLapack PUBLIC
    c_std_${CMAKE_C_STANDARD}
    cxx_std_${CMAKE_CXX_STANDARD})

# definitions
target_compile_definitions(
    EmbeddedLapack PUBLIC
    _FORTIFY_SOURCE=1)

# C/CXX flags
target_compile_options(
    EmbeddedLapack PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${SUPPORTED_CXX_COMPILE_FLAGS}>
    $<$<COMPILE_LANGUAGE:C>:${SUPPORTED_C_COMPILE_FLAGS}>)

# link flags
target_link_options(
    EmbeddedLapack PUBLIC
    ${SUPPORTED_LINK_FLAGS})

# include directories
target_include_directories(
    EmbeddedLapack PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Lapack/Include>
    $<INSTALL_INTERFACE:Lapack/Include>
)
