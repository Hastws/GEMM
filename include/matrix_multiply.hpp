#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_MATRIX_MULTIPLY_HPP
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_MATRIX_MULTIPLY_HPP

// #define BACKEND_AVX2

#if !defined(BACKEND_REF) && !defined(BACKEND_AVX2) && !defined(BACKEND_NEON)
#define BACKEND_REF
#endif

#if defined(BACKEND_AVX2)
#include "backend_avx2.hpp"
#elif defined(BACKEND_NEON)
#include "backend_neon.hpp"
#elif defined(BACKEND_REF)
#include "backend_ref.hpp"
#else
#error "Define one of: BACKEND_REF / BACKEND_AVX2 / BACKEND_NEON"
#endif

#endif