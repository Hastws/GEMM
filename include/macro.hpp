#ifndef GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_MACRO_HPP
#define GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_MACRO_HPP

#if defined(__GNUC__) || defined(__clang__)
#define AA_ALIGN(N) __attribute__((aligned(N)))
#define AA_ALWAYS_INLINE __attribute__((always_inline)) inline
#define AA_RESTRICT __restrict__
#else
#define AA_ALIGN(N) __declspec(align(N))
#define AA_ALWAYS_INLINE __forceinline
#define AA_RESTRICT __restrict
#endif

#endif  // GENERAL_MATRIX_MATRIX_MULTIPLY_INCLUDE_MACRO_HPP
