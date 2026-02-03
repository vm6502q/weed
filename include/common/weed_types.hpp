//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#define _USE_MATH_DEFINES

#include "config.h"

#include <complex>
#include <cstddef>
#include <limits>
#include <math.h>
#include <memory>
#include <unordered_map>

using std::size_t;

#define IS_AMP_0(c) (norm(c) <= REAL1_EPSILON)
#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)
#define IS_SAME(c1, c2) (IS_NORM_0((c1) - (c2)))
#define IS_OPPOSITE(c1, c2) (IS_NORM_0((c1) + (c2)))

#if ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if (FPPOW < 5) && !defined(__arm__)
#include "half.hpp"
#endif

#if TCAPPOW < 8
#define tlenint uint8_t
#elif TCAPPOW < 16
#define tlenint uint16_t
#elif TCAPPOW < 32
#define tlenint uint32_t
#elif (TCAPPOW < 64) || !defined(__SIZEOF_INT128__)
#define tlenint uint64_t
#else
#define tlenint unsigned __int128
#endif

#if TCAPPOW < 4
#define tcapint uint8_t
#elif TCAPPOW < 5
#define tcapint uint16_t
#elif TCAPPOW < 6
#define tcapint uint32_t
#define WEED_MAX_DIM_POW 32
#elif TCAPPOW < 7
#define tcapint uint64_t
#define WEED_MAX_DIM_POW 64
#elif (TCAPPOW < 8) && defined(__SIZEOF_INT128__)
#define tcapint unsigned __int128
#define WEED_MAX_DIM_POW 128
#elif BOOST_AVAILABLE
#include <boost/multiprecision/cpp_int.hpp>
typedef boost::multiprecision::cpp_int tcapint;
constexpr size_t WEED_MAX_DIM_POW = (1 << TCAPPOW);
#else
#include "big_integer.hpp"
#define tcapint BigInteger
constexpr size_t WEED_MAX_DIM_POW = (1 << TCAPPOW);
#endif

#if FPPOW < 5
#ifdef __arm__
namespace Weed {
typedef __fp16 real1;
typedef float real1_f;
typedef float real1_s;
#else
#if (CPP_STD >= 23) && __has_include(<stdfloat>)
#include <stdfloat>
#endif
#if defined(__STDCPP_FLOAT16_T__)
namespace Weed {
typedef float16_t real1;
typedef float real1_f;
typedef float real1_s;
#else
#include "half.hpp"
namespace Weed {
typedef half_float::half real1;
typedef float real1_f;
typedef float real1_s;
#endif
#endif
#elif FPPOW < 6
namespace Weed {
typedef float real1;
typedef float real1_f;
typedef float real1_s;
#elif FPPOW < 7
namespace Weed {
typedef double real1;
typedef double real1_f;
typedef double real1_s;
#else
#if (CPP_STD >= 23) && __has_include(<stdfloat>)
#include <stdfloat>
#endif
#if defined(__STDCPP_FLOAT128_T__)
namespace Weed {
typedef float128_t real1;
typedef float128_t real1_f;
typedef double real1_s;
#else
#include <boost/multiprecision/float128.hpp>
#include <quadmath.h>
namespace Weed {
typedef boost::multiprecision::float128 real1;
typedef boost::multiprecision::float128 real1_f;
typedef double real1_s;
#endif
#endif

typedef std::complex<real1> complex;
constexpr tlenint bitsInCap = ((tlenint)1U) << ((tlenint)TCAPPOW);

struct Node;
typedef std::shared_ptr<Node> NodePtr;
typedef std::unique_ptr<tcapint[], void (*)(tcapint *)> IntPtr;
typedef std::unique_ptr<real1[], void (*)(real1 *)> RealPtr;
typedef std::unique_ptr<complex[], void (*)(complex *)> ComplexPtr;

#define bitsInByte 8U
#define WEED_ALIGN_SIZE 64U

#if FPPOW < 6
#define ZERO_R1_F 0.0f
#define QUARTER_R1_F 0.25f
#define HALF_R1_F 0.5f
#define ONE_R1_F 1.0f
#else
#define ZERO_R1_F 0.0
#define QUARTER_R1_F 0.25
#define HALF_R1_F 0.5
#define ONE_R1_F 1.0
#endif

#if (FPPOW > 4) || defined(__arm__)
#define WEED_CONST constexpr
#else
#define WEED_CONST const
#endif

#define REAL1_DEFAULT_ARG -999.0f
WEED_CONST real1 PI_R1 = (real1)M_PI;
WEED_CONST real1 SQRT2_R1 = (real1)M_SQRT2;
WEED_CONST real1 SQRT1_2_R1 = (real1)M_SQRT1_2;
WEED_CONST real1 E_R1 = (real1)M_E;

#if (FPPOW < 5) || (FPPOW > 6)
WEED_CONST real1 ZERO_R1 = (real1)0.0f;
WEED_CONST real1 HALF_R1 = (real1)0.5f;
WEED_CONST real1 ONE_R1 = (real1)1.0f;
#elif FPPOW == 5
#define ZERO_R1 0.0f
#define QUARTER_R1 0.25f
#define HALF_R1 0.5f
#define ONE_R1 1.0f
#else
#define ZERO_R1 0.0
#define QUARTER_R1 0.25
#define HALF_R1 0.5
#define ONE_R1 1.0
#endif

#if FPPOW < 5
// Half the probability in any single permutation of 20 maximally superposed
// qubits
WEED_CONST real1 REAL1_EPSILON = (real1)0.000000477f;
#elif FPPOW < 6
// Half the probability in any single permutation of 48 maximally superposed
// qubits
#define REAL1_EPSILON 1.7763568394002505e-15f
#elif FPPOW < 7
// Half the probability in any single permutation of 96 maximally superposed
// qubits
#define REAL1_EPSILON 6.310887241768095e-30
#else
// Half the probability in any single permutation of 192 maximally superposed
// qubits
WEED_CONST real1 REAL1_EPSILON = (real1)7.965459555662261e-59;
#endif

WEED_CONST real1 ADAM_BETA1_DEFAULT = (real1)0.9;
WEED_CONST real1 ADAM_BETA2_DEFAULT = (real1)0.999;
WEED_CONST real1 ADAM_EPSILON_DEFAULT = (real1)1e-8;

#if ENABLE_CUDA
#if FPPOW < 5
#include <cuda_fp16.h>
#define qCudaReal1 __half
#define qCudaReal2 __half2
#define qCudaReal4 __half2 *
#define qCudaCmplx __half2
#define qCudaCmplx2 __half2 *
#define qCudaReal1_f float
#define make_qCudaCmplx make_half2
#define ZERO_R1_CUDA ((qCudaReal1)0.0f)
#define REAL1_EPSILON_CUDA ((qCudaReal1)0.000000477f)
#define PI_R1_CUDA M_PI
#elif FPPOW < 6
#define qCudaReal1 float
#define qCudaReal2 float2
#define qCudaReal4 float4
#define qCudaCmplx float2
#define qCudaCmplx2 float4
#define qCudaReal1_f float
#define make_qCudaCmplx make_float2
#define make_qCudaCmplx2 make_float4
#define ZERO_R1_CUDA 0.0f
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#else
#define qCudaReal1 double
#define qCudaReal2 double2
#define qCudaReal4 double4
#define qCudaCmplx double2
#define qCudaCmplx2 double4
#define qCudaReal1_f double
#define make_qCudaCmplx make_double2
#define make_qCudaCmplx2 make_double4
#define ZERO_R1_CUDA 0.0
#define REAL1_EPSILON_CUDA REAL1_EPSILON
#define PI_R1_CUDA PI_R1
#endif
#endif

typedef std::unordered_map<tcapint, tcapint> IntSparseVector;
typedef std::unordered_map<tcapint, real1> RealSparseVector;
typedef std::unordered_map<tcapint, complex> ComplexSparseVector;

constexpr size_t SPARSE_KEY_BYTES = sizeof(tcapint) + sizeof(complex);

WEED_CONST complex ONE_CMPLX = complex(ONE_R1, ZERO_R1);
WEED_CONST complex ZERO_CMPLX = complex(ZERO_R1, ZERO_R1);
WEED_CONST complex I_CMPLX = complex(ZERO_R1, ONE_R1);
WEED_CONST complex HALF_I_HALF_CMPLX = complex(HALF_R1, HALF_R1);
WEED_CONST complex HALF_NEG_I_HALF_CMPLX = complex(HALF_R1, -HALF_R1);
WEED_CONST complex CMPLX_DEFAULT_ARG =
    complex((real1)REAL1_DEFAULT_ARG, (real1)REAL1_DEFAULT_ARG);
WEED_CONST real1 FP_NORM_EPSILON =
    (real1)(std::numeric_limits<real1>::epsilon() / 4);
WEED_CONST real1_f FP_NORM_EPSILON_F = (real1_f)FP_NORM_EPSILON;
const double FIDELITY_MIN = log((double)FP_NORM_EPSILON);
} // namespace Weed
