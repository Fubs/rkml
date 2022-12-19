#pragma once
#include <array>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace boost::multiprecision;

// using dtype = cpp_bin_float_quad; constexpr bool USING_MPFR = false;
// using dtype = cpp_dec_float_50; constexpr bool USING_MPFR = false;
//using dtype = mpf_float; constexpr bool USING_MPFR = true;
//using dtype = mpfr_float; constexpr bool USING_MPFR = true;
// using dtype = mpfr_float_100; constexpr bool USING_MPFR = true;
//using dtype = mpfr_float_50; constexpr bool USING_MPFR = true;
//using dtype = mpfr_float_50; constexpr bool USING_MPFR = true;
using dtype = long double; constexpr bool USING_MPFR = false;
// using dtype = double; constexpr bool USING_MPFR = false;
//using dtype = float;constexpr bool USING_MPFR = false;

/* chua circuit ram limit
constexpr size_t NUM_THREADS = size_t(8);
constexpr dtype initStep = (dtype(1)/150000);
constexpr size_t smallestSplit = NUM_THREADS;
constexpr dtype tmax = (dtype(100));
constexpr dtype g = dtype(9.8);
constexpr bool write_to_file = false;
//#define L dtype(1)
constexpr int ORD = 3;
*/

#ifndef NUM_THREADS
constexpr size_t NUM_THREADS = 256;
#endif
// constexpr size_t NUM_THREADS = size_t(128); //for pgo
#define initStep (dtype(1))
constexpr size_t smallestSplit = NUM_THREADS;
#define tmax (dtype(20))
constexpr bool write_to_file = true;
//#define L dtype(1)
#ifdef ORD_1
constexpr int ORD = 1;
#endif
#ifdef ORD_2
constexpr int ORD = 2;
#endif
#ifdef ORD_3
constexpr int ORD = 3;
#endif
#ifndef ORD_1
#ifndef ORD_2
#ifndef ORD_3
constexpr int ORD = 2;
#endif
#endif
#endif

using dfns =
    std::vector<dtype (*)(std::array<dtype, ORD + 1>)>;  // vector of func ptrs

#define PI boost::math::constants::pi<dtype>()
#define EUL boost::math::constants::e<dtype>()
#define lil_g dtype(9.8)
