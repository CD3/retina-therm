#include <cmath>

#include <boost/math/distributions/non_central_chi_squared.hpp>

#include <emscripten.h>
extern "C" {

EMSCRIPTEN_KEEPALIVE
double MarcumQFunction(double a_v, double a_a, double b_b)
{
  boost::math::non_central_chi_squared dist(2 * a_v, a_a * a_a);
  return 1 - boost::math::cdf(dist, b_b * b_b);
}

EMSCRIPTEN_KEEPALIVE
double MarcumQFunction10(double b_b)
{
  return std::exp(-b_b * b_b / 2);
}
}
