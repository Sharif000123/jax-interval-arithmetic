#include <cmath>
#include <cstdint>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
// #include <string>
// include boost

namespace nb = nanobind;

void printing(const char *message, int number)
{
  std::cout << message << number << std::endl;
}

float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y)
{
  float sm = 0.0f;
  for (int64_t n = 0; n < size; ++n)
  {
    sm += x[n] * x[n];
  }
  float scale = 1.0f / std::sqrt(sm / float(size) + eps);
  for (int64_t n = 0; n < size; ++n)
  {
    y[n] = x[n] * scale;
  }
  return scale;
}

double mult(double a, double b)
{
  return a * b;
}

NB_MODULE(ffi_module, m)
{
  m.def("printing", &printing, "Prints a message with a number");
  m.def("ComputeRmsNorm", &ComputeRmsNorm, "Computes the RMS norm of a vector",
        nb::arg("eps"), nb::arg("size"), nb::arg("x"), nb::arg("y"));
  m.def("mult", &mult, "Multiplies two numbers", nb::arg("a"), nb::arg("b"));
}

// int main()
// {
//   // Example usage
//   std::cout << "Test" << std::endl;
//   // printing("Hello, number: ", 42);
//   return 0;
// }

/*  lax.add_p: interval_add,
    lax.mul_p: interval_mult_elementwise,
    lax.dot_general_p: interval_matrix_mult,
    lax.pow_p: interval_pow,
    lax.transpose_p: interval_transpose,
    lax.reduce_min_p: interval_min,
    lax.reduce_max_p: interval_max,
    lax.max_p: interval_relu,*/