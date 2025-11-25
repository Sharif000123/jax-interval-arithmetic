#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <boost/numeric/interval.hpp>
#include <sstream>
#include <D:/Work/ML Stuff/nanobind/include/nanobind/nanobind.h>

namespace nb = nanobind;
using Interval = boost::numeric::interval<double>;
using IntervalMatrix = std::vector<std::vector<Interval>>;

NB_MAKE_OPAQUE(IntervalMatrix);

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

Interval add(const Interval &a, const Interval &b)
{
  return a + b;
}

Interval mult(const Interval &a, const Interval &b)
{
  return a * b;
}

IntervalMatrix matrixMult(const IntervalMatrix &A, const IntervalMatrix &B)
{
  if (A.empty() || B.empty() || A[0].empty() || B[0].empty())
  {
    throw std::invalid_argument("Input matrices cannot be empty.");
  }
  if (A[0].size() != B.size())
  {
    std::ostringstream oss;
    oss << "Incompatable matrix dimensions. A cols: " << A[0].size() << "B rows: " << B.size();
    throw std::invalid_argument(oss.str());
  }

  size_t n = A.size(), m = B[0].size(), p = B.size();
  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      for (size_t k = 0; k < p; ++k)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

IntervalMatrix matrixMultElementwise(const IntervalMatrix &A, const IntervalMatrix &B)
{
  if (A.empty() || B.empty() || A[0].empty() || B[0].empty())
  {
    throw std::invalid_argument("Input matrices cannot be empty.");
  }
  if (A[0].size() != B[0].size() || A.size() != B.size())
  {
    std::ostringstream oss;
    oss << "Incompatable matrix dimensions for elementwise mult. A rows: " << A.size() << "B rows: " << B.size();
    throw std::invalid_argument(oss.str());
  }

  size_t n = A.size();
  IntervalMatrix C(n, std::vector<Interval>(n, Interval(0.0, 0.0)));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      C[i][j] = A[i][j] * B[i][j];
  return C;
}
IntervalMatrix elementwiseAdd(const IntervalMatrix &X, const IntervalMatrix &Y)
{
  size_t x_rows = X.size(), x_cols = X[0].size();
  size_t y_rows = Y.size(), y_cols = Y[0].size();
  if (x_rows != y_rows || x_cols != y_cols || x_rows == 0 || x_cols == 0)
  {
    throw std::invalid_argument("Input matrix is empty, or has wrong shape.");
  }

  IntervalMatrix C(x_rows, std::vector<Interval>(x_cols, Interval(0.0, 0.0)));

  for (size_t i = 0; i < x_rows; i++)
  {
    for (size_t j = 0; j < x_cols; j++)
    {
      C[i][j] = X[i][j] + Y[i][j];
    }
  }
  return C;
}

IntervalMatrix matrixAddBias(const IntervalMatrix &A, const IntervalMatrix &B)
{
  size_t n = A.size(), m = A[0].size();
  if (n == 0 || A[0].size() == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix is empty.");
  }
  if (B.size() != 1) // bias must be 1 dimensional
  {
    throw std::invalid_argument("Bias must be one dimensional");
  }
  if (B[0].size() != m) // bias must be length of matrix row
  {
    throw std::invalid_argument("Bias must be length of matrix row");
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));
  // for (size_t i = 0; i < n; ++i)
  //   for (size_t j = 0; j < m; ++j)
  //     C[i][j] += A[i][j] + B[0][j];

  constexpr size_t VF = 4; // or 8, 16 depending on your arch
  for (size_t i = 0; i < n; ++i)
  {
    size_t j = 0;
    // process full blocks
    for (; j + VF <= m; j += VF)
    {
      // Load blocks
      for (size_t k = 0; k < VF; ++k)
      {
        C[i][j + k] += A[i][j + k] + B[0][j + k];
      }
    }
    // tail
    for (; j < m; ++j)
    {
      C[i][j] += A[i][j] + B[0][j];
    }
  }
  return C;
}

IntervalMatrix relu(const IntervalMatrix &A)
{
  size_t n = A.size(), m = A[0].size();
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < m; ++j)
    {
      auto a = A[i][j];

      auto lb = a.lower() > 0 ? a.lower() : 0.0;
      auto ub = a.upper() > 0 ? a.upper() : 0.0;
      C[i][j] = Interval(lb, ub);
    }
  }
  return C;
}

IntervalMatrix reluTwo(const IntervalMatrix &X, const IntervalMatrix &Y)
{
  size_t x_rows = X.size(), x_cols = X[0].size();
  size_t y_rows = Y.size(), y_cols = Y[0].size();
  if (x_rows == 0 || y_rows == 0) // checking empty matrices
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  IntervalMatrix C(x_rows, std::vector<Interval>(x_cols, Interval(0.0, 0.0)));

  // Helper function to safe repetetive code/actions
  auto relu_helper = [](const Interval &a, const Interval &b)
  {
    auto lba = std::max(0.0, a.lower());
    auto uba = std::max(0.0, a.upper());
    auto lbb = std::max(0.0, b.lower());
    auto ubb = std::max(0.0, b.upper());
    return Interval(std::max(lba, lbb), std::max(uba, ubb));
  };

  // Case 1: Y is an 1x1 scalar
  if (y_rows == 1 && y_cols == 1)
  { // scalar
    auto b = Y[0][0];
    for (size_t i = 0; i < x_rows; i++)
      for (size_t j = 0; j < x_cols; j++)
        C[i][j] = relu_helper(X[i][j], b);
    return C;
  }
  // Case 2: Y is a vector
  else if (y_rows == 1 && y_cols == x_cols) // broadcasting row vector
  {
    for (size_t i = 0; i < x_rows; i++)
      for (size_t j = 0; j < x_cols; j++)
        C[i][j] = relu_helper(X[i][j], Y[0][j]); // always accessing Y[0] (vector)
  }
  else if (x_rows == y_rows && x_cols == y_cols)
  {
    for (size_t i = 0; i < x_rows; ++i)
      for (size_t j = 0; j < x_cols; ++j)
        C[i][j] = relu_helper(X[i][j], Y[i][j]);
  }
  else
  {
    throw std::invalid_argument("Incompatable shapes for reluTwo().");
  }

  return C;
}

IntervalMatrix reluTwo(const IntervalMatrix &X, const Interval &y)
{
  return reluTwo(X, IntervalMatrix(1, std::vector<Interval>(1, y)));
}

IntervalMatrix sigmoid(IntervalMatrix &A)
{
  size_t n = A.size(), m = A[0].size();
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  for (size_t i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < m; ++j)
    {
      double lower = 1.0 / (1.0 + std::exp(-A[i][j].lower()));
      double upper = 1.0 / (1.0 + std::exp(-A[i][j].upper()));
      A[i][j] = Interval(std::min(lower, upper), std::max(lower, upper)); // ensuring smaller value is lower bound and vice versa
    }
  }
  return A;
}

IntervalMatrix conv2D(const IntervalMatrix &input, const IntervalMatrix &kernel, int stride = 1, int padding = 0)
{

  if (input.empty() || kernel.empty() || input[0].empty() || kernel[0].empty())
  {
    throw std::invalid_argument("Either input or kernel matrix is empty.");
  }

  if (stride < 1 || padding < 0)
  {
    throw std::invalid_argument("Stride must be > 0 and padding >= 0.");
  }

  // Getting dimensions
  size_t input_rows = input.size();
  size_t input_cols = input[0].size();
  size_t kernel_rows = kernel.size();
  size_t kernel_cols = kernel[0].size();

  // Calculating output dimensions
  size_t output_rows = (input_rows - kernel_rows + 2 * padding) / stride + 1;
  size_t output_cols = (input_cols - kernel_cols + 2 * padding) / stride + 1;

  if (output_cols < 1 || output_rows < 1)
  {
    std::cout << "Output columns: " << output_cols << " Output rows: " << output_rows << std::endl;
    throw std::invalid_argument("Kernel size larger than input matrix.");
  }

  // Init output matrix with 0's
  IntervalMatrix outputMatrix(output_rows, std::vector<Interval>(output_cols, Interval(0.0, 0.0)));
  Interval zero(0.0, 0.0); // For padding
  Interval sum;
  // Main convolution loop
  for (size_t i = 0; i < output_rows; ++i)
  {
    for (size_t j = 0; j < output_cols; ++j)
    {
      sum = Interval(0.0, 0.0);

      for (size_t k_i = 0; k_i < kernel_rows; ++k_i)
      {
        for (size_t k_j = 0; k_j < kernel_cols; ++k_j)
        {
          int in_i = i * stride + k_i - padding;
          int in_j = j * stride + k_j - padding;

          if (in_i >= 0 && in_i < input_rows && in_j >= 0 && in_j < input_cols)
          {
            sum += input[in_i][in_j] * kernel[k_i][k_j];
          }
          else
          {
            sum += zero * kernel[k_i][k_j]; // Padding with 0's, because out of bounds
          }
        }
      }
      outputMatrix[i][j] = sum;
    }
  }
  return outputMatrix;
}

Interval toIntervalVal(const double x)
{
  return Interval(x, x);
}

Interval toIntervalValRange(const double x, const double range)
{
  return Interval(x - range, x + range);
}

// Checks Matrix A if values are within Intervals of Matrix B
bool checkValid(const IntervalMatrix &A, const IntervalMatrix &B)
{ // A should be within B

  size_t n = A.size(), m = A[0].size();
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < m; j++)
    {
      if (i < 20 && j < 20)
      {
        std::cout << "Checking A: [" << A[i][j].lower() << ", " << A[i][j].upper() << "] within B: [" << B[i][j].lower() << ", " << B[i][j].upper() << "]" << std::endl;
      }

      if (!(A[i][j].lower() >= B[i][j].lower() && A[i][j].upper() <= B[i][j].upper()))
      {
        std::cout << "A not within B at: " << i << ", " << j << " A: [" << A[i][j].lower() << ", " << A[i][j].upper() << "] B: [" << B[i][j].lower() << ", " << B[i][j].upper() << "]" << std::endl;
        return false;
      }
    }
  }
  std::cout << "A within B" << std::endl;
  return true;
}

IntervalMatrix transposeIntervalMatrix(const IntervalMatrix &M)
{
  size_t n = M.size(), m = M[0].size();

  if (n == 0 || m == 0 || n != m)
  {
    throw std::invalid_argument("Wrong Matrix dimension, either rows != cols, or empty.");
  }
  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));

  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < m; j++)
    {
      C[j][i] = M[i][j];
    }
  }
  return C;
}

NB_MODULE(ffi_module, m)
{
  nb::class_<Interval>(m, "Interval")
      .def(nb::init<double, double>())
      .def("lower", [](const Interval &a)
           { return a.lower(); })
      .def("upper", [](const Interval &a)
           { return a.upper(); })
      .def("__repr__", [](const Interval &i)
           { return "<Interval [" + std::to_string(i.lower()) + ", " + std::to_string(i.upper()) + "]>"; });

  nb::class_<IntervalMatrix>(m, "IntervalMatrix")
      .def(nb::init<>()) // default constructor
      .def("__init__", [](IntervalMatrix *self, nb::list rows)
           {
        new (self) IntervalMatrix();
        for (auto row : rows) {
            std::vector<Interval> r;
            for (auto val : nb::cast<nb::list>(row))
                r.push_back(nb::cast<Interval>(val));
            self->push_back(r);
        } })
      .def("__getitem__", [](const IntervalMatrix &mat, size_t i)
           { return mat[i]; })
      .def("__setitem__", [](IntervalMatrix &mat, size_t i, const std::vector<Interval> &row)
           { mat[i] = row; })
      .def("__repr__", [](const IntervalMatrix &mat)
           {
        std::string s = "IntervalMatrix([\n";
        for (const auto& row : mat) {
            s += "  [";
            for (const auto& val : row)
                s += "<" + std::to_string(val.lower()) + "," + std::to_string(val.upper()) + ">, ";
            s += "]\n";
        }
        s += "])";
        return s; });

  m.def("rows", [](const IntervalMatrix &M)
        { return (int)M.size(); });
  m.def("cols", [](const IntervalMatrix &M)
        { return M.empty() ? 0 : (int)M[0].size(); });
  m.def("get", [](const IntervalMatrix &M, size_t i, size_t j) -> Interval
        { return M.at(i).at(j); });

  m.def("add", &add, "Add two intervals");
  m.def("mult", &mult, "Multiplies two numbers"); //, nb::arg("a"), nb::arg("b"));
  m.def("relu", &relu, "Relu function for an interval");
  // m.def("reluTwo", &reluTwo, "Relu function for an intervalMatrix x (intervalMatrix/vector/scalar)"),
  // nb::arg("x"), nb::arg("y");
  m.def("reluTwo", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const IntervalMatrix &)>(&reluTwo));
  m.def("reluTwo", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const Interval &)>(&reluTwo));
  m.def("sigmoid", &sigmoid, "Sigmoid function for an interval");
  m.def("conv2D", &conv2D, "Convolutional 2D function for an interval matrix",
        nb::arg("input"), nb::arg("kernel"), nb::arg("stride") = 1, nb::arg("padding") = 0);
  m.def("toIntervalVal", &toIntervalVal, "Converts double to interval");
  m.def("toIntervalValRange", &toIntervalValRange, "Converts double to interval, with second range input");
  m.def("matrixMult", &matrixMult, "Multiplies two matrices together");          //, nb::arg("a"), nb::arg("b"));
  m.def("matrixAddBias", &matrixAddBias, "Adds Bias to matrix");                 // arg::Matrix, arg::vector (bias)
  m.def("checkValid", &checkValid, "Checks if first Interval is within second"); // arg::Matrix, arg::vector (bias)
}