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
IntervalMatrix matrixDiv(const IntervalMatrix &A, const Interval &y)
{
  if (A.empty() || A[0].empty())
  {
    throw std::invalid_argument("Input matrices cannot be empty.");
  }

  size_t n = A.size(), m = A[0].size();
  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      C[i][j] += A[i][j] / y;
  return C;
}

IntervalMatrix matrixMultElementwise(const IntervalMatrix &A, const IntervalMatrix &B)
{
  size_t a_rows = A.size(), a_cols = A[0].size();
  size_t b_rows = B.size(), b_cols = B[0].size();
  if (a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input contains at least one empty matrix.");
  }
  if (a_cols != b_cols)
  {
    throw std::invalid_argument("Matrices must have same column numbers");
  }
  if (b_rows > a_rows)
  {
    throw std::invalid_argument("Matrix b must have equal or less rows than matrix a");
  }

  IntervalMatrix C(a_rows, std::vector<Interval>(a_cols, Interval(0.0, 0.0)));

  constexpr size_t VF = 4; // or 8, 16 depending on your arch
  for (size_t i = 0; i < a_rows; ++i)
  {
    size_t j = 0, b_i = i % b_rows;
    // process full blocks
    for (; j + VF <= a_cols; j += VF)
    {
      // Load blocks
      for (size_t k = 0; k < VF; ++k)
      {
        C[i][j + k] += A[i][j + k] * B[b_i][j + k];
      }
    }
    // tail
    for (; j < a_cols; ++j)
    {
      C[i][j] += A[i][j] * B[b_i][j];
    }
  }
  return C;
}
IntervalMatrix matrixMultElementwise(const IntervalMatrix &A, const Interval &x)
{
  size_t a_rows = A.size(), a_cols = A[0].size();
  if (a_rows == 0 || a_cols == 0) // checking empty matrix
  {
    throw std::invalid_argument("Matrix A is empty.");
  }

  IntervalMatrix C(a_rows, std::vector<Interval>(a_cols, Interval(0.0, 0.0)));

  constexpr size_t VF = 4; // or 8, 16 depending on your arch
  for (size_t i = 0; i < a_rows; ++i)
  {
    size_t j = 0;
    // process full blocks
    for (; j + VF <= a_cols; j += VF)
    {
      // Load blocks
      for (size_t k = 0; k < VF; ++k)
      {
        C[i][j + k] += A[i][j + k] * x;
      }
    }
    // tail
    for (; j < a_cols; ++j)
    {
      C[i][j] += A[i][j] * x;
    }
  }
  return C;
}

IntervalMatrix matrixPow(const IntervalMatrix &X, const int &y)
{
  size_t n = X.size(), m = X[0].size();
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix has wrong size.");
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));

  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < m; j++)
    {
      // var = pow(matrixentry, exponent)
      double pow_lb = std::pow(X[i][j].lower(), y);
      double pow_ub = std::pow(X[i][j].upper(), y);

      // determine min and max for correct interval structure
      double x_min = std::min(pow_lb, pow_ub);
      double x_max = std::max(pow_lb, pow_ub);

      C[i][j] = Interval(x_min, x_max);
    }
  }
  return C;
}

IntervalMatrix matrixAbs(const IntervalMatrix &X)
{
  size_t n = X.size(), m = X[0].size();
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix has wrong size.");
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));

  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < m; j++)
    {
      const double abs_x_lb = std::abs(X[i][j].lower());
      const double abs_x_ub = std::abs(X[i][j].upper());

      C[i][j] = Interval(std::min(abs_x_lb, abs_x_ub), std::max(abs_x_lb, abs_x_ub));
    }
  }
  return C;
}
Interval matrixAbs(const Interval &x)
{
  double abs_x_lb = std::abs(x.lower());
  double abs_x_ub = std::abs(x.upper());

  return Interval(std::min(abs_x_lb, abs_x_ub), std::max(abs_x_lb, abs_x_ub));
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
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix is empty.");
  }
  if (B.size() == 0) // bias must not have 0 rows
  {
    throw std::invalid_argument("Bias has no rows");
  }
  if (B[0].size() != m) // bias must be length of matrix row
  {
    throw std::invalid_argument("Bias must be length of matrix row");
  }
  if (B.size() > 1)
  {
    if (n == B.size() && m == B[0].size())
    {
      return elementwiseAdd(A, B);
    }
    else
    {
      throw std::invalid_argument("Bias has wrong shape for elementwise addition and is not 1 dimensional for bias addition.");
    }
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));

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
IntervalMatrix matrixAddBias(const IntervalMatrix &A, const Interval &x)
{
  size_t n = A.size(), m = A[0].size();
  if (n == 0 || m == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  IntervalMatrix C(n, std::vector<Interval>(m, Interval(0.0, 0.0)));

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
        C[i][j + k] += A[i][j + k] + x;
      }
    }
    // tail
    for (; j < m; ++j)
    {
      C[i][j] += A[i][j] + x;
    }
  }
  return C;
}

IntervalMatrix matrixMatrixSub(const IntervalMatrix &A, const IntervalMatrix &B)
{
  size_t a_rows = A.size(), a_cols = A[0].size();
  size_t b_rows = B.size(), b_cols = B[0].size();
  if (a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0) // checking empty matrix
  {
    throw std::invalid_argument("Input contains at least one empty matrix.");
  }
  if (a_cols != b_cols)
  {
    throw std::invalid_argument("Matrices must have same column numbers");
  }
  if (b_rows > a_rows)
  {
    throw std::invalid_argument("Matrix b must have equal or less rows than matrix a");
  }

  IntervalMatrix C(a_rows, std::vector<Interval>(a_cols, Interval(0.0, 0.0)));

  constexpr size_t VF = 4; // or 8, 16 depending on your arch
  for (size_t i = 0; i < a_rows; ++i)
  {
    size_t j = 0, b_i = i % b_rows;
    // process full blocks
    for (; j + VF <= a_cols; j += VF)
    {
      // Load blocks
      for (size_t k = 0; k < VF; ++k)
      {
        C[i][j + k] += A[i][j + k] - B[b_i][j + k];
      }
    }
    // tail
    for (; j < a_cols; ++j)
    {
      C[i][j] += A[i][j] - B[b_i][j];
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

IntervalMatrix max(const IntervalMatrix &X, const IntervalMatrix &Y)
{
  size_t x_rows = X.size(), x_cols = X[0].size();
  size_t y_rows = Y.size(), y_cols = Y[0].size();
  size_t rows = std::max(x_rows, y_rows);
  size_t cols = std::max(x_cols, y_cols);
  if (x_rows == 0 || y_rows == 0) // checking empty matrices
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  IntervalMatrix C(rows, std::vector<Interval>(cols, Interval(0.0, 0.0)));

  // Helper function to safe repetetive code/actions
  auto relu_helper = [](const Interval &a, const Interval &b)
  {
    auto lb = std::max(a.lower(), b.lower());
    auto ub = std::max(a.upper(), b.upper());
    return Interval(lb, ub);
  };

  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      C[i][j] = relu_helper(X[i % x_rows][j % x_cols], Y[i % y_rows][j % y_cols]);

  return C;
}

IntervalMatrix max(const IntervalMatrix &X, const Interval &y)
{
  return max(X, IntervalMatrix(1, std::vector<Interval>(1, y)));
}

IntervalMatrix min(const IntervalMatrix &X, const IntervalMatrix &Y)
{
  size_t x_rows = X.size(), x_cols = X[0].size();
  size_t y_rows = Y.size(), y_cols = Y[0].size();
  size_t rows = std::max(x_rows, y_rows);
  size_t cols = std::max(x_cols, y_cols);
  if (x_rows == 0 || y_rows == 0) // checking empty matrices
  {
    throw std::invalid_argument("Input matrix is empty.");
  }

  IntervalMatrix C(rows, std::vector<Interval>(cols, Interval(0.0, 0.0)));

  // Helper function to safe repetetive code/actions
  auto relu_helper = [](const Interval &a, const Interval &b)
  {
    auto lb = std::min(a.lower(), b.lower());
    auto ub = std::min(a.upper(), b.upper());
    return Interval(lb, ub);
  };

  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      C[i][j] = relu_helper(X[i % x_rows][j % x_cols], Y[i % y_rows][j % y_cols]);

  return C;
}

IntervalMatrix min(const IntervalMatrix &X, const Interval &y)
{
  return min(X, IntervalMatrix(1, std::vector<Interval>(1, y)));
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

  if (n == 0 || m == 0)
  {
    throw std::invalid_argument("Wrong Matrix dimension, either rows or cols is empty.");
  }
  IntervalMatrix C(m, std::vector<Interval>(n, Interval(0.0, 0.0)));

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
  // m.def("max", &max, "Relu function for an intervalMatrix x (intervalMatrix/vector/scalar)"),
  // nb::arg("x"), nb::arg("y");
  m.def("max", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const IntervalMatrix &)>(&max));
  m.def("max", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const Interval &)>(&max));
  m.def("min", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const IntervalMatrix &)>(&min));
  m.def("min", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const Interval &)>(&min));
  m.def("sigmoid", &sigmoid, "Sigmoid function for an interval");
  m.def("conv2D", &conv2D, "Convolutional 2D function for an interval matrix",
        nb::arg("input"), nb::arg("kernel"), nb::arg("stride") = 1, nb::arg("padding") = 0);
  m.def("toIntervalVal", &toIntervalVal, "Converts double to interval");
  m.def("toIntervalValRange", &toIntervalValRange, "Converts double to interval, with second range input");
  m.def("matrixMult", &matrixMult, "Multiplies two matrices together"); //, nb::arg("a"), nb::arg("b"));
  m.def("matrixDiv", &matrixDiv, "Divides two matrices by each other (elementwise division)");
  m.def("matrixAddBias", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const IntervalMatrix &)>(&matrixAddBias), "Adds Bias to matrix"); // arg::Matrix, arg::vector (bias)
  m.def("matrixAddBias", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const Interval &)>(&matrixAddBias), "Adds Bias to matrix");       // arg::Matrix, arg::vector (bias)
  m.def("matrixMultElementwise", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const IntervalMatrix &)>(&matrixMultElementwise));
  m.def("matrixMultElementwise", static_cast<IntervalMatrix (*)(const IntervalMatrix &, const Interval &)>(&matrixMultElementwise));
  m.def("matrixPow", &matrixPow, "Matrix power function, takes matrix and single value");
  m.def("matrixAbs", static_cast<IntervalMatrix (*)(const IntervalMatrix &)>(&matrixAbs));
  m.def("matrixAbs", static_cast<Interval (*)(const Interval &)>(&matrixAbs));
  m.def("matrixMatrixSub", &matrixMatrixSub, "Subtracting function for two matrices");
  m.def("checkValid", &checkValid, "Checks if first Interval is within second");
  m.def("transposeIntervalMatrix", &transposeIntervalMatrix, "Transpose matrix");
}