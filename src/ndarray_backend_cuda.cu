#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t *ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(std::min(static_cast<int>(size), BASE_THREAD_NUM), 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE)
    throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from
// strides

// Calculate indices based on gid, strides and shape
__device__ size_t CompactIndexOffset(size_t gid, CudaVec shape, CudaVec strides) {
  assert(shape.size == strides.size);
  size_t d = shape.size;
  size_t remaining_indices = gid;
  size_t offset = 0;
  for (size_t di = 0; di < d; di++)
  {
    size_t step_size = 1;
    for (size_t j = di + 1; j < d; j++) {
      step_size *= shape.data[j];
    }
    auto idx = (remaining_indices / step_size);
    // Step 1: track offset for current dimension with index
    // Step 2: subtract the total number of items has accounted in current dimension
    offset += idx * strides.data[di];
    remaining_indices -= idx * step_size;
  }
  assert(remaining_indices == 0);
  return offset;
}

__device__ void CompactProcess(const scalar_t *a, scalar_t *out, size_t stride_i, size_t compact_i) {
  out[compact_i] = a[stride_i];
}
__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size,
                              CudaVec shape, CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a
   * single entry in the non-compact input a, to the corresponding item (at
   * location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past
   * passing to CUDA kernel) 
   *   strides: vector of strides of  array offset
   *   offset: of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  /// BEGIN YOUR SOLUTION
  if (gid >= size) { return; }
  offset += CompactIndexOffset(gid, shape, strides);
  CompactProcess(a, out, offset, gid);
  /// END YOUR SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will
   * primarily call the relevant CUDA kernel.  In this case, we illustrate how
   * you should set this up (i.e., we give you the code for this fuction, and
   * also the prototype for the CompactKernel() function).  For the functions
   * after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being
   * compact)
   */
  

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  // NOTE: Syntax <<<>>> is used to launch a kernel function on the GPU, specifying grid and block.
  // Here , each grid and block is a 1 d-array; work is paralleled with 256 thread per block.
  CompactKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);
}

__device__ void EwiseSetitemProcess(const scalar_t *a, scalar_t *out, size_t stride_i, size_t compact_i) {
  out[stride_i] = a[compact_i];
}
__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size,
                              CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  /// BEGIN YOUR SOLUTION
  // NOTE: skip extra threads more than size
  if(gid >= size) { return; }
  offset += CompactIndexOffset(gid, shape, strides);
  EwiseSetitemProcess(a, out, offset, gid);
  /// END YOUR SOLUTION
}

void EwiseSetitem(const CudaArray &a, CudaArray *out,
                  std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                  size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want
   * to implement a EwiseSetitemKernel() function, similar to those above, that
   * will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being
   * compact)
   */
  /// BEGIN YOUR SOLUTION
  // NOTE: Launch kernel according to the size to be written.
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

__device__ void ScalarSetitemProcess(const scalar_t val, scalar_t *out, size_t stride_i, size_t compact_i) {
  out[stride_i] = val;
}

__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t *out, size_t size,
                              CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  /// BEGIN YOUR SOLUTION
  if(gid >= size) { return; }
  offset += CompactIndexOffset(gid, shape, strides);
  ScalarSetitemProcess(val, out, offset, gid);
  /// END YOUR SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out,
                   std::vector<uint32_t> shape, std::vector<uint32_t> strides,
                   size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note
   * be the same as out.size, because out is a non-compact subset array);  it
   * _will_ be the same as the product of items in shape, but covenient to just
   * pass it here. val: scalar value to write to out: non-compact array whose
   * items are to be written shape: shapes of each dimension of out strides:
   * strides of the out array offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  // Launch with number of elements to write
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(
      val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

/**
 * In the code the follows, use the above template to create analogous
 * elementise and and scalar operators for the following functions.  See the
 * numpy backend for examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define
 * these functions (however you want to do so, as long as the functions match
 * the proper) signatures above.
 */

/// BEGIN YOUR SOLUTION
#define SCALAR_BINARY_OP(OPNAME, OP) \
__global__ void Scalar##OPNAME##Kernel(const scalar_t *a, scalar_t val, scalar_t *out, \
                                size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) \
    out[gid] = a[gid] OP val; \
} \
void Scalar##OPNAME(const CudaArray &a, scalar_t val, CudaArray *out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##OPNAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}
#define SCALAR_BINARY_OP_FN(OPNAME, OP_FN) \
__global__ void Scalar##OPNAME##Kernel(const scalar_t *a, scalar_t val, scalar_t *out, \
                                size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) \
    out[gid] = OP_FN(a[gid], val); \
} \
void Scalar##OPNAME(const CudaArray &a, scalar_t val, CudaArray *out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##OPNAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define EWISE_BINARY_OP(OPNAME, OP) \
__global__ void Ewise##OPNAME##Kernel(const scalar_t *a, const scalar_t *b, \
                               scalar_t *out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) \
    out[gid] = a[gid] OP b[gid]; \
}; \
void Ewise##OPNAME(const CudaArray &a, const CudaArray &b, CudaArray *out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##OPNAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}
#define EWISE_UNARY_OP(OPNAME, OP) \
__global__ void Ewise##OPNAME##Kernel(const scalar_t *a, \
                               scalar_t *out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) \
    out[gid] = OP(a[gid]); \
}; \
void Ewise##OPNAME(const CudaArray &a, CudaArray *out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##OPNAME##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

EWISE_BINARY_OP(Add, +)
EWISE_BINARY_OP(Mul, *)
EWISE_BINARY_OP(Div, /)
EWISE_BINARY_OP(Eq, ==)
EWISE_BINARY_OP(Ge, >=)
EWISE_UNARY_OP(Log, log)
EWISE_UNARY_OP(Exp, exp)
EWISE_UNARY_OP(Tanh, tanh)
__global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = max(a[gid], b[gid]);
};

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

SCALAR_BINARY_OP(Add, +)
SCALAR_BINARY_OP(Mul, *)
SCALAR_BINARY_OP(Div, /)
SCALAR_BINARY_OP(Eq, ==)
SCALAR_BINARY_OP(Ge, >=)
SCALAR_BINARY_OP_FN(Maximum, max)
SCALAR_BINARY_OP_FN(Power, pow)
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M,
            uint32_t N, uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You
   * will want to look at the lecture and notes on GPU-based linear algebra to
   * see how to do this.  Since ultimately mugrade is just evaluating
   * correctness, you _can_ implement a version that simply parallelizes over
   * (i,j) entries in the output array.  However, to really get the full benefit
   * of this problem, we would encourage you to use cooperative fetching, shared
   * memory register tiling, and other ideas covered in the class notes.  Note
   * that unlike the tiled matmul function in the CPU backend, here you should
   * implement a single function that works across all size matrices, whether or
   * not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel
   * call, and you should implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION

  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though
   * it is inefficient, for simplicity you can perform each reduction in a
   * single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION

  /// END YOUR SOLUTION
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION

  /// END YOUR SOLUTION
}

} // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(),
                   numpy_strides.begin(),
                   [](size_t &c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0)
      throw std::bad_alloc();
    cudaError_t err =
        cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset,
                                 deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err = cudaMemcpy(out->ptr, a.request().ptr,
                                 out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
