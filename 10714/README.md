# CMU: 10-714: Deep Learning Systems

The subfolder contains all the homework and assignmend from [10-714](https://dlsyscourse.org/), the course builds a `pytorch` like framework from scratch 
on CPU and CUDA and implements fundational models such as CNN, RNN, LSTM on top of it.

The course offers a balance practice for engineering and mathmatical theory.

## What's extra?

- In `hw3`
  - Experimented with [triton](https://github.com/openai/triton) as a backend for `ndarray`

- In `hw4`
  - Implemented coordinated fetching for `matmul` in the CUDA layer, [source](./hw4/src/ndarray_backend_cuda.cu#L410)
  - Added alternative activation functions, [source](./hw4/python/needle/ops.py#L407)
  - [Implemented](https://github.com/yangchenyun/ml_journey/commit/f7a5aacfda5a04e98fb783d82597e84e51f089d0) `convTranspose` operation, 
	  - passed forward tests for padding / strides
	  - passed backward for unit strides (non-unit strides have shape issues)