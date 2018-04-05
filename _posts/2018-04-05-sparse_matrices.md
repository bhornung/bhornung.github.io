---
layout: post
title:  "Sparse matrices"
date:   2018-04-05 13:36:13 +0000
categories: python
---

## Aim

In this post we write fast python functions to generate sparse block matrices.

A sparse matrix is a matrix where most of its entries are zero. We will restrict our attention to block sparse matrices where there ne nonzero (_nnz_) entries are accumulated in blocks. A sample matrix looks like:

$\begin{bmatrix}
    a & a & a & 0  &   & \cdots  &  & 0 \\
    a & a & a & 0  &   & \cdots &  & 0 \\
    a & a & a & 0  &   & \cdots  &  & 0 \\
    0 & 0 & 0 & b & b & 0 & \cdots & 0 \\
    0 & 0 & 0 & b & b & 0 & \cdots & 0 \\
    0 & 0 & 0 & 0 & \cdots & 0 & 0 & 0 \\    
    \vdots   &   &   & \vdots & & & & \vdots 
\end{bmatrix}$

The term large means the matrices have about $10^{4--7}$ columns. 

## Motivation

In the context of Markov cluster algorithm, the clusters manifest themselves as identical rows of the transition matrix. Depending on the graph and cluster size, these rows can be as large as million with only a few thousand _nnz_ elements in them.

## Initial considerations

### Representation of sparse matrices

Before it is decided which paths will explored, it is important do establish what data structure will be used. The connectivity and the related transition matrices are sparse, that is the number of nonzero elements is much smaller than the maximum possible number of elements $N^{2}$. These matrices can be stored as sparse matrices; predominantly in **compressed sparse row [(csr)](http://www.scipy-lectures.org/advanced/scipy_sparse/index.html) format**. 

A csr matrix consists of three arrays. One that stores the nonzero elements of the matrix, one that stores the column indices and a third one that keeps track of how many nnz entries are in a row (strictly speaking, it stores the cumulative number of _nnz_ entries for each row). These are called `sparse_matrix.data`, `sparse_matrix.inidces`, `sparse_matrix.indptr` in the `scipy.sparse` parlance.

Our hypothetical matrix from the first paragraph would look like
```
sparse_matrix.data = [a, a, a, b, b, ...]
sparse_matrix.indices = [0, 1, 2, 0, 1, ...]
sparse_matrix.indptr = [0, 3, 5, 5, 5, ....]
```

### Coding principles

1. **Ease of coding:** Python has a plethora sparse matrix utilities, such as [networkx](https://networkx.github.io/), [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) and numpy. We will try to use these as much as we can.

1. **Efficiency:** After all, time is money, thus we should aim to write fast functions. This implies

    1. low level data manipulation
    1. usage of compiled program components
    1. avoiding of copying data excessively

## Solutions

### 1. `scipy.sparse.bmat`

`scipy.sparse` provides the `block_diag` class that creates a sparse block diagonal matrix from a list of (sparse) matrices. The task therefore can be broken down to two stages:

**Algorithm 1.1**

$
\begin{eqnarray}
   &1.& \text{create list of blocks as sparse matrices} \\
   &2.& \text{concatenate sparse matrices}
\end{eqnarray}
$

We will generate the list of blocks in two ways

1. as `lists`
1. as `generators`  in order to save memory.

The following function, `make_full_csr_matrix`, creates a block in csr format:


```python
import numpy as np
import scipy.sparse as sps
```


```python
def make_full_csr_matrix(shape, fill_value):
  """
  Creates a full csr matrix.
  Parameters:
    shape ((int,int)) : shape of the matrix
    fill_value (int) : matrix elements are set ot this value
  Returns:
    (scipy.sparse.csr_matrix()) : sparse matrix filled with the specified value
  """
# pass shape parameters
  nrow, ncol = shape
# create uniform data of the correct number
  data = np.full((nrow * ncol), fill_value = 1, dtype = np.int)
# column indices
  indices = np.tile(np.arange(ncol), nrow)
# number of nonzero elements in the rows
  indptr = np.arange(nrow + 1, dtype = np.int) * ncol
# create matrix
  return sps.csr_matrix((data, indices, indptr))
```

These blocks can conveniently be collected in a list:


```python
def make_full_csr_matrix_list(shapes, fill_value):
  """
  Generates a list of csr sparse matrices. The matrices are full, but in sparse format, 
  so that they can easily be processed by sparse matrix constructors.
  Parameters:
    shapes (sequence of tuples) : the sequence of shapes
    fill_value (int) : all matrix elements will have this value
  Returns:
    matrices [scipy.sparse.csr_matrix]: list of csr matrices
  """
  matrices = []

# --- iterate through shapes
  for shape in shapes:
    matrices.append(make_full_csr_matrix(shape, fill_value))

  return matrices
```

However, storing every block in a list increases the memory requirement of our program. It is therefore advantageous to use a generator instead. This is implemented as a class called `FullCsrMatrixFactory`.


```python
class FullCsrMatrixFactory(object):
  """
  Creates an instance of a generator object which generates a sequence of csr matrices.
  The matrices are full, but in sparse format, hence they can easily be processed by sparse matrix constructors.
  Attributes:
    shapes ([(,)]) : list of tuples containing the shapes of the matrices
    fill_value (int) : each element will be set to this value
  """
# -------
  def __init__(self, shapes, fill_value):
    """
    shapes ([(,)]) : list of tuples containing the shapes of the matrices
    fill_value (int) : each element will be set to this value
    """
# set shapes of matrices
    try:
      _ = len(shapes)
    except Exception as err:
      print("parameter 'shapes' must implement the '__len__()' method")
      raise err

    self._shapes = shapes

# set value for elements
    self._fill_value = fill_value

# set counter to zero
    self.__ispent = 0

  @property
  def fill_value(self):
    return self._fill_value

  @property
  def shapes(self):
    return self._shapes

  @property
  def ispent(self):
    return self.__ispent

# -------
  def __iter__(self):
    """
    Returns a sparse matrix
    """
    while self.ispent < len(self):
    
      _shape = self.shapes[self.ispent]
      yield make_full_csr_matrix(_shape, self.fill_value)
      self.__ispent += 1

# -------
  def __len__(self):
    """
    Define a len method, so that the number of matrices can be known in advance.
    """
    return len(self.shapes)
```

Unlike usual generators, ours implements the `__len__()` method which is required by the [`sps.block_diag`](https://github.com/scipy/scipy/blob/v1.0.0/scipy/sparse/construct.py#L625-L676) constructor. The other minor quirks are `shape` and `fill_value` fields being protected. 

These factory functions are then wrapped in the `create_block_diagonal_csr_matrix_sps` function, which returns our desired block sparse matrix. The `keep_density` parameter controls what ratio of the rows are kept. `keep_each_block` setting decides whether at least one row in each block should be retained or not.


```python
def create_block_diagonal_csr_matrix_sps(block_sizes, factory,
                                         fill_value = 1, 
                                         keep_density = 0.1,
                                         keep_each_block = True):
  """
  Creates a block diagonal csr matrix.
  Parameters:
    block_sizes ([int]) : sizes of the blocks.
    factory (callable) : should generate a list of csr matrices.
      signature ~([(int,int)], float) --> iter(scipy.sparse.csr_matrix)

    fill_value (int) : the value of the elements. Default 1.
    keep_density (float) : the proportion of rows to be kept. Default 0.01.
    keep_each_block (bool) : whether to keep at least one row from each block. Default True
  """
# number of rows to be kept
  block_sizes_ = np.array(block_sizes, dtype = np.int)
  n_keep_rows = np.rint(block_sizes_ * keep_density).astype(np.int)
# keep one row from each block all blocks
  if keep_each_block:
    n_keep_rows[n_keep_rows == 0] = 1
# create shapes for blocks with more than zero rows
  shapes = list(filter(lambda x: x[0] > 0, zip(n_keep_rows, block_sizes_)))
# set up generator for the sequence of matrices
  mats = factory(shapes, fill_value)

# create a blockdiagonal matrix by concatenating the blocks
  adj_mat = sps.block_diag(mats, format = 'csr')

  return adj_mat
```

### Testing

After all these heavy lines, it is time to decide which is the fastest solution on a range of blocks varying in number and sizes. I have a natural aversion from `timeit`, thus we are going to use a crude timing method.


```python
block_size = [10, 50, 100, 500]
shape_list = [[x] * x for x in block_size]
keep_density = 0.1
```


```python
from time import perf_counter
performance = []

for shapes in shape_list:
    tstart = perf_counter()
    matrix = create_block_diagonal_csr_matrix_sps(shapes, make_full_csr_matrix_list,
                                                  keep_density = keep_density)
    tdelta = perf_counter() - tstart
    res_dict = {'matrix_size' : matrix.size}
    res_dict.update({'time' : tdelta})
    performance.append(res_dict)
    
for res_dict in performance:
   print("\t".join(["{0} : {1}".format(_k, _v) for _k, _v in res_dict.items()]))
```

    matrix_size : 100	time : 0.6626071659982244
    matrix_size : 12500	time : 0.010648324167505052
    matrix_size : 100000	time : 0.028556179705105933
    matrix_size : 12500000	time : 1.6887429069183781
    

Inspecting the timings shows that creating a matrix with $500^{3} \cdot 0.1 = 1.25 \cdot 10^{7}$ elements takes around 2 seconds. Can the generator do better? Let us see!


```python
performance = []

for shapes in shape_list:
    tstart = perf_counter()
    matrix = create_block_diagonal_csr_matrix_sps(shapes, FullCsrMatrixFactory,
                                                  keep_density = keep_density)
    tdelta = perf_counter() - tstart
    res_dict = {'matrix_size' : matrix.size}
    res_dict.update({'time' : tdelta})
    performance.append(res_dict)
    
for res_dict in performance:
   print("\t".join(["{0} : {1}".format(_k, _v) for _k, _v in res_dict.items()]))
```

    matrix_size : 100	time : 0.015199141779476122
    matrix_size : 12500	time : 0.013311074578496118
    matrix_size : 100000	time : 0.0360058119090354
    matrix_size : 12500000	time : 1.6373866758058284
    

The timings are similar. Are the algorithms fast or slow? I would argue they are somewhat sluggish, for creating a numpy array of the same size is about hundred times faster.


```python
% timeit -n 3 np.full((500*500*50), 1)
```

    3 loops, best of 3: 21.1 ms per loop
    

Why are they so slow then? There are two possibilities. Either the same step takes the longest time in both cases, or we gain and loose speed at different steps. In order to figure it out, we are going to profile these functions.


```python
%prun -l 15 -s "cumulative" create_block_diagonal_csr_matrix_sps(shape_list[-1], make_full_csr_matrix_list, keep_density = keep_density)
```

             110240 function calls (109740 primitive calls) in 1.692 seconds
    
       Ordered by: cumulative time
       List reduced from 88 to 15 due to restriction <15>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    1.692    1.692 {built-in method builtins.exec}
            1    0.024    0.024    1.692    1.692 <string>:1(<module>)
            1    0.001    0.001    1.668    1.668 <ipython-input-63-336c75f19800>:1(create_block_diagonal_csr_matrix_sps)
            1    0.037    0.037    1.559    1.559 construct.py:617(block_diag)
            1    0.133    0.133    1.522    1.522 construct.py:501(bmat)
            1    0.000    0.000    1.110    1.110 base.py:236(asformat)
            1    0.000    0.000    1.110    1.110 coo.py:301(tocsr)
            1    0.027    0.027    1.030    1.030 coo.py:449(sum_duplicates)
            1    0.220    0.220    1.003    1.003 coo.py:460(_sum_duplicates)
            1    0.667    0.667    0.667    0.667 {built-in method numpy.core.multiarray.lexsort}
     1001/501    0.006    0.000    0.273    0.001 coo.py:118(__init__)
         1001    0.008    0.000    0.179    0.000 coo.py:212(_check)
         4504    0.171    0.000    0.171    0.000 {method 'reduce' of 'numpy.ufunc' objects}
          500    0.004    0.000    0.156    0.000 compressed.py:905(tocoo)
            1    0.001    0.001    0.108    0.108 <ipython-input-61-ae99d5d6084c>:1(make_full_csr_matrix_list)
     


```python
%prun -l 15 -s "cumulative" create_block_diagonal_csr_matrix_sps(shape_list[-1], FullCsrMatrixFactory, keep_density = keep_density)
```

             114250 function calls (113248 primitive calls) in 1.689 seconds
    
       Ordered by: cumulative time
       List reduced from 93 to 15 due to restriction <15>
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    1.689    1.689 {built-in method builtins.exec}
            1    0.010    0.010    1.688    1.688 <string>:1(<module>)
            1    0.014    0.014    1.678    1.678 <ipython-input-63-336c75f19800>:1(create_block_diagonal_csr_matrix_sps)
            1    0.036    0.036    1.664    1.664 construct.py:617(block_diag)
            1    0.130    0.130    1.509    1.509 construct.py:501(bmat)
            1    0.000    0.000    1.097    1.097 base.py:236(asformat)
            1    0.000    0.000    1.097    1.097 coo.py:301(tocsr)
            1    0.022    0.022    1.028    1.028 coo.py:449(sum_duplicates)
            1    0.219    0.219    1.006    1.006 coo.py:460(_sum_duplicates)
            1    0.671    0.671    0.671    0.671 {built-in method numpy.core.multiarray.lexsort}
     1001/501    0.006    0.000    0.274    0.001 coo.py:118(__init__)
         1001    0.008    0.000    0.179    0.000 coo.py:212(_check)
         4504    0.173    0.000    0.173    0.000 {method 'reduce' of 'numpy.ufunc' objects}
          500    0.004    0.000    0.157    0.000 compressed.py:905(tocoo)
          501    0.002    0.000    0.118    0.000 <ipython-input-62-df5c2c6352ed>:43(__iter__)
     

The majority of the time is spent with [converting](https://github.com/scipy/scipy/blob/v1.0.0/scipy/sparse/construct.py#L569) the `csr` matrices to coordinate (`coo`)  format, in both cases. What can be done to avoid this expensive transformation?

1. Create `coo` matrices. **Why not?** We will need `csr` matrices thus the same amount of conversion is needed, but to the opposite direction.
2. Create a sequence of `csr` matrices and sequentially add them to the `indices`, `indptr` and `data` fields of the final matrix. **Why not?** Too much moving aroung of memory.
3. Write a native numpy implementation. **Why?** Why not? We expect it to be efficient.

### 2. Pure `numpy` implementation

The patient reader is only asked to digest one more chunk of code. In the `create_block_diagonal_csr_matrix_np` implementation we circumvent any recursion to third party libraries such as `scipy.sparse`. The procedure can be outlined as follows:

**Algorithm 2**

$
\begin{eqnarray}
   &1.& size \leftarrow \text{sum of block sizes} \\
   &2.& \mathbf{A} \in \mathcal{R}^{size \times size}  \leftarrow \text{empty} \\
   &3.& \text{choose row indices to be kept randomly} \\
   &4.& \text{calculate } \mathbf{A}.indptr \\
   &5.& \text{calculate } \mathbf{A}.indices \\
   &6.& \text{fill } \mathbf{A}.data
\end{eqnarray}
$

In addition, the maximum temporary storage is always at most $N^{2}$, where $N$ is the number of columns in the largest block of the matrix.


```python
def create_block_diagonal_csr_matrix_np(block_sizes, fill_value = 1, keep_density = 0.1, keep_each_block = True):
  """
  Creates a block diagonal csr matrix.
  Parameters:
    block_sizes ([int]) : sizes of the blocks.
    fill_value (int) : the value of the elements. Default 1.
    keep_density (float) : the proportion of rows to be kept. Default 0.01.
    keep_each_block (bool) : whether to keep at least one row from each block. Default True
  """
  block_sizes_ = np.array(block_sizes, dtype = np.int)
# number of columns in each block
  n_cols = np.array(block_sizes, dtype = np.int)
# number of rows in each block
  n_rows = np.rint(block_sizes_ * keep_density).astype(np.int)

# keep one row from each block all blocks
  if keep_each_block:
    n_rows[n_rows == 0] = 1

# discard empty blocks
  n_cols = n_cols[n_rows > 0]
  n_rows = n_rows[n_rows > 0]

# total number of rows, columns and nonzero elements
  num_tot_rows = np.sum(n_rows)
  num_tot_cols = np.sum(n_cols)
  num_tot_nnz = np.sum(n_cols * n_rows)

# create empty matrix with the right dimensions
  mat = sps.csr_matrix((num_tot_rows, num_tot_cols), dtype = np.int)

# set cumulative number of nnz elements for each row
  mat.indptr[1:] = np.cumsum(np.repeat(n_cols, n_rows))

# set column indices in each row
  mat.indices = np.zeros(num_tot_nnz, dtype = np.int)
  offset = 0 # offset of column indices for each block
  ilow = 0  # position of start index in indices for each row

# fill in column indices for each row successively
  for ncol, nrow in zip(n_cols, n_rows):
    ihgh = ilow + nrow * ncol
    mat.indices[ilow:ihgh] = np.tile(np.arange(offset, offset + ncol), nrow)
    offset += ncol
    ilow = ihgh + 0

# set data
  mat.data = np.full(num_tot_nnz, fill_value)

  return mat
```

Let us see whether we have managed to write a faster routine.


```python
performance = []

shape_list.extend([[x] * x for x in [1000, 2000]])

for shapes in shape_list:
    tstart = perf_counter()
    matrix = create_block_diagonal_csr_matrix_np(shapes, keep_density = keep_density)
    tdelta = perf_counter() - tstart
    res_dict = {'matrix_size' : matrix.size}
    res_dict.update({'time' : tdelta})
    performance.append(res_dict)
    
for res_dict in performance:
   print("\t".join(["{0} : {1}".format(_k, _v) for _k, _v in res_dict.items()]))
```

    matrix_size : 100	time : 0.014747986983934425
    matrix_size : 12500	time : 0.0008425126153497331
    matrix_size : 100000	time : 0.0033231946222258557
    matrix_size : 12500000	time : 0.049718061721250706
    matrix_size : 100000000	time : 0.3397356259672506
    matrix_size : 800000000	time : 3.803759712546139
    

Indeed, it is about thirty times faster than the solutions using the `scipy.sparse` functions.

## Summary

We have learnt a number of tricks. Sparse matrices provide an efficient way to store large volume of numeric data. One should always excercise caution at constructing them in order to avoid costly memory manipulations. Using low level numpy functions as opposed to built-in sparse matrix functions can considerably speed up the creation of them. 
