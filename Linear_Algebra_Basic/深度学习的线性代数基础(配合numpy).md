# 深度学习的线性代数基础(配合numpy)



## 1.标量、向量、矩阵和张量

**标量**就是一个单独的数。

**向量**是一列数，有行向量和列向量，numpy对行向量的支持并不是特别好。很多时候从矩阵中取出一行你会发现仅仅是一个数组。

可以运行下列代码，col就是创建了列向量，row就是创建了行向量，最后就是一个数组。

```python
import numpy as np

col = np.array([[1], [2], [3]])
print(col.shape)
#(3, 1)

row = np.array([1, 2, 3]).reshape(1, -1)
# row = np.array([[1, 2, 3]])
print(row.shape)
#(1, 3)

array = np.array([1, 2, 3])
print(array.shape)
#(3,)
```



**矩阵**就是二维数组。可以模仿行列向量的形式去构造一个矩阵，也要对numpy提供的特殊的全0全1，对角矩阵等构造方式比较熟悉。

```python
matrix = np.array([[1, 2, 3], [5, 6, 7]])
print(matrix.shape)

matrix_0 = np.zeros((3, 3))
print(matrix_0.shape)
```



**张量**用来描述超过二维的数组。python的深度学习库torch、tensorflow都对张量提供了很好的支持。

为什么需要超过二维的数组？我们在训练网络时，是一个batch的图像喂给神经网络的吧，所以看到的数据基本都是[batch, w, h, channel]这样的格式，深层次的神经元往往会提取出更高维的特征。张量还有一个特性就是可以对其求导，当然numpy也能求导，但这里指的是autograd自动求导机制。

下面的代码构造了三维张量img，第一维度有两个元素，其中每个元素就是一个3 * 3的矩阵。

```python
import torch

img = torch.randn((2, 3, 3))
print(img.shape, img.type, img, end='\n')

'''
torch.Size([2, 3, 3]) <built-in method type of Tensor object at 0x0000025EFC887270> tensor([[[-0.9109,  0.4624,  1.7858],
         [ 0.7252,  0.2279, -0.7761],
         [ 0.1194, -0.5134, -0.5476]],

        [[ 1.2003,  0.4247,  2.9966],
         [-0.0940,  0.2624,  1.0951],
         [-1.1343, -0.6103,  0.8214]]])
'''
```



## 2.矩阵向量运算

### 2.1 转置

在我们构造的matrix里索引一个值是很简单的事，把行列值打进去就行，注意行列起始都从0开始。

下列代码提供了两种索引方式，应该是后面这种更受欢迎！

```python
matrix = np.array([[1, 2, 3], [5, 6, 7]])
print(matrix[1][2], matrix[1, 2])
```



转置就是把矩阵第一行变为第一列，第二行变为第二列，以此类推。因此如果原先矩阵大小是(m, n)，转置后就成了(n , m)了。看一个实例(提供了两种方式转置)：

```python
import numpy as np
import torch

matrix = np.array([[1, 2, 3], [5, 6, 7]])
print(matrix)

print(matrix.T)

matrix_t = np.transpose(matrix)
print(matrix_t)

'''
[[1 2 3]
 [5 6 7]]
[[1 5]
 [2 6]
 [3 7]]
[[1 5]
 [2 6]
 [3 7]]
'''
```



因此通过转置，原先在矩阵(i, j)位置的值，可以通过(j, i)在转置后矩阵索引得到。

行列向量转置无非就是从行->列，从列->行，标量转置是其本身。



### 2.2 矩阵运算

标量+矩阵，这是一个element-wise的操作，就是说是对矩阵中每个元素执行的。即矩阵中每个元素的值都加上此标量。

标量*矩阵，同上。

矩阵+矩阵，一定要确保两矩阵shape相同，然后是对应位置相加，结果矩阵大小和原矩阵相同。



