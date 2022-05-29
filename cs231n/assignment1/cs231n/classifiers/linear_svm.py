from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            # 链式求导 d_loss / d_score   *    d_score / d_W
            if margin > 0:
                loss += margin
                dW[:,j] += X[i, :]
                dW[:,y[i]] -= X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                 #
    # Compute the gradient of the loss function and store it dW.             #
    # Rather that first computing the loss and then computing the derivative,    #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW += reg * W * 2
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 根据Video08-Lecture04 back-propagation提供的思路进行
    
    # 1.矩阵点乘，scores (N, C), result[i, j] 就是第i张图像对应第j类的得分
    scores = np.dot(X, W)
    
    # 2. 根据max(0, s_i - s_yi + 1)计算margin
    num_train = X.shape[0]
    index = np.arange(num_train)
    # (我们先根据正确标签y获得（1000,1）[配合了index的使用]的正确分数，然后根据广播规则每一列都会减去这个值。np.maximum做的是element-wise的操作)
    scores -= scores[index, y].reshape(-1, 1)
    scores += 1
    margins = np.maximum(0, scores)
    # 因为根据上面的步骤，即使是正确的分类，也会有损失值1，显然我们需要剔除这些损失值，然后取均值
    data_loss = (np.sum(margins) - num_train) / num_train
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss
    
    # 和 Video上说的一样，共8行实现 Vectorized-Loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Video中提到了要学会利用 coputational graph将复杂计算分解成简单结点的简单计算
    
    # 与此同时，给到了提示是5行代码实现Vectorized-Gradient
    
    
    # 1. 对scores求导，易知：对于一个分数大于0的位置，其错误分数导数为1，正确分数导数为-1，得到的shape应该和scores同大小
    # 这步操作的意思是，对于margins，其中<=0的元素不变，其余位置全为1
    dscores = np.where(margins <= 0, margins, 1)
    # 显然对于正确分数那一列，还要减去n * 1， 其中n是当前行1的数目,所以要统计出每一行有几个1
    ones_num = np.count_nonzero(dscores, axis = 1)
    dscores[index, y] -= ones_num
    # 2. 通过dscores对dw求导
    # scores = XW
    dW = np.dot(X.T, dscores) / num_train
    # 3. 正则项求导
    dW += reg * W * 2
    
    
    # corresponding to the Video Hint，5行代码
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
