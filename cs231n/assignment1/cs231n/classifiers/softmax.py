from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    classes = W.shape[1]
    for i in range(num_train):
        # 1. 得到分数矩阵
        scores = X[i].dot(W)
        # 2. 对分数进行正则化，即每一个都减去当前行的最大分数值
        max_score = np.max(scores)
        scores -= max_score
        # 3. softmax转换成概率
        probs = np.exp(scores)
        # 概率归一化
        probs_all = np.sum(probs)
        probs_reg = probs / probs_all
        # 4. cross_entropy计算交叉熵损失函数
        loss += (-1) * np.log(probs_reg[y[i]])
        # 5. 求梯度
        probs_square = np.square(probs_all)
        # 6. 链式求导法则，先用loss对probs_reg求导，后面的循环用probs_reg对scores求导，再scores对W求导
        # (注意使用计算图---By Serena Young漂亮助教小姐姐的意见，其中我把softmax作为一个计算结点，实际上也可拆分开来)
        pre_chain = (-1 / probs_reg[y[i]])
        for j in range(classes):
            if j == y[i]:
                dW[:, j] += pre_chain * (probs[j] * (probs_all - probs[j]) / probs_square) * X[i]
            else:
                dW[:, j] += pre_chain * (-1 * probs[j] * probs[y[i]] / probs_square) * X[i]
        
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W * W)
    dW += reg * W * 2
    
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    # 1. 按行找出最大
    max_scores = np.max(scores, axis=1).reshape(-1, 1)
    # 2. 正则化分数
    scores -= max_scores
    # 3. Softmax操作
    probs = np.exp(scores)
    probs_sum = np.sum(probs, axis=1).reshape(-1, 1)
    probs_reg = np.true_divide(probs, probs_sum)
    # 4. 计算损失
    index = np.arange(num_train)
    loss += np.sum(-1 * np.log(probs_reg[index, y]))
    # 5. 计算梯度(链式求导法则)
    
    #  (根据最简求导式可以直接得到，但需要预先构造一个独热编码的矩阵)
    one_hot = np.zeros_like(probs_reg)
    one_hot[index, y] = 1
    dScores = probs_reg - one_hot
    dW = np.dot(X.T, dScores)
    
    
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W * W)
    dW += reg * W * 2
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
