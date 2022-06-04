from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. 构造的W，也就是滤波核参数应该符合的形状是 W1:（Filter_NUM, Channel, Filter_Height, Filter_Width）  b1(Filter_Num,)
        self.params['b1'] = np.zeros(num_filters)
        # 此API可以生成符合高斯分布的参数，loc代表均值，scale代表标准差
        self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        
        # 经过第一层卷积，输出的应该是(N, numfilter, out_height, out_width)   ----> 其中N代表了训练MiniBatch的图片数量
        # 看提示，提示说明了，第一层卷积后我们可以默认为shape是preserved的，因此无需手算out的shape
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters * input_dim[1] * input_dim[2] // 4, hidden_dim))
        
        # 最后一层是全连接层
        self.params['b3'] = np.zeros(num_classes)
        self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        
        
        '''
        参数W2：实际上是第一次卷积输出后---大小为（Train_Num, Filter_Num, out_height, out_width），stretch成二维（Train_Num, NUM）。
        然后就回到了神经网络的知识，矩阵相乘即可。
        因此，这设置权值矩阵时，为了满足点乘规则即：X * W，W的大小需要为（NUM， hidden_dim）。所以知道确切的NUM的大小很重要！！！
        
        参数W2的设置遇到了问题，就是在第一次conv之后（根据提示不改变大小，即还是32*32），同时ReLU固然不改变大小。
        但是maxpool会改变大小啊，这里initialize的过程却又不给出max-pool相关参数，难以得知pooling操作后的size，那就没法设置W2大小啊？？
        
        我这里目前是根据先看了一下loss里的pooling参数，得知了会使得缩小4倍，因此直接在W2定义时整除了4！！
        '''

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 完成前向传播，利用init前的网络结构，copy到这儿：(去利用layers、fast_layers、layer_utils中已经实现的结构！！！)
        # -------------(思来想去还是尽量别去layers引用自己的naive算法，那真的很慢【如果你的实现和fast速度相差无几，我这就是屁话！】)--------------
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out, cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        out1, cache1 = affine_relu_forward(out, self.params['W2'], self.params['b2'])
        scores, cache2 = affine_forward(out1, self.params['W3'], self.params['b3'])
        
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        loss = loss + self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3'])))
        
        dout, dW3, db3 = affine_backward(dscores, cache2)
        grads['W3'] = dW3 + self.params['W3'] * self.reg
        grads['b3'] = db3
        
        dout, dW2, db2 = affine_relu_backward(dout, cache1)
        grads['W2'] = dW2 + self.params['W2'] * self.reg
        grads['b2'] = db2
        
        dout, dW1, db1 = conv_relu_pool_backward(dout, cache)
        grads['W1'] = dW1 + self.params['W1'] * self.reg
        grads['b1'] = db1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
