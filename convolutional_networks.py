"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, ReLU, adam


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None

        # Implementation of convolutional forward pass.
        # Hint: we can use function torch.nn.functional.pad for padding.
        # We will NOT be using anything from torch.nn elsewhere.
        pad = conv_param['pad']
        stride = conv_param['stride']
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        N, C, H, W = x.shape
        F, C, HH, WW = w.shape

        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride

        out = torch.zeros(N, F, H_prime, W_prime, device=x.device)
        for n in range(N):
            for f in range(F):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        out[n, f, i//stride, j//stride] = torch.sum(x_pad[n, :, i:i+HH, j:j+WW] * w[f]) + b[f]

        cache = (x, w, b, conv_param)
        return out, cache


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None

        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']

        N, C, H, W = x.shape

        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        out = torch.zeros(N, C, H_prime, W_prime, device=x.device)

        for n in range(N):
            for c in range(C):
                for i in range(0, H_prime * stride, stride):
                    for j in range(0, W_prime * stride, stride):
                        out[n, c, i//stride, j//stride] = torch.max(x[n, c, i:i+pool_height, j:j+pool_width])

        cache = (x, pool_param)
        return out, cache

class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A custom layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        out = None
        cache = None

        #  Implementation of a forward function that first performs a
        # convolution, then a ReLU then a pool
        # done by refering to the Conv_ReLU class above
        conv_output, conv_cache = FastConv.forward(x, w, b, conv_param)
        relu_output, relu_cache = ReLU.forward(conv_output)
        out, pool_cache = FastMaxPool.forward(relu_output, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases for the three-layer convolutional network.
        # Weights should be initialized from a Gaussian distribution centered at 0.0
        # with a standard deviation equal to 'weight_scale'. Biases should be initialized to zero.
        # Store all weights and biases in the 'self.params' dictionary.

        # For the convolutional layer, use the keys 'W1' and 'b1'.
        # For the hidden linear layer, use keys 'W2' and 'b2'.
        # For the output linear layer, use keys 'W3' and 'b3'.

        # IMPORTANT: We can assume that the padding and stride of the first convolutional layer
        # are chosen so that the width and height of the input are preserved. Check the
        # 'loss()' function for more details on this.
        # HINT: Use input data (conv_param, pool_param, input_dim, filter_size etc) 
        #       to figure out the dimensions of weights and biases

        C, H, W = input_dims

        weight_conv = (num_filters, C, filter_size, filter_size)

        self.params['W1'] = weight_scale * torch.randn(weight_conv, dtype = dtype, device = device)
        self.params['b1'] = torch.zeros(num_filters, dtype = dtype, device = device)

        size_edited = (num_filters * (H // 2) * (W // 2))

        self.params['W2'] = weight_scale * torch.randn(size_edited, hidden_dim, dtype = dtype, device = device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype = dtype, device = device)
        self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype = dtype, device = device)
        self.params['b3'] = torch.zeros(num_classes, dtype = dtype, device = device)

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Inputs:
        - X: Tensor of input data
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the label for X[i].
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # Implementing the forward pass for the three-layer conv net.
        # Computing the class scores for X and storing them in the scores
        # variable. Store outputs of all the layers since they will be used
        # in computing gradients
        # HINT: Use forward functions of the custom layers that we have
        # already implemented before.

        # Forward pass
        out1, cache_CRP = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out2, cache_LR = Linear_ReLU.forward(out1, W2, b2)
        scores, cache_L = Linear.forward(out2, W3, b3)

        if y is None:
            return scores

        # Following is the implementation of the backward pass for the 
        # three-layer convolutional network.
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)


        # Incorporating L2 regularization into loss
        loss += self.reg * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2) + torch.sum(W3 ** 2))
        
        # Assigning the 2nd output variables (cache) of our custom layers to the given below 
        # variables so that it can be used in calculating gradients below
        cache_L = cache_L   # Assigned to 2nd output of forward for linear layer
        cache_LR = cache_LR # Assigned to 2nd output of forward for linear_relu layer
        cache_CRP = cache_CRP   # Assigned to 2nd output of forward for conv_relu_pool layer


        #Following code calculates gradients
        i = 3
        last_dout, dw, db  = Linear.backward(dout, cache_L)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Linear_ReLU.backward(last_dout, cache_LR)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db
        i-=1
        last_dout, dw, db  = Conv_ReLU_Pool.backward(last_dout, cache_CRP)
        grads['W{}'.format(i)] = dw + 2*self.params['W{}'.format(i)]*self.reg
        grads['b{}'.format(i)] = db

        return loss, grads


def create_convolutional_solver_instance(data_dict, dtype, device):

    #### putting our final hyperparameters here ####
    num_filters = 32
    filter_size = 3
    hidden_dim = 100
    reg = 0.001
    weight_scale = 0.01
    learning_rate = 0.001
    num_epochs = 30
    batch_size = 64
    update_rule = 'sgd_momentum'


    input_dims = data_dict['X_train'].shape[1:]
    model = ThreeLayerConvNet(num_filters=num_filters, filter_size=filter_size, 
                              hidden_dim=hidden_dim, reg=reg, weight_scale=weight_scale, 
                              dtype=torch.float, device='cpu')

    solver = Solver(model, data_dict,
                    num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=adam,
                    optim_config={
                      'learning_rate': learning_rate,
                    },
                    device='cpu')
    #solver.train()
    #solver.model.save('final_threeconvnet.pth')
    return solver