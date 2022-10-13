import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))


    ### YOUR CODE HERE
    pass
    #2 loop cycle for image
    for i_image in range (Hk // 2, Hi - (Hk // 2)):
        for j_image in range (Wk //2, Wi - (Wk //2)):
            #creating additional var 
            out2 = 0
            #2 loop cycle for kernel
            for i_kernel in range (Hk):
                for j_kernel in range (Wk):
                    out2 += kernel[i_kernel][j_kernel] * image[i_image - i_kernel + 1][j_image - j_kernel + 1]
            out[i_image][j_image] = out2       
            
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))

    ### YOUR CODE HERE
    pass
    #going through image height and then width
    for i in range (H):
        for j in range (W):
            #add padding to height and weight
            out[i + pad_height][j + pad_width] = image[i][j]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    #add zero padding to image with parameter of 1/2 of kernel
    image = zero_pad(image, Hk // 2, Wk // 2)
    #flip kernel vertically
    kernel = np.flip(kernel, 0)
    #flip kernel horizontally
    kernel = np.flip(kernel, 1)
    
    
    for i_image in range (Hi):
        for j_image in range (Wi):
            #convolution
            out[i_image, j_image] = np.sum(kernel * image[i_image : i_image + Hk, j_image : j_image + Wk])
    
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    #cross-correlation defers from convolution by not flipping the kernel
    #flip g vertically
    g = np.flip(g, 0)
    #flip g horizontally
    g = np.flip(g, 1)
    #call convolution fast
    out = conv_fast(f, g)
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
 
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    
    for i in range (g.shape[0]):
        for j in range (g.shape[1]):
            g[i][j] = g[i][j] - g_mean
    
    out = cross_correlation(f, g)
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    #normalized g: g -mean(g) / deviation(g)
    g = (g - np.mean(g)) / np.std(g)
    #add zero pad to f
    f = zero_pad(f, Hg // 2, Wg // 2)
    
    ### YOUR CODE HERE
    
    for i in range (Hf):
        for j in range (Wf):
            p = f[i : i + Hg, j : j + Wg]
            norm_p = (p - np.mean(p)) / np.std(p)
            out[i,j] += np.sum(norm_p * g)
            
    pass
    ### END YOUR CODE

    return out
