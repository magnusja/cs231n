import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    log_probs = X[i].dot(W)
    stability = -log_probs.max()
    exp_this = np.exp(log_probs + stability)
    exp_sum = np.sum(exp_this)

    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += -X[i] + (exp_this[j] / exp_sum) * X[i]
      else:
        dW[:,j] += (exp_this[j] / exp_sum) * X[i]

    loss += -np.log(exp_this[y[i]] / exp_sum)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  log_probs = X.dot(W)
  exp_this = np.exp(log_probs)
  exp_sum = np.sum(exp_this, axis=1)

  loss = -np.log(exp_this[np.arange(num_train), y] / exp_sum)
  loss = np.sum(loss)

  loss /= num_train
  loss += reg * np.sum(W * W)

  grad = exp_this / exp_sum[:, np.newaxis]
  grad[np.arange(num_train),y] += -1.0
  dW = X.T.dot(grad)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

