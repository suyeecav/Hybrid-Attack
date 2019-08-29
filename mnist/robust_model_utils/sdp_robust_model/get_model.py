
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from .two_layer import two_layer

""" 
Helper to return the correct model based on args
args["model"]: contains model

"""

def get_model (x, FLAGS, keep_prob = None, num_hidden = None):

    y_model = two_layer(x, FLAGS)

    return y_model
