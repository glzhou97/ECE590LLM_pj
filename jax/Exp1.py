import functools
import time
from absl import logging
from absl.testing import absltest
from flax.linen import attention
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp
import fast_attention
from fast_attention_test import kernel_feature_creator
import numpy as np

#this function is for plotting the results of the experiment
def plot_results(unstruct_errors, ortho_errors):
    import matplotlib.pyplot as plt
    plt.plot([10, 20, 50, 80, 100, 120, 150, 200], unstruct_errors, label='Unstructured Random Matrix')
    plt.plot([10, 20, 50, 80, 100, 120, 150, 200], ortho_errors, label='Orthogonal Random Matrix')
    plt.xlabel('Number of Random Features')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig('exp1.png')

#this experiment is responsible to test the fast softmax attention vs. the regular softmax attention
def Exp1():

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]
    unstruct_errors, ortho_errors = [], []
    for nb_random_features in [10, 20, 50, 80, 100, 120, 150, 200]:
        qk_dim = 10
        v_dim = 10  # this might be sequence length?
        batch_size = 1
        dim1 = 16  # this is embedding dimension? 
        num_heads = 1
        shape_query = (batch_size, dim1, num_heads, qk_dim)
        shape_key = (batch_size, dim1, num_heads, qk_dim)
        shape_value = (batch_size, dim1, num_heads, v_dim)
        query = random.normal(random.PRNGKey(0), shape_query)
        key = random.normal(random.PRNGKey(0), shape_key)
        value = random.normal(random.PRNGKey(0), shape_value)

        renormalize_attention = True
        numerical_stabilizer = 0.0
        redraw_features = False
        unidirectional = False

        unstructured_random_matrix_creator = functools.partial(
            fast_attention.GaussianUnstructuredRandomMatrix, nb_random_features,
            qk_dim)
        ortho_random_matrix_creator = functools.partial(
            fast_attention.GaussianOrthogonalRandomMatrix, nb_random_features,
            qk_dim)
        fast_unstruct_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            unstructured_random_matrix_creator, kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)
        fast_ortho_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            ortho_random_matrix_creator, kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)

        standard_attention_result = attention.dot_product_attention(
            query, key, value)
        unstruct_rfm_attention_result = fast_unstruct_rfm_dot_product_attention.dot_product_attention(
            query, key, value)
        ortho_rfm_attention_result = fast_ortho_rfm_dot_product_attention.dot_product_attention(
            query, key, value)

        unstruct_error = jnp.abs((standard_attention_result - unstruct_rfm_attention_result)**2)
        ortho_error = jnp.abs((standard_attention_result - ortho_rfm_attention_result)**2)
        unstruct_errors.append(np.mean(unstruct_error))
        ortho_errors.append(np.mean(ortho_error))
    plot_results(unstruct_errors, ortho_errors)

Exp1()
