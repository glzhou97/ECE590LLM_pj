import functools
import time
import matplotlib.pyplot as plt
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

#this experiment is responsible to test the fast softmax attention vs. the regular softmax attention
def Exp1():

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]
    unstruct_errors, ortho_errors = [], []
    unstruct_errors_std, ortho_errors_std = [], []
    random_feature_range = range(15, 300, 20)
    for nb_random_features in random_feature_range:
        qk_dim = 16 # dimension of embeddings after W_k or W_q's transformation
        v_dim = 16 # dimension of embeddings after W_vs transformation
        seq_length = 4096
        num_heads = 1
        batch_size = 15
        shape_query = (batch_size, seq_length, num_heads, qk_dim)
        shape_key = (batch_size, seq_length, num_heads, qk_dim)
        shape_value = (batch_size, seq_length, num_heads, v_dim)
        query = random.normal(random.PRNGKey(0), shape_query)
        key = random.normal(random.PRNGKey(0), shape_key)
        value = random.normal(random.PRNGKey(0), shape_value)

        renormalize_attention = True
        numerical_stabilizer = 0
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

        unstruct_errors_per_batch, ortho_errors_per_batch = [], []
        # calculate the error per batch and aggregate to get stc
        for b in range(0, batch_size): 
            unstruct_error = (standard_attention_result[b] - unstruct_rfm_attention_result[b])**2
            ortho_error = (standard_attention_result[b] - ortho_rfm_attention_result[b])**2
            unstruct_errors_per_batch.append(np.mean(unstruct_error))
            ortho_errors_per_batch.append(np.mean(ortho_error))
        
        unstruct_errors.append(np.mean(unstruct_errors_per_batch))
        ortho_errors.append(np.mean(ortho_errors_per_batch))
        unstruct_errors_std.append(np.std(unstruct_errors_per_batch))
        ortho_errors_std.append(np.std(ortho_errors_per_batch))

    plt.plot(random_feature_range, unstruct_errors, label='Unstructured Random Matrix', color='red')
    plt.plot(random_feature_range, ortho_errors, label='Orthogonal Random Matrix', color='blue')
    plt.fill_between(random_feature_range, np.array(unstruct_errors)-np.array(unstruct_errors_std), np.array(unstruct_errors)+np.array(unstruct_errors_std), color='red', alpha=0.3)
    plt.fill_between(random_feature_range, np.array(ortho_errors)-np.array(ortho_errors_std),np.array(ortho_errors)+np.array(ortho_errors_std), color='blue', alpha=0.3)
    plt.xlabel('Number of Random Features')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig('exp1.png')


Exp1()
