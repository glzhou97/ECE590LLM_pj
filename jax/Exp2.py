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
# note this is the sin cos matrix
from fast_attention_test import kernel_feature_creator
# this is the non-negative one
from fast_attention import nonnegative_softmax_kernel_feature_creator
import numpy as np

# this experiment for testing sin-cos feature generation & the exp positive feature generation


def Exp2():

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]
    sin_cos_errors, nonnegative_errors = [], []
    sin_cos_errors_std, nonnegative_errors_std = [], []
    random_feature_range = range(5, 300, 20)
    for nb_random_features in random_feature_range:
        qk_dim = 16  # dimension of embeddings after W_k or W_q's transformation
        v_dim = 16  # dimension of embeddings after W_vs transformation
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
        numerical_stabilizer = 1e-6
        redraw_features = False
        unidirectional = True

        unstructured_random_matrix_creator = functools.partial(
            fast_attention.GaussianUnstructuredRandomMatrix, nb_random_features,
            qk_dim)
        sin_cos_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            unstructured_random_matrix_creator, kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)
        nonnegative_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            unstructured_random_matrix_creator, nonnegative_softmax_kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)

        standard_attention_result = attention.dot_product_attention(
            query, key, value)
        sin_cos_rfm_attention_result = sin_cos_dot_product_attention.dot_product_attention(
            query, key, value)
        nonnegative_rfm_attention_result = nonnegative_rfm_dot_product_attention.dot_product_attention(
            query, key, value)

        sin_cos_errors_per_batch, nonnegative_errors_per_batch = [], []
        # calculate the error per batch and aggregate to get stc
        for b in range(0, batch_size):
            sin_cos_error = (
                standard_attention_result[b] - sin_cos_rfm_attention_result[b])**2
            nonnegative_error = (
                standard_attention_result[b] - nonnegative_rfm_attention_result[b])**2
            sin_cos_errors_per_batch.append(np.mean(sin_cos_error))
            nonnegative_errors_per_batch.append(np.mean(nonnegative_error))

        sin_cos_errors.append(np.mean(sin_cos_errors_per_batch))
        nonnegative_errors.append(np.mean(nonnegative_errors_per_batch))
        sin_cos_errors_std.append(np.std(sin_cos_errors_per_batch))
        nonnegative_errors_std.append(np.std(nonnegative_errors_per_batch))

    plt.plot(random_feature_range, sin_cos_errors,
             label='Sin/Cos Features', color='red')
    plt.plot(random_feature_range, nonnegative_errors,
             label='Non-negative Features', color='blue')
    plt.fill_between(random_feature_range, np.array(sin_cos_errors)-np.array(sin_cos_errors_std),
                     np.array(sin_cos_errors)+np.array(sin_cos_errors_std), color='red', alpha=0.3)
    plt.fill_between(random_feature_range, np.array(nonnegative_errors)-np.array(nonnegative_errors_std),
                     np.array(nonnegative_errors)+np.array(nonnegative_errors_std), color='blue', alpha=0.3)
    plt.xlabel('Number of Random Features')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title(
        'Attention matrix comparison of sine/cosine features vs non-negative features')
    plt.legend()
    plt.savefig('exp2.png', dpi=400, bbox_inches="tight")


Exp2()
