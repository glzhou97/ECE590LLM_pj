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
from fast_attention import hyperbolic_softmax_kernel_feature_creator
from fast_attention import nonnegative_softmax_kernel_feature_creator
import numpy as np

# this experiment for testing sm^{hyp+} vs. sm, using orthogomal randomization


def Exp3():

    # query -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # key -> [batch_size, dim1, dim2, ..., dimN, num_heads, mem_channels]
    # value -> [batch_size, dim1, dim2, ..., dimN, num_heads, value_channels]
    sm_errors, hypo_errors = [], []
    sm_errors_std, hypo_errors_std = [], []
    random_feature_range = range(1, 100, 5)
    for nb_random_features in random_feature_range:
        qk_dim = 16  # dimension of embeddings after W_k or W_q's transformation
        v_dim = 16  # dimension of embeddings after W_vs transformation
        seq_length = 4000
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

        ortho_random_matrix_creator = functools.partial(
            fast_attention.GaussianOrthogonalRandomMatrix, nb_random_features,
            qk_dim)
        sm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            ortho_random_matrix_creator, nonnegative_softmax_kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)
        hypo_rfm_dot_product_attention = fast_attention.FastAttentionviaLowRankDecomposition(
            ortho_random_matrix_creator, hyperbolic_softmax_kernel_feature_creator,
            renormalize_attention, numerical_stabilizer, redraw_features,
            unidirectional)

        standard_attention_result = attention.dot_product_attention(
            query, key, value)
        sm_rfm_attention_result = sm_dot_product_attention.dot_product_attention(
            query, key, value)
        hypo_rfm_attention_result = hypo_rfm_dot_product_attention.dot_product_attention(
            query, key, value)

        sm_errors_per_batch, hypo_errors_per_batch = [], []
        # calculate the error per batch and aggregate to get stc
        for b in range(0, batch_size):
            sm_error = (
                standard_attention_result[b] - sm_rfm_attention_result[b])**2
            hypo_error = (
                standard_attention_result[b] - hypo_rfm_attention_result[b])**2
            sm_errors_per_batch.append(np.mean(sm_error))
            hypo_errors_per_batch.append(np.mean(hypo_error))

        sm_errors.append(np.mean(sm_errors_per_batch))
        hypo_errors.append(np.mean(hypo_errors_per_batch))
        sm_errors_std.append(np.std(sm_errors_per_batch))
        hypo_errors_std.append(np.std(hypo_errors_per_batch))

    plt.plot(random_feature_range, sm_errors,
             label=r'$\widehat{SM}$ Estimator', color='red')
    plt.plot(random_feature_range, hypo_errors,
             label=r'$\widehat{SM}^{hyp+}$ Estimator', color='blue')
    plt.fill_between(random_feature_range, np.array(sm_errors)-np.array(sm_errors_std),
                     np.array(sm_errors)+np.array(sm_errors_std), color='red', alpha=0.3)
    plt.fill_between(random_feature_range, np.array(hypo_errors)-np.array(hypo_errors_std),
                     np.array(hypo_errors)+np.array(hypo_errors_std), color='blue', alpha=0.3)
    plt.xlabel('Number of Random Features')
    plt.ylabel('MSE')
    plt.title(
        r'Attention matrix comparison of $\widehat{SM}$ and $\widehat{SM}^{hyp+}$')
    plt.yscale('log')
    plt.legend()
    plt.savefig('exp3.png', dpi=500, bbox_inches="tight")


Exp3()
