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


def get_run_time_ms(fast: bool, mode: str, length: int, batch_size: int = 1, nb_features: int = 256, features_type: str = 'iid'):
    jit = True
    qk_dim = 16
    sample_number = 10
    num_heads = 1
    renormalize_attention = True
    unidirectional = False

    if fast:
        raw_attention_fn = fast_attention.make_fast_generalized_attention(
            qk_dim // num_heads,
            renormalize_attention=renormalize_attention,
            nb_features=nb_features,
            features_type=features_type,
            unidirectional=unidirectional)
    else:
        raw_attention_fn = attention.dot_product_attention

    def sum_attention_fn(*args, **kwargs):
        return jnp.sum(raw_attention_fn(*args, **kwargs))

    if jit:
        attention_fn = jax.jit(sum_attention_fn)

    shape_query = (batch_size, length, num_heads, qk_dim)
    shape_key = (batch_size, length, num_heads, qk_dim)
    shape_value = (batch_size, length, num_heads, qk_dim)

    query = jnp.array(onp.random.rand(*shape_query)) * 0.001
    key = jnp.array(onp.random.rand(*shape_key)) * 0.001
    value = jnp.array(onp.random.rand(*shape_value)) * 0.001

    raw_grad_fn = jax.grad(lambda q: sum_attention_fn(q, key=key, value=value))
    def grad_fn(q): return jnp.sum(raw_grad_fn(q))

    if jit:
        grad_fn = jax.jit(grad_fn)

    time_taken = []
    for s in range(sample_number):
        if mode == "forward":
            start = time.time()
            attention_fn(query, key, value).block_until_ready()
            end = time.time()
        elif mode == "backward":
            start = time.time()
            grad_fn(query).block_until_ready()
            end = time.time()
        time_taken.append(end - start)
    return np.mean(np.array(time_taken)) * 1000  # return in ms


def favor_plus_comparison():
    # comparing favor+ with differet number of features vs regular
    runtime_att = []
    runtime_fast_512 = []
    runtime_fast_256 = []
    runtime_fast_128 = []
    length_range = range(1, 15)
    
    fig, axs = plt.subplots(1, 2)
    back_prop = False
    for n in length_range:
        runtime_fast_512.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 512))
        runtime_fast_256.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 256))
        runtime_fast_128.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 128))
        runtime_att.append(get_run_time_ms(False, "backward" if back_prop else "forward", 2**n, 1, -1))

    axs[0, 0].plot(length_range, runtime_att,
             label='Regular softmax attention', color='red')
    axs[0, 0].plot(length_range, runtime_fast_512,
             label='FAVOR+ attention with 512 features', color='orange')
    axs[0, 0].plot(length_range, runtime_fast_256,
             label='FAVOR+ attention with 256 features', color='blue')
    axs[0, 0].plot(length_range, runtime_fast_128,
             label='FAVOR+ attention with 128 features', color='green')
    axs[0, 0].set_title(r'Forward pass')
    axs[0, 0].legend()

    back_prop = False
    for n in length_range:
        runtime_fast_512.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 512))
        runtime_fast_256.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 256))
        runtime_fast_128.append(get_run_time_ms(
            True, "backward" if back_prop else "forward", 2**n, 1, 128))
        runtime_att.append(get_run_time_ms(False, "backward" if back_prop else "forward", 2**n, 1, -1))

    axs[0, 0].plot(length_range, runtime_att,
             label='Regular softmax attention', color='red')
    axs[0, 0].plot(length_range, runtime_fast_512,
             label='FAVOR+ attention with 512 features', color='orange')
    axs[0, 0].plot(length_range, runtime_fast_256,
             label='FAVOR+ attention with 256 features', color='blue')
    axs[0, 0].plot(length_range, runtime_fast_128,
             label='FAVOR+ attention with 128 features', color='green')
    axs[0, 0].set_title(
        r'SBackpropogation')
    axs[0, 0].legend()

    for ax in axs.flat:
        ax.set(xlabel=r'$Log_2(L)$', ylabel='Log(T) (ms)')
    for ax in axs.flat:
        ax.label_outer()

    plt.yscale('log')
    plt.savefig('favor_plus.png', dpi=400, bbox_inches="tight")


def batch_size_comparison():
    # comparing favor+ with differet number of features vs regular
    batch_size = [1, 5, 10, 20]
    length_range = range(1, 15)
    back_prop = True
    for b in batch_size:
        time_per_batch = []
        for n in length_range:
            time_per_batch.append(get_run_time_ms(True, "backward" if back_prop else "forward", 2**n, b, 256))
        plt.plot(length_range, time_per_batch, label="batch size {}".format(b))
    plt.xlabel(r'$Log_2(L)$')
    plt.ylabel('Log(T) (ms)')
    plt.title(
        r'FAVOR+ attention backpropogation speed with different batch sizes')
    plt.yscale('log')
    plt.legend()
    plt.savefig('batch_size.png', dpi=400, bbox_inches="tight")


def Exp4():
    favor_plus_comparison()
    # batch_size_comparison()


Exp4()
