import asyncio
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer
import jax.numpy as jnp
import os
from pathlib import Path
import jax.nn as jnn
import jax
import numpy as np
from jax.scipy.special import entr
from queue import Queue
from threading import Lock
import time
import psutil
from jax import pmap, vmap
import jax.lax as lax
from functools import partial
import logging
logger = logging.getLogger(__name__)

class LanguageModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.min_batch_size = 16
        self.max_batch_size = 128
        self.current_batch_size = self.min_batch_size
        self.inference_queue = None
        self.lock = None
        self.memory_threshold = 0.8  # 80% memory usage threshold
        self.loop = None  # We'll set this later in set_event_loop
        self.eps = 0.01  # Default value for eps
        self.eff_zero = 1e-15
        self.vocab_size = None  # We'll set this when loading the model

    @classmethod
    def from_pretrained(cls, model_name, cache_dir, **kwargs):
        if cache_dir is None:
            cache_dir = Path(os.environ.get('MODEL_CACHE_DIR', '/home/cloudforest/Weights/pretrained'))
        else:
            cache_dir = Path(cache_dir)
        
        if model_name is None:
            available_models = [d.name for d in cache_dir.iterdir() if d.is_dir()]
            if not available_models:
                raise ValueError(f"No models found in {cache_dir}. Please specify a model name.")
            print("Available models:", ", ".join(available_models))
            model_name = input("Enter the name of the model to use: ")
        
        instance = cls()
        try:
            instance.model = FlaxAutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", **kwargs)
        except Exception as e:
            print(f"Failed to initialize model with CUDA: {e}. Falling back to CPU.")
            jax.config.update('jax_platform_name', 'cpu')
            instance.model = FlaxAutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
        
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
        
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        instance.vocab_size = len(instance.tokenizer)  # Set the vocabulary size
        return instance

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    async def get_next_token_logits(self, state):
        async with self.lock:
            await self.inference_queue.put(state)
            if self.inference_queue.qsize() >= self.current_batch_size:
                states = [await self.inference_queue.get() for _ in range(self.current_batch_size)]
                logits, _ = await self.get_next_token_logits_batch(states)
                return logits[0], state
            else:
                return None, state

    async def get_next_token_logits_batch(self, states, max_wait_time=0.5):
        batch = []
        start_time = time.time()  # Define start_time here
        
        for state in states:
            await self.inference_queue.put(state)
        
        while len(batch) < self.max_batch_size and time.time() - start_time < max_wait_time:
            try:
                state = await asyncio.wait_for(self.inference_queue.get(), timeout=max_wait_time - (time.time() - start_time))
                batch.append(state)
            except asyncio.TimeoutError:
                break

        if not batch:
            return None

        try:
            max_length = max(state.shape[0] for state in batch)
            padded_states = [jnp.pad(state, (0, max_length - state.shape[0]), constant_values=self.tokenizer.pad_token_id) for state in batch]
            input_ids = jnp.stack(padded_states)
            
            attention_mask = (input_ids != self.tokenizer.pad_token_id).astype(jnp.int32)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits[:, -1, :]
            
            return logits
        except Exception as e:
            print(f"Error in get_next_token_logits_batch: {e}")
            return None

    async def extract_distribution_batch(self, states, temperature=1.0, min_prob=1e-6, entropy_factor=1.0):
        logits = await self.get_next_token_logits_batch(states)
        if logits is None:
            return None

        clipped_probs, entropies = self._clip_logits_batch(logits, temperature, min_prob)
        truncated_probs, truncated_indices = self._truncate_probs_batch(clipped_probs, entropy_factor ** entropies)
        optimal_topk_batch = self._find_optimal_topk_batch(clipped_probs, self.eps)

        results = []
        for i in range(truncated_probs.shape[0]):
            valid_mask = truncated_indices[i] != -1
            distribution = {int(idx): float(prob) for idx, prob in zip(truncated_indices[i][valid_mask], truncated_probs[i][valid_mask])}
            results.append((distribution, float(entropies[i]), int(optimal_topk_batch[i])))

        return results

    def set_event_loop(self, loop):
        self.loop = loop
        self.inference_queue = asyncio.Queue()

    def set_eps(self, eps):
        self.eps = eps

    @staticmethod
    @jax.jit
    def _clip_logits_batch(logits, temperature, min_prob):
        scaled_logits = logits / temperature
        probs = jax.nn.softmax(scaled_logits, axis=-1)
        clipped_probs = jnp.where(probs < min_prob, 0.0, probs)
        clipped_probs = clipped_probs / jnp.sum(clipped_probs, axis=-1, keepdims=True)
        entropy = jnp.sum(-clipped_probs * jnp.log(clipped_probs + 1e-10), axis=-1)
        return clipped_probs, entropy

    @staticmethod
    @jax.jit
    def _find_optimal_topk(probs, eps):
        sorted_probs = jnp.sort(probs)[::-1]  # Sort in descending order
        cumsum_probs = jnp.cumsum(sorted_probs)
        kl_divs = - jnp.log(cumsum_probs)
        # Find the first entry in kl_div smaller than eps
        optimal_topk = jnp.sum(kl_divs >= eps)
        return optimal_topk

    @staticmethod
    @jax.jit
    def _find_optimal_topk_batch(probs, eps):
        return jax.vmap(LanguageModel._find_optimal_topk, in_axes=(0, None))(probs, eps)

    @staticmethod
    @jax.jit
    def _truncate_probs(probs, num):
        # Sort probabilities in descending order
        sorted_indices = jnp.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Create a mask for the top 'num' probabilities
        mask = jnp.arange(len(probs)) < num
        
        # Apply the mask to the sorted probabilities and indices
        truncated_probs = jnp.where(mask, sorted_probs, 0.0)
        truncated_probs /= jnp.sum(truncated_probs)
        truncated_indices = jnp.where(mask, sorted_indices, -1)
        
        return truncated_probs, truncated_indices

    @staticmethod
    @jax.jit
    def _truncate_probs_batch(probs_batch, nums):
        return jax.vmap(LanguageModel._truncate_probs)(probs_batch, nums)

