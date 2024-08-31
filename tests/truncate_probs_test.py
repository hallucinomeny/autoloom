

import jax
import jax.numpy as jnp

@jax.jit
def truncate_single(probs, num):
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


@jax.jit
def truncate_probs_batch(probs_batch, nums):
    return jax.vmap(truncate_single)(probs_batch, nums)

import unittest
import numpy as np

class TestTruncateProbsFunctions(unittest.TestCase):
    def setUp(self):
        # Set up some test data
        self.probs = jnp.array([0.1, 0.2, 0.3, 0.4])
        self.num = 2
        self.probs_batch = jnp.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.25, 0.25, 0.25, 0.25]
        ])
        self.nums = jnp.array([2, 3, 4])

    def test_truncate_single(self):
        truncated_probs, truncated_indices = truncate_single(self.probs, self.num)
        
        # Check if only top 2 probabilities are non-zero
        self.assertEqual(np.count_nonzero(truncated_probs), 2)
        
        # Check if probabilities sum to 1
        self.assertAlmostEqual(jnp.sum(truncated_probs), 1.0)
        
        # Check if indices are correct
        expected_indices = jnp.array([3, 2, -1, -1])
        np.testing.assert_array_equal(truncated_indices, expected_indices)

    def test_truncate_probs_batch(self):
        truncated_probs_batch, truncated_indices_batch = truncate_probs_batch(self.probs_batch, self.nums)
        
        # Check if batch shape is preserved
        self.assertEqual(truncated_probs_batch.shape, self.probs_batch.shape)
        
        # Check if each row has correct number of non-zero elements
        for row, num in zip(truncated_probs_batch, self.nums):
            self.assertEqual(np.count_nonzero(row), num)
        
        # Check if probabilities in each row sum to 1 (or very close to 1)
        row_sums = jnp.sum(truncated_probs_batch, axis=1)
        np.testing.assert_allclose(row_sums, jnp.ones_like(row_sums), rtol=1e-5)
    
    def test_truncate_single_edge_cases(self):
        # Test with num greater than array length
        probs = jnp.array([0.2, 0.3, 0.5])
        num = 5
        truncated_probs, truncated_indices = truncate_single(probs, num)
        self.assertEqual(np.count_nonzero(truncated_probs), 3)
        np.testing.assert_array_equal(truncated_indices, [2, 1, 0])
        
        # Test with num equal to zero
        num = 0
        truncated_probs, truncated_indices = truncate_single(probs, num)
        np.testing.assert_array_equal(truncated_indices, [-1, -1, -1])
        
        # Test with all equal probabilities
        probs = jnp.array([0.25, 0.25, 0.25, 0.25])
        num = 2
        truncated_probs, truncated_indices = truncate_single(probs, num)
        self.assertEqual(np.count_nonzero(truncated_probs), 2)
        self.assertAlmostEqual(jnp.sum(truncated_probs), 1.0)
        
        # Test with very small probabilities
        probs = jnp.array([1e-10, 1e-9, 1e-8, 0.99999999])
        num = 3
        truncated_probs, truncated_indices = truncate_single(probs, num)
        self.assertEqual(np.count_nonzero(truncated_probs), 3)
        self.assertAlmostEqual(jnp.sum(truncated_probs), 1.0)

    def test_batch_equivalent_to_batch_loop(self):
        # Test with various batch sizes, num_to_keep values, and sequence lengths
        batch_sizes = [1, 5, 10, 50, 100]
        num_to_keep_values = [1, 3, 5, 7, 10]
        seq_lengths = [10, 20, 50, 100]
        num_trials = 10  # Number of randomized trials for each configuration

        for batch_size in batch_sizes:
            for num_to_keep in num_to_keep_values:
                for seq_length in seq_lengths:
                    for _ in range(num_trials):
                        # Generate random probabilities
                        key = jax.random.PRNGKey(jax.random.randint(jax.random.PRNGKey(0), (), 0, 1000000))
                        probs = jax.random.uniform(key, shape=(batch_size, seq_length))
                        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize

                        # Randomly generate nums (number to keep for each batch)
                        nums = jax.random.randint(key, (batch_size,), 1, min(num_to_keep, seq_length) + 1)

                        # Run batch version
                        truncated_probs_batch, truncated_indices_batch = truncate_probs_batch(probs, nums)

                        # Run loop version
                        truncated_probs_loop = []
                        truncated_indices_loop = []
                        for i in range(batch_size):
                            trunc_probs, trunc_indices = truncate_single(probs[i], nums[i])
                            truncated_probs_loop.append(trunc_probs)
                            truncated_indices_loop.append(trunc_indices)

                        truncated_probs_loop = jnp.array(truncated_probs_loop)
                        truncated_indices_loop = jnp.array(truncated_indices_loop)

                        # Compare results
                        np.testing.assert_allclose(truncated_probs_batch, truncated_probs_loop, rtol=1e-5)
                        np.testing.assert_array_equal(truncated_indices_batch, truncated_indices_loop)

                        # Additional checks
                        self.assertEqual(truncated_probs_batch.shape, (batch_size, seq_length))
                        self.assertEqual(truncated_indices_batch.shape, (batch_size, seq_length))
                        for i in range(batch_size):
                            self.assertEqual(np.count_nonzero(truncated_probs_batch[i]), nums[i])
                            self.assertAlmostEqual(jnp.sum(truncated_probs_batch[i]), 1.0, places=5)



if __name__ == '__main__':
    unittest.main()
