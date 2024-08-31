from tree import Node
import jax
import jax.numpy as jnp
from jax.scipy.special import entr

def calculate_node_score(node: Node) -> float:
    return node.logprob

def get_full_text(node: Node) -> str:
    text_parts = []
    current = node
    while current:
        text_parts.append(current.text)
        current = current.parent
    return ''.join(reversed(text_parts))

# def find_optimal_topk_slow(probs, eps=0.1):
#     sorted_probs = jnp.sort(probs)[::-1]  # Sort in descending order
#     cumulative_probs = jnp.cumsum(sorted_probs)
    
#     # Calculate KL divergence
#     kl = jnp.cumsum(sorted_probs * jnp.log(sorted_probs / cumulative_probs))
    
#     # Find the optimal k
#     threshold = eps * jnp.sum(entr(probs))
#     optimal_k = jnp.sum(kl < threshold)
    
#     return int(optimal_k)


def find_optimal_topk(probs, eps=0.01):
    sorted_probs = jnp.sort(probs)[::-1]  # Sort in descending order
    if sorted_probs[0] > 1 - eps:
        return 1

    i = 1
    while i < len(sorted_probs):
        truncated_probs = sorted_probs[:i] / jnp.sum(sorted_probs[:i])
        kl_div = jnp.sum(truncated_probs * (jnp.log(truncated_probs) - jnp.log(sorted_probs[:i])))
        if kl_div < eps:
            break
        i = i * 2

    left = i // 2
    right = min(i, len(sorted_probs) - 1)
    
    while left <= right:
        mid = (left + right) // 2
        truncated_probs = sorted_probs[:mid] / jnp.sum(sorted_probs[:mid])
        previous_truncated_probs = sorted_probs[:mid-1] / jnp.sum(sorted_probs[:mid-1])
        kl_div = jnp.sum(truncated_probs * (jnp.log(truncated_probs) - jnp.log(sorted_probs[:mid])))
        previous_kl_div = jnp.sum(previous_truncated_probs * (jnp.log(previous_truncated_probs) - jnp.log(sorted_probs[:mid-1])))
        if kl_div < eps and (mid == 1 or previous_kl_div >= eps):
            return mid
        elif kl_div < eps:
            right = mid - 1
        else:
            left = mid + 1
    return left

def _find_optimal_topk(probs, eps):
    sorted_probs = jnp.sort(probs)[::-1]  # Sort in descending order
    
    cumsum_probs = jnp.cumsum(sorted_probs)
    
    normalized_probs = jnp.expand_dims(sorted_probs, axis=0) / jnp.expand_dims(cumsum_probs+1e-10, axis=1)
    mask = jnp.triu(jnp.ones((len(sorted_probs), len(sorted_probs)))).T
    normalized_probs *= mask
    
    log_diff = jnp.log(normalized_probs) - jnp.log(jnp.expand_dims(sorted_probs, axis=0))
    
    kl_divs = jnp.sum(jnp.where(mask > 0, normalized_probs * log_diff, 0), axis=1)
    
    optimal_k = jnp.sum(kl_divs >= eps)
    return optimal_k

def find_optimal_topk_stupid(probs, eps):
    print(f"Starting find_optimal_topk_stupid with eps={eps}")
    sorted_probs = jnp.sort(probs)[::-1]  # Sort in descending order
    cumsum_probs = jnp.cumsum(sorted_probs)
    print(f"Cumulative sum of probabilities: {cumsum_probs}")
    
    for i in range(1, len(sorted_probs) + 1):
        truncated_probs = sorted_probs[:i] / cumsum_probs[i-1]
        print(f"Truncated sum of probabilities: {truncated_probs}")
        kl_div = jnp.sum(truncated_probs * (jnp.log(truncated_probs) - jnp.log(sorted_probs[:i])))
        print(f"Iteration {i}: KL divergence = {kl_div}")
        
        if kl_div < eps:
            print(f"Found optimal k: {i}")
            return i
    
    print(f"No suitable k found, returning full length: {len(sorted_probs)}")
    return len(sorted_probs)  # If no suitable k is found, return the full length

def optimal_topk_condition_1(p, q, eps):
    return jnp.sum(p * jnp.log(p/(q+1e-15))) < eps

def optimal_topk_condition_2(p, q, eps):    
    return  jnp.sum(p*jnp.log(p/(q+1e-15)))  < 1 + eps
