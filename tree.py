from dataclasses import dataclass, field
from typing import List, Optional
import jax.numpy as jnp

class Node:
    def __init__(self, token_ids, parent=None):
        self.token_ids = jnp.array(token_ids, dtype=jnp.int32)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.logprob = 0.0
        self.entropy = 0
        self.optimal_topk = 0
        self.depth = 0 if parent is None else parent.depth + 1

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

class Tree:
    def __init__(self, root_token_ids):
        self.root = Node(root_token_ids)