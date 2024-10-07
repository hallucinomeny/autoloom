import asyncio
import random
from PyQt6.QtCore import QThread, pyqtSignal
from tree import Node, Tree
import math
from jax.scipy.special import entr
import jax
import jax.numpy as jnp
import psutil
import os
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCTSWorker(QThread):
    finished = pyqtSignal()
    update_ui_signal = pyqtSignal(list, int)
    performance_signal = pyqtSignal(dict)

    def __init__(self, language_model, num_workers=4):
        super().__init__()
        self.language_model = language_model
        self.tree = None
        self.running = False
        self.temperature = 1.0
        self.min_prob = 1e-6
        self.entropy_factor = 3.0 
        self.eps = 0.01
        self.num_workers = num_workers
        self.lock = asyncio.Lock()
        self.performance_data = self.reset_performance_data()
        self.paused = False
        self.pause_condition = asyncio.Condition()

    def set_params(self, prompt_token_ids, temperature, min_prob, entropy_factor):
        self.prompt_token_ids = prompt_token_ids
        self.temperature = temperature
        self.min_prob = min_prob
        self.entropy_factor = entropy_factor
        # Removed eps

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.language_model.set_event_loop(loop)
        loop.run_until_complete(self.run_async())

    async def run_async(self):
        try:
            logger.info("Starting MCTS run_async")
            self.reset_performance_data()
            self.tree = Tree(self.prompt_token_ids)
            iteration = 0
            process = psutil.Process(os.getpid())
            self.running = True
            last_update_time = 0
            update_interval = 0.5  # Update UI every 0.5 seconds
            
            start_time = time.time()
            while self.running:
                async with self.pause_condition:
                    while self.paused:
                        await self.pause_condition.wait()
                
                iteration_start = time.time()
                logger.debug(f"Starting iteration {iteration}")
                await self.mcts_iteration()
                
                iteration_time = time.time() - iteration_start
                logger.debug(f"Iteration {iteration} completed in {iteration_time:.4f} seconds")
                
                self.performance_data["iterations"] += 1
                self.performance_data["total_time"] += iteration_time
                
                # Limit UI updates
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    paths = self.get_all_leaf_paths(self.tree.root)
                    total_nodes = self.count_nodes(self.tree.root)
                    logger.debug(f"Emitting update signal: {len(paths)} paths, {total_nodes} total nodes")
                    self.update_ui_signal.emit(paths, total_nodes)
                    last_update_time = current_time
                    
                    # Calculate average times
                    for key in ["selection", "expansion", "simulation", "backpropagation", "language_model"]:
                        count_key = f"{key}_count"
                        if count_key in self.performance_data and self.performance_data[count_key] > 0:
                            avg_time = self.performance_data.get(f"{key}_time", 0) / self.performance_data[count_key]
                            self.performance_data[f"avg_{key}_time"] = avg_time
                    
                    # Emit performance data
                    self.performance_signal.emit(self.performance_data)
                
                # Check memory usage
                memory_percent = process.memory_percent()
                if memory_percent > 80:
                    logger.warning(f"Memory usage reached {memory_percent:.2f}%. Stopping MCTS.")
                    break
                
                iteration += 1
                await asyncio.sleep(0.01)  # Add a small delay between iterations
            
            end_time = time.time()
            logger.info(f"MCTS completed in {end_time - start_time:.2f} seconds")
            self.finished.emit()
        except Exception as e:
            logger.error(f"Error in MCTS process: {e}", exc_info=True)
            self.finished.emit()

    def reset(self):
        self.running = False
        self.tree = None

    async def mcts_iteration(self):
        logger.debug("Starting MCTS iteration")
        leaf_nodes = []
        for _ in range(self.num_workers):
            leaf_node = await self.select(self.tree.root)
            if not leaf_node.children:
                leaf_nodes.append(leaf_node)
        
        logger.debug(f"Selected {len(leaf_nodes)} leaf nodes")
        
        if leaf_nodes:
            logger.debug("Expanding leaf nodes")
            await self.expand_batch(leaf_nodes)
        
        for leaf_node in leaf_nodes:
            logger.debug("Simulating and backpropagating")
            value = await self.simulate(leaf_node)
            await self.backpropagate(leaf_node, value)
        
        logger.debug("MCTS iteration completed")

    async def select(self, node):
        async with self.lock:
            start_time = time.time()
            while node.children:
                if not all(child.visits > 0 for child in node.children):
                    node = random.choice([child for child in node.children if child.visits == 0])
                else:
                    ucb_values = [
                        (child.value / child.visits) + math.sqrt(2 * math.log(node.visits) / child.visits)
                        for child in node.children
                    ]
                    node = node.children[ucb_values.index(max(ucb_values))]
            self.performance_data["selection_time"] += time.time() - start_time
            self.performance_data["select_count"] += 1
            return node

    async def expand_batch(self, nodes):
        states = [self.get_state(node) for node in nodes]
        results = await self.language_model.extract_distribution_batch(
            states, 
            temperature=self.temperature, 
            min_prob=self.min_prob, 
            entropy_factor=self.entropy_factor
        )
        
        for node, result in zip(nodes, results):
            if result is None:
                continue
            distribution, entropy, raw_sum = result
            node.entropy = entropy
            node.raw_sum = raw_sum  # Store the raw sum in the node
            
            for token_id, prob in distribution.items():
                child = Node([int(token_id)], parent=node)
                child.logprob = jnp.log(prob)
                node.add_child(child)
            
            node.child_count = len(node.children)

    async def simulate(self, node):
        start_time = time.time()
        value = node.logprob if node.parent and not jnp.isinf(node.logprob) else 0.0
        self.performance_data["simulation_time"] += time.time() - start_time
        self.performance_data["simulate_count"] += 1
        return value

    async def backpropagate(self, node, value):
        start_time = time.time()
        async with self.lock:
            while node:
                node.visits += 1
                node.value += math.exp(value) if not jnp.isinf(value) else 0
                node = node.parent
        self.performance_data["backpropagation_time"] += time.time() - start_time
        self.performance_data["backpropagate_count"] += 1

    def get_all_leaf_paths(self, root):
        paths = []
        self.dfs_parents(root, [], 0.0, paths)
        return sorted(paths, key=lambda x: x[1], reverse=True)

    def get_state(self, node):
        path = []
        current = node
        while current:
            path.append(current.token_ids)
            current = current.parent
        return jnp.concatenate(path[::-1])  # Reverse the path and concatenate

    def dfs_parents(self, node, current_path, current_logprob, paths):
        new_path = current_path + list(node.token_ids)
        new_logprob = current_logprob + (node.logprob if node != self.tree.root and not jnp.isinf(node.logprob) else 0.0)

        if node.children and all(not child.children for child in node.children):
            child_data = [(int(child.token_ids[0]), math.exp(child.logprob), child.logprob) for child in node.children]
            child_data.sort(key=lambda x: x[1], reverse=True)
            paths.append((new_path, math.exp(new_logprob) if not jnp.isinf(new_logprob) else 0, 
                          node.entropy, node.depth, node.raw_sum, len(node.children), child_data))
        else:
            for child in node.children:
                self.dfs_parents(child, new_path, new_logprob, paths)

    def count_nodes(self, node):
        return 1 + sum(self.count_nodes(child) for child in node.children)

    def reset_performance_data(self):
        return {
            "iterations": 0,
            "total_time": 0,
            "selection_time": 0,
            "expansion_time": 0,
            "simulation_time": 0,
            "backpropagation_time": 0,
            "language_model_time": 0,
            "select_count": 0,
            "expand_count": 0,
            "simulate_count": 0,
            "backpropagate_count": 0,
            "language_model_count": 0,
            "selection_count": 0,  # Add this line
        }

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        asyncio.run_coroutine_threadsafe(self.pause_condition.notify_all(), self.loop)