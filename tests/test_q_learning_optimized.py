"""Tests for the optimized Q-Learning implementation."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from blokus_env.q_learning import QLearningAgent
from blokus_env.q_learning_optimized import QLearningAgentOptimized
from blokus_env.blokus_env import BlokusEnv


def test_optimized_initialization():
    """Test optimized Q-Learning agent initialization."""
    agent = QLearningAgentOptimized()

    assert agent.learning_rate == 0.1
    assert agent.discount_factor == 0.9
    assert agent.exploration_rate == 1.0
    assert agent.exploration_decay == 0.995
    assert agent.min_exploration == 0.01
    assert agent.episodes == 0
    assert agent.wins == 0
    assert agent.losses == 0
    assert agent.ties == 0

    # Check that caches are initialized
    assert hasattr(agent, '_state_cache')
    assert hasattr(agent, '_legal_actions_cache')
    assert len(agent._state_cache) == 0
    assert len(agent._legal_actions_cache) == 0

    print("Optimized Q-Learning initialization test passed!")


def test_optimized_state_hashing():
    """Test optimized state hashing functionality."""
    agent = QLearningAgentOptimized()
    env = BlokusEnv()

    # Test with initial state
    state, _ = env.reset()
    state_hash = agent.get_state_hash(state, 0)

    assert isinstance(state_hash, str)
    assert len(state_hash) > 0

    # Test caching
    state_hash2 = agent.get_state_hash(state, 0)
    assert state_hash == state_hash2  # Should be cached

    # Check cache stats
    stats = agent.get_cache_stats()
    assert stats['cache_hits'] >= 1
    assert stats['total_state_hashes'] >= 2

    print("Optimized state hashing test passed!")


def test_optimized_legal_actions():
    """Test optimized legal actions detection."""
    agent = QLearningAgentOptimized()
    env = BlokusEnv()
    state, _ = env.reset()

    legal_actions = agent.get_legal_actions(env, 0)

    assert isinstance(legal_actions, list)
    assert len(legal_actions) > 0  # Should have legal first moves

    # Check action format
    if legal_actions:
        action = legal_actions[0]
        assert isinstance(action, tuple)
        assert len(action) == 4
        assert all(isinstance(x, int) for x in action)

    # Test caching
    legal_actions2 = agent.get_legal_actions(env, 0)
    assert legal_actions == legal_actions2  # Should be cached

    # Check cache stats
    stats = agent.get_cache_stats()
    assert stats['cache_hits'] >= 1
    assert stats['total_legal_action_calls'] >= 2

    print("Optimized legal actions test passed!")


def test_optimized_action_selection():
    """Test optimized action selection (exploration vs exploitation)."""
    agent = QLearningAgentOptimized()
    env = BlokusEnv()
    state, _ = env.reset()

    # Test exploration (should return random action)
    agent.exploration_rate = 1.0  # Force exploration
    action = agent.choose_action(env, state, 0, training=True)
    assert action is not None
    assert isinstance(action, tuple)

    # Test exploitation (should return best action or random if no Q-values)
    agent.exploration_rate = 0.0  # Force exploitation
    action = agent.choose_action(env, state, 0, training=False)
    assert action is not None
    assert isinstance(action, tuple)

    print("Optimized action selection test passed!")


def test_performance_comparison():
    """Compare performance between original and optimized versions."""
    print("\n=== Performance Comparison Test ===")

    # Create both agents
    original_agent = QLearningAgent()
    optimized_agent = QLearningAgentOptimized()
    
    # Create environment
    env = BlokusEnv()
    state, _ = env.reset()

    # Test state hashing performance
    print("Testing state hashing performance...")
    
    # Original
    start_time = time.time()
    for i in range(100):
        for player in range(4):
            state_hash = original_agent.get_state_hash(state, player)
    original_hash_time = time.time() - start_time

    # Optimized
    start_time = time.time()
    for i in range(100):
        for player in range(4):
            state_hash = optimized_agent.get_state_hash(state, player)
    optimized_hash_time = time.time() - start_time

    print(f"Original state hashing: {original_hash_time:.3f}s")
    print(f"Optimized state hashing: {optimized_hash_time:.3f}s")
    print(f"Improvement: {(1 - optimized_hash_time/original_hash_time) * 100:.1f}%")

    # Test legal actions performance
    print("\nTesting legal actions performance...")
    
    # Original
    start_time = time.time()
    for i in range(10):
        for player in range(4):
            legal_actions = original_agent.get_legal_actions(env, player)
    original_legal_time = time.time() - start_time

    # Optimized
    start_time = time.time()
    for i in range(10):
        for player in range(4):
            legal_actions = optimized_agent.get_legal_actions(env, player)
    optimized_legal_time = time.time() - start_time

    print(f"Original legal actions: {original_legal_time:.3f}s")
    print(f"Optimized legal actions: {optimized_legal_time:.3f}s")
    print(f"Improvement: {(1 - optimized_legal_time/original_legal_time) * 100:.1f}%")

    # Test full action selection cycle
    print("\nTesting full action selection cycle...")
    
    # Original
    start_time = time.time()
    for i in range(5):
        for player in range(4):
            legal_actions = original_agent.get_legal_actions(env, player)
            if legal_actions:
                action = original_agent.choose_action(env, state, player, training=False)
    original_cycle_time = time.time() - start_time

    # Optimized
    start_time = time.time()
    for i in range(5):
        for player in range(4):
            legal_actions = optimized_agent.get_legal_actions(env, player)
            if legal_actions:
                action = optimized_agent.choose_action(env, state, player, training=False)
    optimized_cycle_time = time.time() - start_time

    print(f"Original action cycle: {original_cycle_time:.3f}s")
    print(f"Optimized action cycle: {optimized_cycle_time:.3f}s")
    print(f"Improvement: {(1 - optimized_cycle_time/original_cycle_time) * 100:.1f}%")

    print("\n✓ Performance comparison test completed!")


def test_cache_functionality():
    """Test cache functionality in optimized version."""
    agent = QLearningAgentOptimized()
    env = BlokusEnv()
    state, _ = env.reset()

    # Clear caches
    agent.clear_caches()
    
    # First call should miss cache
    legal_actions1 = agent.get_legal_actions(env, 0)
    stats1 = agent.get_cache_stats()
    
    # Second call should hit cache
    legal_actions2 = agent.get_legal_actions(env, 0)
    stats2 = agent.get_cache_stats()

    assert legal_actions1 == legal_actions2
    assert stats2['cache_hits'] > stats1['cache_hits']
    assert stats2['cache_misses'] == stats1['cache_misses']

    print("Cache functionality test passed!")


def test_optimized_training():
    """Test optimized training for a few steps."""
    agent = QLearningAgentOptimized(
        learning_rate=0.2,
        discount_factor=0.8,
        exploration_rate=1.0,
        exploration_decay=0.9,
        min_exploration=0.1,
    )

    # Train for just 2 episodes with very few steps
    agent.train(episodes=2, max_steps=3)

    assert agent.episodes == 2
    assert agent.wins + agent.losses + agent.ties == 2

    # Check that caches were used
    stats = agent.get_cache_stats()
    assert stats['cache_hits'] > 0 or stats['cache_misses'] > 0

    print("Optimized training test passed!")


if __name__ == "__main__":
    test_optimized_initialization()
    test_optimized_state_hashing()
    test_optimized_legal_actions()
    test_optimized_action_selection()
    test_cache_functionality()
    test_optimized_training()
    test_performance_comparison()
    print("\n🎉 All optimized Q-Learning tests passed!")