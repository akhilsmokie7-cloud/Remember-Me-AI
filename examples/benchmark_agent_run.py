
import time
import unittest
from unittest.mock import MagicMock
from remember_me.integrations.agent import SovereignAgent
from remember_me.integrations.engine import ModelRegistry
from remember_me.integrations.tools import ToolArsenal

class MockEngine(ModelRegistry):
    def generate_response(self, user_input, context_str, system_prompt=None):
        time.sleep(2.0) # Simulate LLM
        return "This is a mock response."

class MockTools(ToolArsenal):
    def generate_image(self, prompt, output_path="output.png"):
        time.sleep(2.0) # Simulate SD
        return "Image Generated"

    def web_search(self, query, max_results=3):
        return "Search Results"

def benchmark():
    engine = MockEngine()
    tools = MockTools()
    agent = SovereignAgent(engine, tools)

    print("--- Benchmarking SovereignAgent.run [IMAGE Intent] ---")
    start = time.time()
    # Trigger IMAGE and SEARCH (but search is fast here)
    response = agent.run("generate an image of a cat", "")
    end = time.time()

    duration = end - start
    print(f"Total Duration: {duration:.4f}s")
    print(f"Expected Baseline (Serial): ~4.0s (2s LLM + 2s SD)")
    print(f"Expected Optimized (Parallel): ~2.0s (max(2s, 2s))")

if __name__ == "__main__":
    benchmark()
