import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock torch and heavy dependencies BEFORE importing anything
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

sys.modules["transformers"] = MagicMock()
sys.modules["duckduckgo_search"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["pyttsx3"] = MagicMock()
sys.modules["xxhash"] = MagicMock()

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now import targets
from remember_me.integrations.agent import SovereignAgent
from remember_me.core.csnp import CSNPManager
from remember_me.core.embedder import LocalEmbedder

class TestBoltOptimizations(unittest.TestCase):

    def setUp(self):
        # Mocks
        self.engine = MagicMock()
        self.tools = MagicMock()

    def test_sovereign_agent_threadpool_reuse(self):
        """
        Verify that SovereignAgent reuses its ThreadPoolExecutor.
        """
        agent = SovereignAgent(self.engine, self.tools)

        # Verify executor exists (after refactor)
        if not hasattr(agent, '_executor'):
            self.fail("SovereignAgent should have _executor attribute")

        executor_first = agent._executor

        # Simulate run with IMAGE intent
        # We need to mock _detect_intents because we mocked re and patterns are there,
        # but regex compiled on imported re might fail if re is not mocked but used?
        # Actually standard lib re is fine.
        # But we can just mock _detect_intents for simplicity.
        agent._detect_intents = MagicMock(return_value=["IMAGE"])
        agent.engine.generate_response.return_value = "Response"
        self.tools.generate_image.return_value = "Image Generated"

        # We assume run() submits to executor.
        # Since we mocked concurrent.futures inside the module if we could...
        # But we are testing the structure.

        # NOTE: agent._executor will be a real ThreadPoolExecutor if we don't mock it?
        # But concurrent.futures is standard lib.
        # So it will create real threads. This is fine for 1 test.

        agent.run("draw cat", "context")

        executor_second = agent._executor

        self.assertIs(executor_first, executor_second, "Executor should be reused")

        # Verify it wasn't shutdown
        future = agent._executor.submit(lambda: True)
        self.assertTrue(future.result(), "Executor should still be active")

        if hasattr(agent, 'shutdown'):
            agent.shutdown()

    @patch('remember_me.core.csnp.torch.load')
    def test_csnp_load_state_device_enforcement(self, mock_load):
        """
        Verify that load_state forces tensors to the configured device.
        """
        target_device = 'cpu'

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.device = target_device

        manager = CSNPManager(embedder=mock_embedder)

        # Simulate loaded tensor
        loaded_tensor = MagicMock()
        loaded_tensor.device = 'cuda:0'
        # Need to allow shape access for validation
        loaded_tensor.shape = [5, 384]
        loaded_tensor.__len__.return_value = 5

        # Mock .to()
        converted_tensor = MagicMock()
        converted_tensor.device = target_device
        loaded_tensor.to.return_value = converted_tensor

        # Fix formatting for print statement: self.identity_state.norm().item()
        # The code prints {item:.4f}. So item() must return a float.
        # loaded_tensor is used as identity_state.
        # But we assign state_dict["identity_state"].to(self.device) to self.identity_state.
        # So self.identity_state will be converted_tensor.
        converted_tensor.norm.return_value.item.return_value = 1.0
        # Also need shape for loaded_size check
        converted_tensor.shape = [5, 384]

        state_dict = {
            "memory_bank": loaded_tensor,
            "memory_norms": loaded_tensor,
            "identity_state": loaded_tensor,
            "text_buffer": ["a"] * 5,
            "chain_data": ["a"] * 5,
            "config": {"dim": 384, "context_limit": 50}
        }
        mock_load.return_value = state_dict

        with patch('os.path.exists', return_value=True):
            manager.load_state("fake.pt")

        # Verify loaded tensor was moved to target device
        # This asserts that we CALLED .to('cpu') on the object we got from state_dict
        loaded_tensor.to.assert_any_call(target_device)

if __name__ == "__main__":
    unittest.main()
