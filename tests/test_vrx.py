import unittest
from unittest.mock import patch
from llama_hub.tools.vrx.base import VRXToolSpec, VRXKnowledgeGraphRetriever
from vrx_integrations.models import VRXInferenceResponse, VRXProvenance


class TestVRXLlamaHub(unittest.TestCase):
    @patch("vrx_integrations.client.VRXClient.query")
    def test_tool_spec_query(self, mock_query):
        mock_query.return_value = VRXInferenceResponse(
            status="success",
            domain="Healthcare Prior-Authorization",
            query="What is the window for CPT 70553?",
            prediction="3",
            unit="business days",
            provenance=VRXProvenance(
                node_id="Rule_UHC_70553_v2",
                graph_snapshot="G1 (Mutated Live State)",
                active_revision=2
            )
        )

        spec = VRXToolSpec(api_token="test_token")
        result = spec.query_policy("What is the window for CPT 70553?")
        self.assertIn("Result: 3 business days", result)
        mock_query.assert_called_once()

    @patch("vrx_integrations.client.VRXClient.acid_test")
    def test_tool_spec_acid_test(self, mock_acid):
        g0 = VRXInferenceResponse(
            status="success",
            query="test",
            prediction="14",
            unit="days",
            provenance=VRXProvenance(node_id="Node_v1")
        )
        g1 = VRXInferenceResponse(
            status="success",
            query="test",
            prediction="3",
            unit="days",
            provenance=VRXProvenance(node_id="Node_v2")
        )
        mock_acid.return_value = (g0, g1)

        spec = VRXToolSpec(api_token="test_token")
        res = spec.run_acid_mutation_test("test")
        self.assertTrue(res["zero_retraining_mutation_confirmed"])
        self.assertEqual(res["baseline_G0"]["prediction"], "14")
        self.assertEqual(res["mutated_G1"]["prediction"], "3")

    @patch("vrx_integrations.client.VRXClient.query")
    def test_knowledge_graph_retriever(self, mock_query):
        mock_query.return_value = VRXInferenceResponse(
            status="success",
            domain="Healthcare Prior-Authorization",
            query="What is the window for CPT 70553?",
            prediction="3",
            unit="business days",
            provenance=VRXProvenance(node_id="Rule_v2", graph_snapshot="G1")
        )

        retriever = VRXKnowledgeGraphRetriever(api_token="test_token")
        nodes = retriever.retrieve("What is the window for CPT 70553?")
        self.assertEqual(len(nodes), 1)
        self.assertIn("Result: 3 business days", nodes[0].node.text)
        self.assertEqual(nodes[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
