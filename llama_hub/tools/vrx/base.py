"""
LlamaIndex (LlamaHub) Tool Specification for VRX Studio.
Exposes deterministic neuro-symbolic reasoning tools to LlamaIndex agents.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from vrx_integrations.client import VRXClient
except ImportError:
    try:
        from vrx_studio.client import VRXClient
    except ImportError:
        VRXClient = None  # type: ignore

try:
    from llama_index.core.tools.tool_spec.base import BaseToolSpec
    from llama_index.core.tools.types import FunctionTool
except ImportError:
    try:
        from llama_index.tools.tool_spec.base import BaseToolSpec
        from llama_index.tools.types import FunctionTool
    except ImportError:
        class BaseToolSpec:  # type: ignore
            """Fallback tool spec base when llama-index is not installed."""
            spec_functions: List[str] = []

            def to_tool_list(self) -> List[Any]:
                return [getattr(self, func) for func in self.spec_functions if hasattr(self, func)]

        FunctionTool = Any  # type: ignore

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
except ImportError:
    try:
        from llama_index.retrievers import BaseRetriever
        from llama_index.schema import NodeWithScore, TextNode
    except ImportError:
        from pydantic import BaseModel, Field

        class TextNode(BaseModel):  # type: ignore
            text: str
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class NodeWithScore(BaseModel):  # type: ignore
            node: TextNode
            score: float = 1.0

        class BaseRetriever(BaseModel):  # type: ignore
            def retrieve(self, query: str, **kwargs: Any) -> List[NodeWithScore]:
                return self._retrieve(query, **kwargs)


class VRXToolSpec(BaseToolSpec):
    """
    VRX Studio Tool Spec for LlamaIndex Agents.

    Provides deterministic policy evaluation and zero-retraining mutation checks.
    """

    spec_functions = ["query_policy", "run_acid_mutation_test"]

    def __init__(
        self,
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        domain: Optional[str] = None,
        default_version: str = "G1",
        _client: Optional[Any] = None
    ):
        if _client:
            self.client = _client
        else:
            if VRXClient is None:
                raise ImportError(
                    "Could not import vrx_integrations python package. "
                    "Please install it with `pip install vrx-integrations`."
                )
            token = api_token or os.environ.get("VRX_API_TOKEN") or os.environ.get("VRX_JWT_TOKEN")
            url = base_url or os.environ.get("VRX_BASE_URL") or "https://api.vrxstudio.ai"
            self.client = VRXClient(api_token=token, base_url=url, default_domain=domain)
        self.domain = domain
        self.default_version = default_version

    def query_policy(self, query: str, version: Optional[str] = None) -> str:
        """
        Query an authoritative enterprise business policy, healthcare code, tariff, or SLA.

        Args:
            query: The domain question or rule condition to check.
            version: 'G1' for live mutated policy or 'G0' for baseline historical policy.

        Returns:
            Authoritative deterministic result with provenance node and revision.
        """
        ver = version or self.default_version
        resp = self.client.query(query=query, version=ver, domain=self.domain, include_provenance=True)
        return resp.format_text()

    def run_acid_mutation_test(self, query: str) -> Dict[str, Any]:
        """
        Verifies zero-retraining adaptation across a policy revision ($G_0 \\to G_1$).

        Args:
            query: The policy condition to check under both baseline and mutated states.

        Returns:
            Dictionary comparing G0 (baseline) and G1 (live mutated) outcomes.
        """
        g0_resp, g1_resp = self.client.acid_test(query=query, domain=self.domain)
        return {
            "query": query,
            "baseline_G0": {
                "prediction": g0_resp.prediction,
                "unit": g0_resp.unit,
                "node_id": g0_resp.provenance.node_id if g0_resp.provenance else None
            },
            "mutated_G1": {
                "prediction": g1_resp.prediction,
                "unit": g1_resp.unit,
                "node_id": g1_resp.provenance.node_id if g1_resp.provenance else None
            },
            "zero_retraining_mutation_confirmed": (g0_resp.prediction != g1_resp.prediction),
            "retraining_cycles": 0
        }


class VRXKnowledgeGraphRetriever(BaseRetriever):
    """
    LlamaIndex Retriever backed by VRX Studio's Versioned Relational Knowledge Graph.
    Ensures 100% discrete recall without embedding cosine drift.
    """

    api_token: Optional[str] = None
    base_url: Optional[str] = None
    domain: Optional[str] = None
    version: str = "G1"
    include_provenance: bool = True
    _client: Optional[Any] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        client = data.get("_client")
        if client:
            object.__setattr__(self, "_client", client)
        else:
            if VRXClient is None:
                raise ImportError(
                    "Could not import vrx_integrations python package. "
                    "Please install it with `pip install vrx-integrations`."
                )
            token = self.api_token or os.environ.get("VRX_API_TOKEN") or os.environ.get("VRX_JWT_TOKEN")
            url = self.base_url or os.environ.get("VRX_BASE_URL") or "https://api.vrxstudio.ai"
            object.__setattr__(self, "_client", VRXClient(api_token=token, base_url=url, default_domain=self.domain))

    def _retrieve(self, query_bundle: Any) -> List[NodeWithScore]:
        """Synchronously fetch deterministic facts from VRX Studio."""
        query_str = query_bundle.query_str if hasattr(query_bundle, "query_str") else str(query_bundle)
        client = getattr(self, "_client", None)
        if not client:
            if VRXClient is None:
                raise ImportError(
                    "Could not import vrx_integrations python package. "
                    "Please install it with `pip install vrx-integrations`."
                )
            token = self.api_token or os.environ.get("VRX_API_TOKEN") or os.environ.get("VRX_JWT_TOKEN")
            url = self.base_url or os.environ.get("VRX_BASE_URL") or "https://api.vrxstudio.ai"
            client = VRXClient(api_token=token, base_url=url, default_domain=self.domain)
            object.__setattr__(self, "_client", client)

        resp = client.query(
            query=query_str,
            version=self.version,
            domain=self.domain,
            include_provenance=self.include_provenance
        )

        metadata: Dict[str, Any] = {
            "source": "vrx_studio",
            "domain": resp.domain,
            "version": self.version,
            "prediction": resp.prediction,
            "unit": resp.unit,
            "latency_ms": resp.latency_ms,
            "merkle_audit_hash": resp.merkle_audit_hash
        }
        if resp.provenance:
            metadata["node_id"] = resp.provenance.node_id
            metadata["graph_snapshot"] = resp.provenance.graph_snapshot
            metadata["active_revision"] = resp.provenance.active_revision

        node = TextNode(
            text=resp.format_text(),
            metadata=metadata
        )
        return [NodeWithScore(node=node, score=1.0)]
