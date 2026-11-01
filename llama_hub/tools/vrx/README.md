# VRX Studio Tools and Retriever

This tool integrates **VRX Studio** (`vrx-studio` / `vrx-integrations`) into LlamaIndex / LlamaHub agents and query pipelines.

VRX Studio provides a deterministic neuro-symbolic governance layer for LLMs. While traditional Vector RAG struggles with temporal policy updates, versioning, and compliance hierarchies, VRX Studio utilizes domain-adapted Task adapters with runtime relational graph state injection to deliver:
- 100% discrete mathematical recall without cosine embedding drift
- Zero-retraining instant adaptation across policy mutations ($G_0 \to G_1$)
- Full provenance attribution with cryptographic Merkle audit proofs

## Components

1. **`VRXToolSpec`**:
   - `query_policy(query, version='G1')`: Queries deterministic rules with node-level provenance.
   - `run_acid_mutation_test(query)`: Verifies zero-retraining adaptation by evaluating baseline $G_0$ against mutated $G_1$ live states.
2. **`VRXKnowledgeGraphRetriever`**:
   - Subclasses `BaseRetriever` to return `NodeWithScore` containing deterministic graph facts and Merkle audit hashes.

## Usage

### 1. Function Calling Agent with VRXToolSpec

```python
from llama_hub.tools.vrx import VRXToolSpec
from llama_index.agent import OpenAIAgent

vrx_spec = VRXToolSpec(
    api_token="vrx_live_sk_...",
    domain="healthcare_prior_auth"
)

agent = OpenAIAgent.from_tools(vrx_spec.to_tool_list())

# Query live policy
response = agent.chat("What is the prior-auth window for CPT 70553?")
print(response)

# Run zero-retraining acid mutation test
mutation_check = vrx_spec.run_acid_mutation_test("What is the prior-auth window for CPT 70553?")
print("Zero retraining confirmed:", mutation_check["zero_retraining_mutation_confirmed"])
```

### 2. Deterministic Knowledge Graph Retriever

```python
from llama_hub.tools.vrx import VRXKnowledgeGraphRetriever

retriever = VRXKnowledgeGraphRetriever(
    api_token="vrx_live_sk_...",
    domain="healthcare_prior_auth",
    version="G1"
)

nodes = retriever.retrieve("What is the prior-auth window for CPT 70553?")
for node in nodes:
    print(node.node.text)
    print("Node Provenance ID:", node.node.metadata.get("node_id"))
    print("Merkle Audit Hash:", node.node.metadata.get("merkle_audit_hash"))
```

## Setup

```bash
pip install vrx-integrations
export VRX_API_TOKEN="your_vrx_token"
```
