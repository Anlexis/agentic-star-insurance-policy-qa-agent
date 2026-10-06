"""AgentCore Platform v1.0"""

# AgentRegistry auto-discovery entry point (config/agent.yaml module: src.graph,
# class: InsurancePolicyQAAgent) — must expose the manifest class here.
from .graph import InsurancePolicyQAAgent

__all__ = ["InsurancePolicyQAAgent"]
