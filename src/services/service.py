"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls shared/services/ for external integrations.

from __future__ import annotations

from typing import Any


class Service:
    """Domain service seam — unused in this deterministic version.

    This template performs no external calls (retrieval runs over the seeded
    local KB), so no node constructs this class. It is the documented
    integration seam: a live-data upgrade implements fetch() here and calls
    it from a node, leaving the node/state contracts unchanged.
    """

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query."""
        raise NotImplementedError("Service.fetch() is not used by this deterministic version")
