from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("gateway.audit")


class AuditLogger:
    def log_chat(self, req, normalized) -> dict[str, Any]:
        record = {
            "model": req.model,
            "stream": req.stream,
            "metadata": req.metadata or {},
            "messages": [m.model_dump() for m in req.messages],
            "assistant_text": normalized.assistant_text,
            "citations": [c.model_dump() for c in normalized.citations],
            "builtin_tool_events": [e.model_dump() for e in normalized.builtin_tool_events],
            "file_search_results": normalized.file_search_results,
            "legacy_steps": normalized.legacy_steps,
            "bridge_executions": [b.model_dump() for b in normalized.bridge_executions],
            "usage": normalized.usage,
        }
        logger.info("audit=%s", json.dumps(record, ensure_ascii=False, default=str))
        return record
