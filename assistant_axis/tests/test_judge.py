import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_axis.judge import RateLimiter, call_judge_batch


def _fake_client() -> MagicMock:
    """Client whose chat completion always answers "3"; the API itself is external."""
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="3"))])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=reply)
    return client


class TestCallJudgeBatch:
    @property
    def prompts(self) -> list:
        return ["prompt a", "prompt b", "prompt c"]

    @pytest.mark.parametrize(
        "extra_body",
        [
            None,
            {
                "provider": {"order": ["z-ai"], "allow_fallbacks": False},
                "reasoning": {"enabled": False},
            },
        ],
    )
    def test_extra_body_reaches_every_request(self, extra_body: Optional[Dict[str, Any]]) -> None:
        client = _fake_client()
        results = asyncio.run(
            call_judge_batch(
                client=client,
                prompts=self.prompts,
                model="z-ai/glm-5.3-flash",
                max_tokens=10,
                rate_limiter=RateLimiter(1000),
                batch_size=2,
                extra_body=extra_body,
            )
        )

        assert results == ["3", "3", "3"]
        calls = client.chat.completions.create.call_args_list
        assert len(calls) == len(self.prompts)
        assert all(call.kwargs["extra_body"] == extra_body for call in calls)
