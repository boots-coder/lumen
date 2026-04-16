"""
Concurrent tool executor — read parallel, write serial, batched dispatch.

Mirrors src/services/tools/toolOrchestration.ts + StreamingToolExecutor.ts:
  - Partitions tool calls by concurrency safety
  - Read-only tools execute in parallel (max 10)
  - Write tools execute serially
  - Results returned in original order
  - Abort propagation via asyncio.Event
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_CONCURRENT_TOOLS = 10

# Tools that are safe to run concurrently (read-only)
_READ_ONLY_TOOLS = frozenset({
    "read_file", "glob", "grep", "tree", "definitions",
})

# Tools that MUST run serially (write / side-effect)
_WRITE_TOOLS = frozenset({
    "write_file", "edit_file", "bash",
})


# ── Types ────────────────────────────────────────────────────────────────────

class ToolStatus(str, Enum):
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ToolExecResult:
    """Result of a single tool execution."""
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool
    status: ToolStatus
    duration_ms: float = 0.0


@dataclass
class ToolExecRequest:
    """A tool call to be executed."""
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


# ── Concurrency classification ───────────────────────────────────────────────

def is_concurrency_safe(tool_name: str) -> bool:
    """Check if a tool can be executed concurrently."""
    return tool_name in _READ_ONLY_TOOLS


def partition_tool_calls(
    requests: list[ToolExecRequest],
) -> list[list[ToolExecRequest]]:
    """
    Partition tool calls into batches for execution.

    Consecutive read-only tools → one concurrent batch.
    Each write tool → its own serial batch.

    Returns a list of batches (each batch is a list of requests).
    """
    if not requests:
        return []

    batches: list[list[ToolExecRequest]] = []
    current_batch: list[ToolExecRequest] = []
    current_is_concurrent = True

    for req in requests:
        safe = is_concurrency_safe(req.tool_name)

        if safe:
            if current_is_concurrent:
                current_batch.append(req)
            else:
                # Flush the serial batch, start new concurrent batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [req]
                current_is_concurrent = True
        else:
            # Write tool — flush current batch, add as single-item batch
            if current_batch:
                batches.append(current_batch)
            batches.append([req])
            current_batch = []
            current_is_concurrent = True

    if current_batch:
        batches.append(current_batch)

    return batches


# ── Executor ─────────────────────────────────────────────────────────────────

ToolExecuteFn = Callable[[str, dict[str, Any]], Any]


class ToolExecutor:
    """
    Executes tool calls with concurrency control.

    Read-only tools run in parallel (up to MAX_CONCURRENT_TOOLS).
    Write tools run serially, one at a time.
    Supports abort via asyncio.Event.
    """

    def __init__(
        self,
        execute_fn: ToolExecuteFn,
        abort_event: asyncio.Event | None = None,
        max_concurrent: int = MAX_CONCURRENT_TOOLS,
    ) -> None:
        """
        Parameters
        ----------
        execute_fn : callable
            async fn(tool_name, arguments) → ToolResult
        abort_event : asyncio.Event | None
            If set, signals abort. Checked before each tool execution.
        max_concurrent : int
            Max number of concurrent read-only tools.
        """
        self._execute_fn = execute_fn
        self._abort = abort_event
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_batch(
        self,
        requests: list[ToolExecRequest],
    ) -> list[ToolExecResult]:
        """
        Execute a list of tool calls with proper batching.

        Returns results in the SAME ORDER as requests.
        """
        batches = partition_tool_calls(requests)
        results: list[ToolExecResult] = []

        for batch in batches:
            if self._is_aborted():
                # Fill remaining with aborted results
                for req in batch:
                    results.append(ToolExecResult(
                        tool_call_id=req.tool_call_id,
                        tool_name=req.tool_name,
                        output="Aborted",
                        is_error=True,
                        status=ToolStatus.ABORTED,
                    ))
                continue

            if len(batch) == 1:
                # Serial execution (write tool or single read)
                r = await self._execute_one(batch[0])
                results.append(r)
            else:
                # Concurrent execution (read-only batch)
                batch_results = await self._execute_concurrent(batch)
                results.extend(batch_results)

        return results

    async def _execute_concurrent(
        self,
        batch: list[ToolExecRequest],
    ) -> list[ToolExecResult]:
        """Execute a batch of read-only tools concurrently."""
        async def _run(req: ToolExecRequest) -> ToolExecResult:
            async with self._semaphore:
                return await self._execute_one(req)

        tasks = [asyncio.create_task(_run(req)) for req in batch]
        return list(await asyncio.gather(*tasks))

    async def _execute_one(self, req: ToolExecRequest) -> ToolExecResult:
        """Execute a single tool call."""
        if self._is_aborted():
            return ToolExecResult(
                tool_call_id=req.tool_call_id,
                tool_name=req.tool_name,
                output="Aborted",
                is_error=True,
                status=ToolStatus.ABORTED,
            )

        start = time.monotonic()
        try:
            result = await self._execute_fn(req.tool_name, req.arguments)
            duration = (time.monotonic() - start) * 1000

            if result.success:
                return ToolExecResult(
                    tool_call_id=req.tool_call_id,
                    tool_name=req.tool_name,
                    output=result.output,
                    is_error=False,
                    status=ToolStatus.COMPLETED,
                    duration_ms=duration,
                )
            else:
                return ToolExecResult(
                    tool_call_id=req.tool_call_id,
                    tool_name=req.tool_name,
                    output=result.error or "Tool failed",
                    is_error=True,
                    status=ToolStatus.FAILED,
                    duration_ms=duration,
                )

        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.error("Tool %s threw: %s", req.tool_name, e, exc_info=True)
            return ToolExecResult(
                tool_call_id=req.tool_call_id,
                tool_name=req.tool_name,
                output=str(e),
                is_error=True,
                status=ToolStatus.FAILED,
                duration_ms=duration,
            )

    def _is_aborted(self) -> bool:
        return self._abort is not None and self._abort.is_set()
