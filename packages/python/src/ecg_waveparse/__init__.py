"""ECG WaveParse: local digitization using one shared engine."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, NotRequired, TypedDict

__version__ = "0.1.0"


class RunAsset(TypedDict):
    label: str
    path: str
    absolutePath: str
    mimeType: NotRequired[str]
    sizeBytes: NotRequired[int]
    identity: NotRequired[dict[str, Any]]


class RunResult(TypedDict):
    schemaVersion: int
    id: str
    status: str
    source: str
    fileName: str
    createdAt: str
    updatedAt: str
    localPath: str
    sourceIdentity: NotRequired[dict[str, Any]]
    captureQuality: NotRequired[dict[str, Any]]
    digitizer: NotRequired[dict[str, Any]]
    qa: NotRequired[dict[str, Any]]
    waveparse: dict[str, Any]
    assets: dict[str, RunAsset]
    message: NotRequired[str]
    layout: NotRequired[str]
    layoutCost: NotRequired[float]
    sampleRateHz: NotRequired[int]
    effectiveSampleRateHz: NotRequired[float]
    paperSpeedMmPerSecond: NotRequired[float]
    gainMmPerMv: NotRequired[float]
    selectedCandidateId: NotRequired[str]
    publicationDecision: NotRequired[dict[str, Any]]
    reliability: NotRequired[dict[str, Any]]
    review: NotRequired[dict[str, Any]]
    processing: NotRequired[dict[str, Any]]
    retention: NotRequired[dict[str, Any]]


class WaveParseError(RuntimeError):
    def __init__(self, code: str, message: str, run_id: str | None = None):
        super().__init__(message)
        self.code, self.run_id = code, run_id


def _resources() -> Path:
    return Path(__file__).resolve().parent / "runtime"


def _node() -> str:
    value = os.environ.get("WAVEPARSE_NODE") or shutil.which("node")
    if not value:
        raise WaveParseError("runtime_unavailable", "Node.js 22.22.2 is required; set WAVEPARSE_NODE to its executable.")
    return value


def _decode(stdout: bytes, stderr: bytes, code: int | None) -> Any:
    run_id, final = None, None
    try:
        for line in stdout.decode().splitlines():
            event = json.loads(line)
            if event["type"] == "started":
                run_id = event["runId"]
            elif event["type"] in ("result", "error"):
                final = event
            else:
                raise ValueError("Unknown runner event")
    except (ValueError, KeyError, UnicodeError) as error:
        raise WaveParseError("protocol_error", str(error), run_id) from error
    if final and "error" in final:
        error = final["error"]
        raise WaveParseError(error["code"], error["message"], error.get("runId"))
    if code != 0 or final is None or "result" not in final:
        raise WaveParseError("runner_failed", stderr.decode(errors="replace")[-8192:] or f"Runner exited with code {code}.", run_id)
    return final["result"]


class Digitizer:
    """A workspace-scoped engine. Imports never download files or start processes."""

    def __init__(self, *, workspace_dir: str | Path, runtime_dir: str | Path | None = None):
        if not str(workspace_dir):
            raise WaveParseError("invalid_request", "workspace_dir is required.")
        self.workspace_dir = str(Path(workspace_dir).resolve())
        self.runtime_dir = str(Path(runtime_dir).resolve()) if runtime_dir else None

    def _invocation(self, operation: str, **kwargs: Any) -> tuple[list[str], dict[str, str], bytes]:
        root = _resources()
        request = {"protocolVersion": 1, "operation": operation, "workspaceDir": self.workspace_dir, **kwargs}
        if self.runtime_dir:
            request["runtimeDir"] = self.runtime_dir
        env = {**os.environ, "WAVEPARSE_RESOURCE_ROOT": str(root), "WAVEPARSE_WORKSPACE_ROOT": self.workspace_dir}
        return [_node(), str(root / "runner.cjs")], env, (json.dumps(request) + "\n").encode()

    def _request(self, operation: str, **kwargs: Any) -> Any:
        command, env, data = self._invocation(operation, **kwargs)
        try:
            process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        except OSError as error:
            raise WaveParseError("runtime_unavailable", str(error)) from error
        try:
            stdout, stderr = process.communicate(data)
        except BaseException:
            process.terminate()
            try:
                process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
        return _decode(stdout, stderr, process.returncode)

    async def _request_async(self, operation: str, **kwargs: Any) -> Any:
        command, env, data = self._invocation(operation, **kwargs)
        try:
            process = await asyncio.create_subprocess_exec(*command, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
        except OSError as error:
            raise WaveParseError("runtime_unavailable", str(error)) from error
        communication = asyncio.create_task(process.communicate(data))
        try:
            stdout, stderr = await asyncio.shield(communication)
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(communication), 30)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await communication
            raise
        return _decode(stdout, stderr, process.returncode)

    def digitize(self, input_path: str | Path, *, device: str = "cpu", timeout_ms: int = 1_800_000) -> RunResult:
        return self._request("digitize", inputPath=str(Path(input_path).resolve()), device=device, timeoutMs=timeout_ms)

    async def digitize_async(self, input_path: str | Path, *, device: str = "cpu", timeout_ms: int = 1_800_000) -> RunResult:
        return await self._request_async("digitize", inputPath=str(Path(input_path).resolve()), device=device, timeoutMs=timeout_ms)

    def get_run(self, run_id: str) -> RunResult | None:
        return self._request("get", runId=run_id)

    async def get_run_async(self, run_id: str) -> RunResult | None:
        return await self._request_async("get", runId=run_id)

    def review(self, run_id: str, *, decision: str, reviewer: str, notes: str, confirmations: dict[str, bool] | None = None) -> RunResult:
        return self._request("review", runId=run_id, decision=decision, reviewer=reviewer, notes=notes, confirmations=confirmations)

    async def review_async(self, run_id: str, *, decision: str, reviewer: str, notes: str, confirmations: dict[str, bool] | None = None) -> RunResult:
        return await self._request_async("review", runId=run_id, decision=decision, reviewer=reviewer, notes=notes, confirmations=confirmations)
