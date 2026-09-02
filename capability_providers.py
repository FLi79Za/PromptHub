"""Privileged local capability providers for PromptHub Skills.

Providers are configured records, not instructions supplied by imported Skills.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from generation_runtime import ComfyUIClient, GenerationError, sha256_json, validate_api_workflow


class ProviderError(Exception):
    def __init__(self, message: str, code: str = "PROVIDER_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.details = code, details or {}


class CapabilityProvider(ABC):
    provider_id: str
    capability: str

    @abstractmethod
    def status(self) -> dict[str, Any]: ...

    @abstractmethod
    def validate_request(self, request: dict[str, Any]) -> None: ...

    @abstractmethod
    def execute(self, request: dict[str, Any]) -> dict[str, Any]: ...


class ComfyUIProvider(CapabilityProvider):
    provider_id = "comfyui"
    capability = "image_generation"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_url = str(config.get("server_url") or "http://127.0.0.1:8188").rstrip("/")

    def status(self) -> dict[str, Any]:
        if not self.config.get("enabled", False):
            return {"available": False, "state": "disabled", "provider": self.provider_id}
        health = ComfyUIClient(self.base_url, timeout=float(self.config.get("health_timeout", 3))).health()
        return {**health, "available": bool(health["reachable"]), "state": "ready" if health["reachable"] else "unreachable",
                "provider": self.provider_id, "server_url": self.base_url}

    def _workflow(self) -> tuple[dict[str, Any], Path]:
        path = Path(str(self.config.get("workflow_path") or ""))
        if not path.is_file():
            raise ProviderError("Configured ComfyUI API workflow was not found.", "PROVIDER_CONFIGURATION_ERROR", {"workflow_path": str(path)})
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderError("Configured ComfyUI workflow is not valid JSON.", "PROVIDER_CONFIGURATION_ERROR") from exc
        try:
            validate_api_workflow(graph)
        except GenerationError as exc:
            raise ProviderError(str(exc), "PROVIDER_CONFIGURATION_ERROR", exc.details) from exc
        return graph, path

    def validate_request(self, request: dict[str, Any]) -> None:
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ProviderError("image_generation requires a prompt.", "INVALID_CAPABILITY_REQUEST")
        self._workflow()
        mappings = self.config.get("input_mappings") or {}
        if not isinstance(mappings, dict) or "prompt" not in mappings:
            raise ProviderError("ComfyUI provider requires a prompt input mapping.", "PROVIDER_CONFIGURATION_ERROR")

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        self.validate_request(request)
        readiness = self.status()
        if not readiness["available"]:
            code = "PROVIDER_DISABLED" if readiness["state"] == "disabled" else "COMFYUI_UNREACHABLE"
            raise ProviderError("ComfyUI provider is not ready.", code, readiness)
        graph, path = self._workflow()
        graph = copy.deepcopy(graph)
        values = {"prompt": request["prompt"], **(request.get("parameters") or {})}
        for key, mapping in (self.config.get("input_mappings") or {}).items():
            if key not in values or values[key] in (None, ""):
                continue
            if not isinstance(mapping, dict) or not mapping.get("node_id") or not mapping.get("input"):
                raise ProviderError(f"Invalid mapping for {key}.", "PROVIDER_CONFIGURATION_ERROR")
            node_id, input_name = str(mapping["node_id"]), str(mapping["input"])
            if node_id not in graph:
                raise ProviderError(f"Configured node {node_id} is absent from workflow.", "PROVIDER_CONFIGURATION_ERROR")
            graph[node_id].setdefault("inputs", {})[input_name] = values[key]
        client_id = str(uuid.uuid4())
        digest = sha256_json(graph)
        try:
            prompt_id = ComfyUIClient(self.base_url, timeout=float(self.config.get("submit_timeout", 20))).submit(graph, client_id)
        except GenerationError as exc:
            raise ProviderError("ComfyUI queue request failed.", "COMFYUI_QUEUE_FAILURE", exc.details) from exc
        return {"provider": self.provider_id, "capability": self.capability, "status": "queued", "prompt_id": prompt_id,
                "client_id": client_id, "workflow": str(path), "workflow_hash": digest, "submitted_at": time.time()}


class CapabilityRegistry:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def resolve(self, capability: str) -> CapabilityProvider | None:
        for record in self.records:
            if record.get("capability") == capability and record.get("provider_id") == "comfyui":
                return ComfyUIProvider(record)
        return None

    def report(self, capability: str) -> dict[str, Any]:
        provider = self.resolve(capability)
        if provider is None:
            return {"capability": capability, "status": "UNAVAILABLE", "provider": None}
        state = provider.status()
        return {"capability": capability, "status": "PRESERVED" if state.get("available") else "DEGRADED", "provider": provider.provider_id, "readiness": state}
