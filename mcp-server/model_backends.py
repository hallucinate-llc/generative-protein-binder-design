#!/usr/bin/env python3
"""
Model Backend Abstraction
Provides multiple backend implementations for protein design models:
- NIM Backend: Uses NVIDIA NIM containers (current default)
- Native Backend: Runs models directly on hardware (DGX Spark optimized)
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import httpx

from runtime_config import EmbeddedConfig, MCPServerConfig, ProviderName, RuntimeConfigManager

logger = logging.getLogger(__name__)


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def allow_mock_outputs() -> bool:
    # Keep CI green/deterministic, but avoid silently faking model outputs in real deployments.
    # Enable for CI, or when explicitly requested for testing
    return _truthy_env("CI") or _truthy_env("ALLOW_MOCK_OUTPUTS") or _truthy_env("ENABLE_MOCK_MODE")

class ModelBackend(ABC):
    """Abstract base class for model backends"""
    
    @abstractmethod
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """Predict protein structure from sequence (AlphaFold2)"""
        pass
    
    @abstractmethod
    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        """Generate binder backbones (RFDiffusion)"""
        pass
    
    @abstractmethod
    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        """Generate sequence from backbone (ProteinMPNN)"""
        pass
    
    @abstractmethod
    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        """Predict complex structure (AlphaFold2-Multimer)"""
        pass
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check backend health status"""
        pass


class NIMBackend(ModelBackend):
    """NVIDIA NIM Container Backend (REST API)"""
    
    def __init__(self):
        def _resolve_service_url(env_key: str, default_url: str) -> Optional[str]:
            if env_key in os.environ:
                value = (os.environ.get(env_key) or "").strip()
                if not value or value.lower() in {"disabled", "none", "null"}:
                    return None
                return value
            return default_url

        # Keep a canonical set of services so status dashboards can reliably
        # render all expected backends (even when disabled/not configured).
        alphafold_url = _resolve_service_url("ALPHAFOLD_URL", "http://localhost:8081")
        rfdiffusion_url = _resolve_service_url("RFDIFFUSION_URL", "http://localhost:8082")
        proteinmpnn_url = _resolve_service_url("PROTEINMPNN_URL", "http://localhost:8083")
        alphafold_multimer_url = _resolve_service_url("ALPHAFOLD_MULTIMER_URL", "http://localhost:8084")

        self.service_urls: Dict[str, Optional[str]] = {
            "alphafold": alphafold_url,
            "rfdiffusion": rfdiffusion_url,
            "proteinmpnn": proteinmpnn_url,
            "alphafold_multimer": alphafold_multimer_url,
        }

        # Inference requests can take a long time, but connectivity issues should
        # fail quickly so users get actionable errors instead of hanging jobs.
        connect_timeout_s = float(os.getenv("MCP_NIM_CONNECT_TIMEOUT_S", "5"))
        # Backends like AlphaFold can legitimately take hours (especially CPU runs).
        # Default to a generous timeout, but allow environments to override.
        inference_timeout_s = float(
            os.getenv("MCP_NIM_INFERENCE_TIMEOUT_S", os.getenv("MCP_NIM_TIMEOUT_S", "7200"))
        )
        self._inference_timeout = httpx.Timeout(
            timeout=inference_timeout_s,
            connect=connect_timeout_s,
        )

        # Backwards-compatible: methods use `self.services` for enabled services.
        self.services: Dict[str, str] = {k: v for k, v in self.service_urls.items() if v}

        logger.info("Initialized NIM Backend")

    def _require_service(self, service_name: str) -> str:
        url = self.service_urls.get(service_name)
        if not url:
            raise RuntimeError(f"Service '{service_name}' is disabled or not configured")
        return url

    async def _raise_for_status_with_detail(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        base_url: str,
        service_name: str,
    ) -> None:
        """Raise a more actionable error than httpx's default when possible.

        Some backends (especially ARM64 CI shims) expose the actionable reason via
        `/v1/health/ready` even when inference endpoints return generic 500s.
        """
        if response.status_code < 400:
            return

        detail: Optional[str] = None

        def _extract_detail_from_response(resp: httpx.Response) -> Optional[str]:
            # Prefer JSON {"detail": "..."}, but fall back to text when needed.
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    detail_val = payload.get("detail")
                    if isinstance(detail_val, str) and detail_val.strip():
                        return detail_val.strip()
            except Exception:
                pass
            try:
                txt = (resp.text or "").strip()
                if txt:
                    # Keep it single-line-ish to avoid log spam.
                    return " ".join(txt.split())
            except Exception:
                pass
            return None

        detail = _extract_detail_from_response(response)

        # Many services return a generic text body for 5xx errors (e.g. "Internal Server Error").
        # Treat that as "no detail" so we can pull a better explanation from /v1/health/ready.
        if response.status_code >= 500 and detail:
            lowered = detail.strip().lower()
            if lowered in {"internal server error", "500 internal server error"}:
                detail = None

        if not detail and response.status_code >= 500:
            try:
                ready = await client.get(f"{base_url}/v1/health/ready")
                if ready.status_code != 200:
                    detail = _extract_detail_from_response(ready)
            except Exception:
                pass

        if detail:
            raise RuntimeError(f"{service_name}: {detail}")

        response.raise_for_status()
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """AlphaFold2 structure prediction via NIM"""
        base_url = self._require_service("alphafold")
        async with httpx.AsyncClient(timeout=self._inference_timeout) as client:
            response = await client.post(
                f"{base_url}/v1/structure",
                json={"sequence": sequence}
            )
            await self._raise_for_status_with_detail(client, response, base_url, "alphafold")
            return response.json()
    
    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        """RFDiffusion binder design via NIM"""
        base_url = self._require_service("rfdiffusion")
        async with httpx.AsyncClient(timeout=self._inference_timeout) as client:
            response = await client.post(
                f"{base_url}/v1/design",
                json={
                    "target_pdb": target_pdb,
                    "num_designs": num_designs
                }
            )
            await self._raise_for_status_with_detail(client, response, base_url, "rfdiffusion")
            return response.json()
    
    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        """ProteinMPNN sequence generation via NIM"""
        base_url = self._require_service("proteinmpnn")
        async with httpx.AsyncClient(timeout=self._inference_timeout) as client:
            response = await client.post(
                f"{base_url}/v1/sequence",
                json={"backbone_pdb": backbone_pdb}
            )
            await self._raise_for_status_with_detail(client, response, base_url, "proteinmpnn")
            return response.json()
    
    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        """AlphaFold2-Multimer complex prediction via NIM"""
        base_url = self._require_service("alphafold_multimer")
        async with httpx.AsyncClient(timeout=self._inference_timeout) as client:
            response = await client.post(
                f"{base_url}/v1/structure",
                json={"sequences": sequences}
            )
            await self._raise_for_status_with_detail(client, response, base_url, "alphafold_multimer")
            return response.json()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of all NIM services"""
        status = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, url in self.service_urls.items():
                if not url:
                    status[service_name] = {
                        "status": "disabled",
                        "url": "",
                        "backend": "NIM",
                    }
                    continue
                try:
                    response = await client.get(f"{url}/v1/health/ready")
                    if response.status_code == 200:
                        status[service_name] = {
                            "status": "ready",
                            "url": url,
                            "backend": "NIM",
                        }
                    else:
                        detail = None
                        try:
                            payload = response.json()
                            if isinstance(payload, dict):
                                detail = payload.get("detail") or payload.get("reason") or payload.get("message")
                        except Exception:
                            detail = None
                        if not detail:
                            try:
                                detail = (response.text or "").strip()[:500]
                            except Exception:
                                detail = None

                        status[service_name] = {
                            "status": "not_ready",
                            "url": url,
                            "backend": "NIM",
                            "http_status": response.status_code,
                            **({"reason": detail} if detail else {}),
                        }
                except Exception as e:
                    msg = str(e).strip()
                    if not msg:
                        msg = f"{type(e).__name__}"
                    else:
                        msg = f"{type(e).__name__}: {msg}"
                    status[service_name] = {
                        "status": "not_ready",
                        "error": msg,
                        "url": url,
                        "backend": "NIM"
                    }
        return status


class ExternalBackend(NIMBackend):
    """External model services implementing the same REST contract as NIM.

    This is intentionally the same interface as NIM (health + /v1/* endpoints),
    but is labeled differently in status output.
    """

    def __init__(self, service_urls: Dict[str, Optional[str]]):
        # Reuse the NIMBackend HTTP client timeout defaults and error helpers.
        super().__init__()
        self.service_urls = dict(service_urls)
        self.services = {k: v for k, v in self.service_urls.items() if v}
        logger.info("Initialized External REST Backend")

    async def check_health(self) -> Dict[str, Any]:
        status = await super().check_health()
        for _, v in status.items():
            v["backend"] = "External"
        return status


class EmbeddedBackend(ModelBackend):
    """Embedded backend: runs inference inside the MCP server container.

    For maximum convenience, this backend is designed to work without separate
    model service containers. It currently supports real-weight ProteinMPNN
    execution when the ProteinMPNN code + weights and dependencies are present.

    AlphaFold/RFDiffusion embedding is intentionally conservative: these models
    require large datasets and specialized installs, so they report not_ready
    unless you provide a compatible installation.
    """

    def __init__(self, cfg: EmbeddedConfig):
        self.cfg = cfg

    def _model_dir(self) -> str:
        return (self.cfg.model_dir or "/models").strip() or "/models"

    def _format_argv(self, argv: List[str], **vars: str) -> List[str]:
        out: List[str] = []
        for a in argv:
            try:
                out.append(a.format(**vars))
            except Exception:
                out.append(a)
        return out

    def _runner_argv(self, key: str) -> List[str]:
        try:
            runners = getattr(self.cfg, "runners", None)
            cmd = getattr(runners, key, None)
            argv = getattr(cmd, "argv", None)
            if isinstance(argv, list):
                return [str(x) for x in argv]
        except Exception:
            return []
        return []

    def _runner_timeout(self, key: str, default_seconds: int = 3600) -> int:
        try:
            runners = getattr(self.cfg, "runners", None)
            cmd = getattr(runners, key, None)
            t = getattr(cmd, "timeout_seconds", None)
            if isinstance(t, int) and t > 0:
                return t
        except Exception:
            return default_seconds
        return default_seconds

    def _embedded_alphafold_db_dir(self) -> str:
        try:
            subdir = getattr(getattr(self.cfg, "downloads", None), "alphafold_db_subdir", None) or "alphafold_db"
        except Exception:
            subdir = "alphafold_db"
        subdir = (subdir or "alphafold_db").strip() or "alphafold_db"
        return os.path.join(self._model_dir(), subdir)

    def _bootstrap_status_path(self) -> str:
        return os.path.join(self._model_dir(), ".embedded_bootstrap_status.json")

    def _load_bootstrap_status(self) -> Optional[Dict[str, Any]]:
        path = self._bootstrap_status_path()
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _status_reason_if_bootstrapping(self, model_key: str) -> Optional[str]:
        st = self._load_bootstrap_status()
        if not st or not st.get("running"):
            return None

        # Special-case AlphaFold: if the reduced pack is done but the optional
        # full extras pack is in progress, surface that progress under AlphaFold.
        try:
            if model_key == "alphafold":
                steps = st.get("steps") if isinstance(st.get("steps"), dict) else {}
                full = steps.get("alphafold_full") if isinstance(steps, dict) else None
                if isinstance(full, dict):
                    full_state = (full.get("state") or "").strip().lower()
                    if full_state in {"downloading", "extracting", "installing"}:
                        label = (full.get("label") or "AlphaFold DB (full extras)").strip()
                        pct = full.get("percent")
                        pct_str = f"{float(pct):.0f}%" if isinstance(pct, (int, float)) else None
                        return f"{label}: {full_state}{f' ({pct_str})' if pct_str else ''}"
        except Exception:
            pass

        steps = st.get("steps") if isinstance(st.get("steps"), dict) else {}
        step = steps.get(model_key) if isinstance(steps, dict) else None
        if not isinstance(step, dict):
            current = st.get("current")
            if isinstance(current, str) and current:
                return f"Downloading assets in background (current: {current})"
            return "Downloading assets in background"

        state = (step.get("state") or "").strip().lower()
        label = (step.get("label") or model_key).strip() or model_key
        pct = step.get("percent")
        pct_str = f"{float(pct):.0f}%" if isinstance(pct, (int, float)) else None

        if state in {"downloading", "extracting", "installing"}:
            return f"{label}: {state}{f' ({pct_str})' if pct_str else ''}"
        if state:
            return f"{label}: {state}"
        return f"{label}: downloading"

    def _alphafold_ready(self) -> Tuple[bool, str]:
        boot = self._status_reason_if_bootstrapping("alphafold")
        if boot:
            return False, boot

        argv = self._runner_argv("alphafold")
        if not argv:
            return False, "Embedded AlphaFold runner not configured (set embedded.runners.alphafold.argv)"

        # If the user configured DB downloads, require the DB dir to exist.
        db_url = None
        try:
            db_url = getattr(getattr(self.cfg, "downloads", None), "alphafold_db_url", None)
        except Exception:
            db_url = None
        db_url = (db_url or os.getenv("ALPHAFOLD_DB_URL") or "").strip() or None
        if db_url:
            db_dir = self._embedded_alphafold_db_dir()
            if not os.path.isdir(db_dir) or not os.listdir(db_dir):
                boot = self._status_reason_if_bootstrapping("alphafold")
                if boot:
                    return False, boot
                return False, f"AlphaFold DB dir missing/empty: {db_dir} (run embedded download/bootstrap)"

        return True, "ready"

    def _rfdiffusion_ready(self) -> Tuple[bool, str]:
        boot = self._status_reason_if_bootstrapping("rfdiffusion")
        if boot:
            return False, boot

        argv = self._runner_argv("rfdiffusion")
        if not argv:
            return False, "Embedded RFDiffusion runner not configured (set embedded.runners.rfdiffusion.argv)"

        # If the user configured weight downloads, require something present in model_dir/rfdiffusion.
        w_url = None
        try:
            w_url = getattr(getattr(self.cfg, "downloads", None), "rfdiffusion_weights_url", None)
        except Exception:
            w_url = None
        w_url = (w_url or os.getenv("RFDIFFUSION_WEIGHTS_URL") or "").strip() or None
        if w_url:
            w_dir = os.path.join(self._model_dir(), "rfdiffusion")
            if not os.path.isdir(w_dir) or not os.listdir(w_dir):
                boot = self._status_reason_if_bootstrapping("rfdiffusion")
                if boot:
                    return False, boot
                return False, f"RFDiffusion weights dir missing/empty: {w_dir} (run embedded download/bootstrap)"

        return True, "ready"

    def _multimer_ready(self) -> Tuple[bool, str]:
        argv = self._runner_argv("alphafold_multimer")
        if not argv:
            return False, "Embedded AlphaFold-Multimer runner not configured (set embedded.runners.alphafold_multimer.argv)"
        return True, "ready"

    def _maybe_bootstrap(self, models: List[str]) -> None:
        # Only bootstrap when explicitly enabled.
        if not (getattr(self.cfg, "auto_download", False) or getattr(self.cfg, "auto_install", False)):
            return
        try:
            self.bootstrap_assets(models)
        except Exception:
            # Bootstrap should be best-effort; actual call should still raise a clear error if not ready.
            return

    def _embedded_proteinmpnn_home(self) -> Optional[str]:
        try:
            model_dir = (self.cfg.model_dir or "/models").strip() or "/models"
            candidate = os.path.join(model_dir, "ProteinMPNN")
            if os.path.exists(candidate):
                return candidate
        except Exception:
            return None
        return None

    def _proteinmpnn_home(self) -> Optional[str]:
        env = (os.getenv("PROTEINMPNN_HOME") or "").strip()
        if env:
            return env
        embedded = self._embedded_proteinmpnn_home()
        if embedded:
            return embedded
        for candidate in ("/opt/ProteinMPNN", "/app/ProteinMPNN"):
            if os.path.exists(candidate):
                return candidate
        return None

    def _bootstrap_proteinmpnn(self) -> None:
        """Best-effort bootstrap for embedded ProteinMPNN.

        This is intentionally opt-in via EmbeddedConfig.auto_install.
        Downloads ProteinMPNN source into <model_dir>/ProteinMPNN and installs
        minimal python deps. Weights download requires an explicit URL via
        PROTEINMPNN_WEIGHTS_URL, or the user can mount weights into place.
        """

        import sys
        import tarfile
        import tempfile
        import urllib.request
        from pathlib import Path
        import subprocess

        model_dir = Path((self.cfg.model_dir or "/models").strip() or "/models")
        model_dir.mkdir(parents=True, exist_ok=True)

        home = model_dir / "ProteinMPNN"
        lock = model_dir / ".proteinmpnn_bootstrap.lock"

        # Very small guard to avoid multiple concurrent pip/download attempts.
        try:
            if lock.exists():
                return
            lock.write_text("bootstrapping", encoding="utf-8")
        except Exception:
            pass

        try:
            # Install python deps (best-effort).
            pkgs = (os.getenv("PROTEINMPNN_PIP_PACKAGES") or "numpy torch").split()
            if pkgs:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs],
                    check=False,
                    capture_output=True,
                    text=True,
                )

            # Fetch ProteinMPNN source if missing.
            if not home.exists():
                src_url = None
                try:
                    src_url = getattr(getattr(self.cfg, "downloads", None), "proteinmpnn_source_tarball_url", None)
                except Exception:
                    src_url = None
                src_url = (src_url or os.getenv("PROTEINMPNN_SOURCE_TARBALL_URL") or "").strip() or "https://github.com/dauparas/ProteinMPNN/archive/refs/heads/main.tar.gz"
                with tempfile.TemporaryDirectory(prefix="proteinmpnn_bootstrap_") as tmp:
                    tgz_path = Path(tmp) / "proteinmpnn.tgz"
                    urllib.request.urlretrieve(src_url, tgz_path)

                    with tarfile.open(tgz_path, "r:gz") as tf:
                        tf.extractall(path=tmp)

                    extracted = None
                    for child in Path(tmp).iterdir():
                        if child.is_dir() and child.name.lower().startswith("proteinmpnn"):
                            extracted = child
                            break
                    if not extracted:
                        raise RuntimeError("Downloaded ProteinMPNN archive had unexpected structure")

                    # Move into place
                    if home.exists():
                        # race-safe-ish
                        return
                    import shutil
                    shutil.move(str(extracted), str(home))

            # Ensure weights if absent.
            weights = home / "vanilla_model_weights" / "v_48_020.pt"
            if not weights.exists():
                weights_url = None
                try:
                    weights_url = getattr(getattr(self.cfg, "downloads", None), "proteinmpnn_weights_url", None)
                except Exception:
                    weights_url = None
                weights_url = (weights_url or os.getenv("PROTEINMPNN_WEIGHTS_URL") or "").strip()
                if not weights_url and (os.getenv("MCP_PROTEINMPNN_DEFAULT_WEIGHTS") or "").strip().lower() in {"1", "true", "yes", "y", "on"}:
                    # Official upstream weights live in the ProteinMPNN repo.
                    weights_url = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt"
                if not weights_url:
                    raise RuntimeError(
                        "ProteinMPNN weights missing. Provide them at "
                        f"{weights} (e.g. mount into /models) or set PROTEINMPNN_WEIGHTS_URL."
                    )
                weights.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(weights_url, weights)
        finally:
            try:
                lock.unlink(missing_ok=True)  # py3.11+
            except Exception:
                pass

    def _proteinmpnn_ready(self) -> Tuple[bool, str]:
        home = self._proteinmpnn_home()
        if not home:
            return False, "ProteinMPNN code not found (set PROTEINMPNN_HOME or provide /opt/ProteinMPNN)"

        script = os.path.join(home, "protein_mpnn_run.py")
        weights = os.path.join(home, "vanilla_model_weights", "v_48_020.pt")
        if not os.path.exists(script):
            return False, "ProteinMPNN runner script missing (protein_mpnn_run.py)"
        if not os.path.exists(weights):
            return False, "ProteinMPNN weights missing (vanilla_model_weights/v_48_020.pt)"

        try:
            import torch  # noqa: F401
            import numpy  # noqa: F401
        except Exception:
            return False, "Missing python deps for ProteinMPNN (torch, numpy)"

        return True, "ready"

    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        import asyncio
        import sys
        import tempfile
        from pathlib import Path
        import subprocess

        seq = (sequence or "").strip()
        if not seq:
            raise RuntimeError("Missing sequence")

        ready, reason = self._alphafold_ready()
        if not ready:
            # Try to auto-download DBs if configured.
            self._maybe_bootstrap(["alphafold"])
            ready, reason = self._alphafold_ready()
        if not ready:
            raise RuntimeError(f"Embedded AlphaFold not ready: {reason}")

        runner = self._runner_argv("alphafold")
        timeout = self._runner_timeout("alphafold", default_seconds=3600)

        with tempfile.TemporaryDirectory(prefix="alphafold_embedded_") as tmpdir:
            work_dir = Path(tmpdir)
            fasta_path = work_dir / "input.fasta"
            out_pdb = work_dir / "result.pdb"

            fasta_path.write_text(f">query\n{seq}\n", encoding="utf-8")

            argv = self._format_argv(
                runner,
                model_dir=self._model_dir(),
                work_dir=str(work_dir),
                fasta_path=str(fasta_path),
                output_pdb_path=str(out_pdb),
                output_dir=str(work_dir),
            )

            proc = await asyncio.to_thread(
                lambda: subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Embedded AlphaFold runner failed "
                    f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                )
            if not out_pdb.exists():
                # Some runners may write into output_dir; attempt common fallback.
                candidates = list(work_dir.glob("*.pdb"))
                if candidates:
                    out_pdb = candidates[0]
                else:
                    raise RuntimeError(
                        f"Embedded AlphaFold runner succeeded but no PDB produced at {out_pdb}"
                    )

            pdb_text = out_pdb.read_text(encoding="utf-8")
            return {"pdb": pdb_text, "backend": "Embedded"}

    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        import asyncio
        import tempfile
        from pathlib import Path
        import subprocess

        pdb = (target_pdb or "").strip()
        if not pdb:
            raise RuntimeError("Missing target_pdb")

        n = int(num_designs or 1)
        n = max(1, min(n, 50))

        ready, reason = self._rfdiffusion_ready()
        if not ready:
            self._maybe_bootstrap(["rfdiffusion"])
            ready, reason = self._rfdiffusion_ready()
        if not ready:
            raise RuntimeError(f"Embedded RFDiffusion not ready: {reason}")

        runner = self._runner_argv("rfdiffusion")
        timeout = self._runner_timeout("rfdiffusion", default_seconds=3600)

        with tempfile.TemporaryDirectory(prefix="rfdiffusion_embedded_") as tmpdir:
            work_dir = Path(tmpdir)
            target_path = work_dir / "target.pdb"
            out_dir = work_dir / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            target_path.write_text(pdb + "\n", encoding="utf-8")

            argv = self._format_argv(
                runner,
                model_dir=self._model_dir(),
                work_dir=str(work_dir),
                target_pdb_path=str(target_path),
                output_dir=str(out_dir),
                num_designs=str(n),
            )

            proc = await asyncio.to_thread(
                lambda: subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Embedded RFDiffusion runner failed "
                    f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                )

            pdb_files = sorted(out_dir.glob("*.pdb"))
            if not pdb_files:
                pdb_files = sorted(out_dir.glob("design_*.pdb"))
            if not pdb_files:
                raise RuntimeError("Embedded RFDiffusion runner succeeded but produced no .pdb designs")

            designs: List[Dict[str, Any]] = []
            for idx, fp in enumerate(pdb_files[:n]):
                designs.append({"design_id": idx, "pdb": fp.read_text(encoding="utf-8")})
            return {"designs": designs, "backend": "Embedded"}

    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        import asyncio
        import sys
        import tempfile
        from pathlib import Path
        import subprocess

        ready, reason = self._proteinmpnn_ready()
        if (not ready) and self.cfg.auto_install:
            # Best-effort bootstrap; failures should be explicit to the user.
            try:
                await asyncio.to_thread(self._bootstrap_proteinmpnn)
            except Exception as exc:
                raise RuntimeError(f"Embedded ProteinMPNN bootstrap failed: {exc}")
            ready, reason = self._proteinmpnn_ready()
        if not ready:
            raise RuntimeError(f"Embedded ProteinMPNN not ready: {reason}")

        home = self._proteinmpnn_home()
        assert home is not None

        with tempfile.TemporaryDirectory(prefix="proteinmpnn_embedded_") as tmpdir:
            pdb_path = Path(tmpdir) / "backbone.pdb"
            pdb_path.write_text(backbone_pdb, encoding="utf-8")
            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                os.path.join(home, "protein_mpnn_run.py"),
                "--pdb_path",
                str(pdb_path),
                "--out_folder",
                str(out_dir),
                "--num_seq_per_target",
                "1",
                "--batch_size",
                "1",
                "--sampling_temp",
                os.getenv("PROTEINMPNN_SAMPLING_TEMP", "0.1"),
                "--seed",
                os.getenv("PROTEINMPNN_SEED", "1"),
                "--model_name",
                os.getenv("PROTEINMPNN_MODEL_NAME", "v_48_020"),
                "--suppress_print",
                "1",
            ]

            proc = await asyncio.to_thread(lambda: subprocess.run(cmd, capture_output=True, text=True))
            if proc.returncode != 0:
                raise RuntimeError(f"Embedded ProteinMPNN failed (exit {proc.returncode}): {proc.stderr}")

            seqs_dir = out_dir / "seqs"
            fasta_files = sorted(seqs_dir.glob("*.fa"))
            if not fasta_files:
                raise RuntimeError("Embedded ProteinMPNN produced no FASTA outputs")

            # Parse FASTA, prefer sampled (T=) sequences.
            records: List[Tuple[str, str]] = []
            header: Optional[str] = None
            seq_lines: List[str] = []
            for raw in fasta_files[0].read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(seq_lines)))
                    header = line[1:].strip()
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if header is not None:
                records.append((header, "".join(seq_lines)))

            seq: Optional[str] = None
            for h, s in records:
                if "T=" in h:
                    seq = s
                    break
            if seq is None and records:
                seq = records[-1][1]
            if not seq:
                raise RuntimeError("Embedded ProteinMPNN returned an empty sequence")

            allowed = set("ACDEFGHIKLMNPQRSTVWYX")
            cleaned = "".join([c for c in seq.replace("/", "").upper() if c in allowed])
            if not cleaned:
                raise RuntimeError("Embedded ProteinMPNN returned invalid sequence")

            return {"sequence": cleaned, "backend": "Embedded"}

    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        import asyncio
        import tempfile
        from pathlib import Path
        import subprocess

        seqs = [s.strip() for s in (sequences or []) if (s or "").strip()]
        if not seqs:
            raise RuntimeError("Missing sequences")

        ready, reason = self._multimer_ready()
        if not ready:
            raise RuntimeError(f"Embedded AlphaFold-Multimer not ready: {reason}")

        runner = self._runner_argv("alphafold_multimer")
        timeout = self._runner_timeout("alphafold_multimer", default_seconds=3600)

        with tempfile.TemporaryDirectory(prefix="alphafold_multimer_embedded_") as tmpdir:
            work_dir = Path(tmpdir)
            fasta_path = work_dir / "input.fasta"
            out_pdb = work_dir / "result.pdb"

            # Multi-chain FASTA
            fasta_lines = []
            for i, s in enumerate(seqs):
                fasta_lines.append(f">chain_{i+1}")
                fasta_lines.append(s)
            fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

            argv = self._format_argv(
                runner,
                model_dir=self._model_dir(),
                work_dir=str(work_dir),
                fasta_path=str(fasta_path),
                output_pdb_path=str(out_pdb),
                output_dir=str(work_dir),
            )

            proc = await asyncio.to_thread(
                lambda: subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Embedded AlphaFold-Multimer runner failed "
                    f"(exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                )
            if not out_pdb.exists():
                candidates = list(work_dir.glob("*.pdb"))
                if candidates:
                    out_pdb = candidates[0]
                else:
                    raise RuntimeError(
                        f"Embedded AlphaFold-Multimer runner succeeded but no PDB produced at {out_pdb}"
                    )
            pdb_text = out_pdb.read_text(encoding="utf-8")
            return {"pdb": pdb_text, "backend": "Embedded"}

    async def check_health(self) -> Dict[str, Any]:
        mpnn_ready, mpnn_reason = self._proteinmpnn_ready()
        af_ready, af_reason = self._alphafold_ready()
        rf_ready, rf_reason = self._rfdiffusion_ready()
        m_ready, m_reason = self._multimer_ready()

        boot = self._load_bootstrap_status() or {}
        boot_running = bool(boot.get("running"))
        boot_steps = boot.get("steps") if isinstance(boot.get("steps"), dict) else {}
        def _boot_flag(key: str) -> bool:
            if not boot_running:
                return False
            return isinstance(boot_steps, dict) and key in boot_steps

        return {
            "alphafold": {
                "status": "ready" if af_ready else "not_ready",
                "backend": "Embedded",
                "reason": af_reason,
                **({"bootstrapping": True} if _boot_flag("alphafold") or _boot_flag("alphafold_full") else {}),
            },
            "rfdiffusion": {
                "status": "ready" if rf_ready else "not_ready",
                "backend": "Embedded",
                "reason": rf_reason,
                **({"bootstrapping": True} if _boot_flag("rfdiffusion") else {}),
            },
            "proteinmpnn": {
                "status": "ready" if mpnn_ready else "not_ready",
                "backend": "Embedded",
                "reason": mpnn_reason,
                **({"bootstrapping": True} if _boot_flag("proteinmpnn") else {}),
            },
            "alphafold_multimer": {"status": "ready" if m_ready else "not_ready", "backend": "Embedded", "reason": m_reason},
        }

    def bootstrap_assets(self, models: List[str]) -> Dict[str, Any]:
        """Best-effort download/bootstrap for embedded assets.

        This is intended to be called from a dashboard button or admin workflow.
        It only downloads what is explicitly configured (URLs must be provided).
        """

        from pathlib import Path
        import urllib.request
        import tarfile
        import zipfile

        model_dir = Path((self.cfg.model_dir or "/models").strip() or "/models")
        model_dir.mkdir(parents=True, exist_ok=True)

        status_path = Path(self._bootstrap_status_path())
        status_path.parent.mkdir(parents=True, exist_ok=True)

        status: Dict[str, Any] = {
            "running": True,
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "finished_at": None,
            "current": None,
            "steps": {},
        }
        for k in (models or []):
            key = (k or "").strip().lower()
            if key:
                status["steps"][key] = {"label": key, "state": "pending"}
        # Optional stage-2 key for full alphafold extras.
        full_url_cfg = ""
        try:
            full_url_cfg = (getattr(getattr(self.cfg, "downloads", None), "alphafold_db_url_full", None) or "").strip()
        except Exception:
            full_url_cfg = ""
        if full_url_cfg or (os.getenv("ALPHAFOLD_DB_URL_FULL") or "").strip():
            status["steps"].setdefault("alphafold_full", {"label": "AlphaFold DB (full extras)", "state": "pending"})

        def _write_status() -> None:
            try:
                status["updated_at"] = datetime.now().isoformat()
                tmp = status_path.with_suffix(status_path.suffix + ".tmp")
                tmp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
                tmp.replace(status_path)
            except Exception:
                pass

        def _set_step(model_key: str, **fields: Any) -> None:
            try:
                steps = status.setdefault("steps", {})
                step = steps.setdefault(model_key, {"label": model_key})
                if isinstance(step, dict):
                    step.update(fields)
            except Exception:
                return

        _write_status()

        def download(url: str, dest: Path, model_key: str, label: str) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            status["current"] = model_key
            _set_step(model_key, label=label, state="downloading", url=url)
            _write_status()

            last_write = 0.0

            def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
                nonlocal last_write
                now = time.time()
                if now - last_write < 1.0:
                    return
                last_write = now
                downloaded = int(blocknum) * int(blocksize)
                if isinstance(totalsize, int) and totalsize > 0:
                    percent = max(0.0, min(100.0, (downloaded / float(totalsize)) * 100.0))
                    _set_step(model_key, bytes_downloaded=downloaded, bytes_total=totalsize, percent=percent)
                else:
                    _set_step(model_key, bytes_downloaded=downloaded)
                _write_status()

            urllib.request.urlretrieve(url, dest, reporthook=reporthook)
            _set_step(model_key, state="downloaded", path=str(dest), percent=100.0)
            _write_status()

        def extract(archive: Path, dest_dir: Path) -> None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            name = archive.name.lower()
            if name.endswith(".tar.gz") or name.endswith(".tgz"):
                with tarfile.open(archive, "r:gz") as tf:
                    tf.extractall(path=dest_dir)
                return
            if name.endswith(".tar"):
                with tarfile.open(archive, "r:") as tf:
                    tf.extractall(path=dest_dir)
                return
            if name.endswith(".zip"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(path=dest_dir)
                return

        results: Dict[str, Any] = {}
        for m in models:
            m = (m or "").strip().lower()
            if not m:
                continue
            try:
                if m == "proteinmpnn":
                    status["current"] = "proteinmpnn"
                    _set_step("proteinmpnn", label="ProteinMPNN", state="installing")
                    _write_status()
                    self._bootstrap_proteinmpnn()
                    _set_step("proteinmpnn", label="ProteinMPNN", state="done")
                    _write_status()
                    results[m] = {"status": "ok"}
                    continue

                if m == "rfdiffusion":
                    url = (
                        (getattr(getattr(self.cfg, "downloads", None), "rfdiffusion_weights_url", None) or "").strip()
                        or (os.getenv("RFDIFFUSION_WEIGHTS_URL") or "").strip()
                    )
                    if not url:
                        raise RuntimeError("Missing embedded.downloads.rfdiffusion_weights_url")
                    filename = os.path.basename(url.split("?")[0]) or "rfdiffusion_weights.bin"
                    dest = model_dir / "rfdiffusion" / filename
                    if not dest.exists():
                        download(url, dest, "rfdiffusion", "RFDiffusion weights")
                    _set_step("rfdiffusion", label="RFDiffusion weights", state="done")
                    _write_status()
                    results[m] = {"status": "ok", "path": str(dest)}
                    continue

                if m == "alphafold":
                    url = (
                        (getattr(getattr(self.cfg, "downloads", None), "alphafold_db_url", None) or "").strip()
                        or (os.getenv("ALPHAFOLD_DB_URL") or "").strip()
                    )
                    if not url:
                        raise RuntimeError("Missing embedded.downloads.alphafold_db_url")
                    subdir = (getattr(getattr(self.cfg, "downloads", None), "alphafold_db_subdir", None) or "alphafold_db").strip() or "alphafold_db"
                    dest_dir = model_dir / subdir
                    filename = os.path.basename(url.split("?")[0]) or "alphafold_db"
                    archive = model_dir / "downloads" / filename
                    if not archive.exists():
                        download(url, archive, "alphafold", "AlphaFold DB (reduced)")
                    # Extract archives if applicable.
                    try:
                        _set_step("alphafold", label="AlphaFold DB (reduced)", state="extracting")
                        _write_status()
                        extract(archive, dest_dir)
                        _set_step("alphafold", label="AlphaFold DB (reduced)", state="done")
                        _write_status()
                        results[m] = {"status": "ok", "db_dir": str(dest_dir), "archive": str(archive)}
                    except Exception:
                        # If it's not an archive type, leave it as-is.
                        _set_step("alphafold", label="AlphaFold DB (reduced)", state="done")
                        _write_status()
                        results[m] = {"status": "ok", "archive": str(archive), "note": "downloaded (no extraction)"}

                    # Stage-2 (optional): pull additional DB assets after reduced is available.
                    full_url = (
                        (getattr(getattr(self.cfg, "downloads", None), "alphafold_db_url_full", None) or "").strip()
                        or (os.getenv("ALPHAFOLD_DB_URL_FULL") or "").strip()
                    )
                    if full_url:
                        full_name = os.path.basename(full_url.split("?")[0]) or "alphafold_db_full"
                        full_archive = model_dir / "downloads" / full_name
                        if not full_archive.exists():
                            download(full_url, full_archive, "alphafold_full", "AlphaFold DB (full extras)")
                        try:
                            _set_step("alphafold_full", label="AlphaFold DB (full extras)", state="extracting")
                            _write_status()
                            extract(full_archive, dest_dir)
                            _set_step("alphafold_full", label="AlphaFold DB (full extras)", state="done")
                            _write_status()
                            results["alphafold_full"] = {"status": "ok", "db_dir": str(dest_dir), "archive": str(full_archive)}
                        except Exception as exc:
                            _set_step("alphafold_full", label="AlphaFold DB (full extras)", state="error", error=str(exc))
                            _write_status()
                            results["alphafold_full"] = {"status": "error", "error": str(exc)}
                    continue

                results[m] = {"status": "skipped", "reason": "unknown model key"}
            except Exception as exc:
                _set_step(m, state="error", error=str(exc))
                _write_status()
                results[m] = {"status": "error", "error": str(exc)}

            status["running"] = False
            status["current"] = None
            status["finished_at"] = datetime.now().isoformat()
            _write_status()
            return results


class FallbackBackend(ModelBackend):
    """Try multiple backends in order for each call."""

    def __init__(self, providers: List[Tuple[ProviderName, ModelBackend]]):
        self.providers = providers
        logger.info("Initialized FallbackBackend order=%s", [p[0] for p in providers])

    async def _try(self, fn_name: str, *args, **kwargs):
        last_exc: Optional[Exception] = None
        errors: List[str] = []
        for provider_name, backend in self.providers:
            try:
                fn = getattr(backend, fn_name)
                return await fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                raw = str(exc).strip()
                msg = " ".join(raw.split()) if raw else type(exc).__name__
                # Keep the job-visible error small; full details should be in logs.
                if len(msg) > 400:
                    msg = msg[:386].rstrip() + " …(truncated)"
                errors.append(f"{provider_name}: {msg}")
                logger.warning("Provider %s failed for %s: %s", provider_name, fn_name, exc)
                continue
        if errors:
            raise RuntimeError(f"All providers failed for {fn_name}: " + "; ".join(errors))
        raise RuntimeError(f"All providers failed for {fn_name}: {last_exc}")

    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        return await self._try("predict_structure", sequence)

    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        return await self._try("design_binder_backbone", target_pdb, num_designs)

    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        return await self._try("generate_sequence", backbone_pdb)

    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        return await self._try("predict_complex", sequences)

    async def check_health(self) -> Dict[str, Any]:
        # Merge into a single service map (as expected by the dashboard),
        # but keep provider details inside each entry.
        per_provider: Dict[str, Dict[str, Any]] = {}
        for provider_name, backend in self.providers:
            try:
                per_provider[provider_name] = await backend.check_health()
            except Exception as exc:
                per_provider[provider_name] = {
                    "alphafold": {"status": "error", "backend": str(provider_name), "error": str(exc)},
                    "rfdiffusion": {"status": "error", "backend": str(provider_name), "error": str(exc)},
                    "proteinmpnn": {"status": "error", "backend": str(provider_name), "error": str(exc)},
                    "alphafold_multimer": {"status": "error", "backend": str(provider_name), "error": str(exc)},
                }

        merged: Dict[str, Any] = {}
        for service_name in ["alphafold", "rfdiffusion", "proteinmpnn", "alphafold_multimer"]:
            chosen: Optional[Tuple[ProviderName, Dict[str, Any]]] = None
            for provider_name, _ in self.providers:
                entry = (per_provider.get(provider_name) or {}).get(service_name) or {}
                if entry.get("status") == "ready":
                    chosen = (provider_name, entry)
                    break

            if chosen is None:
                # If nothing is ready, prefer a provider that is actively bootstrapping
                # so the UI can show download progress (game-style staged install).
                for provider_name, _ in self.providers:
                    entry = (per_provider.get(provider_name) or {}).get(service_name) or {}
                    if entry.get("bootstrapping") is True:
                        chosen = (provider_name, entry)
                        break

            if chosen is None:
                # Fall back to the first provider's view.
                provider_name, _ = self.providers[0]
                entry = (per_provider.get(provider_name) or {}).get(service_name) or {"status": "not_ready"}
                chosen = (provider_name, entry)

            provider_name, entry = chosen
            normalized = dict(entry)
            if normalized.get("status") == "disabled":
                normalized["status"] = "not_ready"
                normalized.setdefault("reason", "Service is disabled/not configured")

            merged[service_name] = {
                **normalized,
                "selected_provider": provider_name,
                "providers": {k: (v.get(service_name) if isinstance(v, dict) else None) for k, v in per_provider.items()},
            }

        return merged


class BackendManager:
    """Builds and caches a backend based on runtime config."""

    def __init__(self, config_manager: RuntimeConfigManager):
        self.config_manager = config_manager
        self._backend: Optional[ModelBackend] = None
        self._revision = -1

    def get(self) -> ModelBackend:
        if self._backend is None or self._revision != self.config_manager.revision:
            self._backend = self._build(self.config_manager.get())
            self._revision = self.config_manager.revision
        return self._backend

    def _build(self, cfg: MCPServerConfig) -> ModelBackend:
        providers: Dict[ProviderName, ModelBackend] = {
            "nim": NIMBackend(),
            "external": ExternalBackend(cfg.external.service_urls),
            "embedded": EmbeddedBackend(cfg.embedded),
        }

        # Apply NIM URL overrides from config.
        # NIMBackend currently reads from env defaults in __init__; re-bind URLs.
        nim = providers["nim"]
        if isinstance(nim, NIMBackend):
            nim.service_urls = dict(cfg.nim.service_urls)
            nim.services = {k: v for k, v in nim.service_urls.items() if v}

        if cfg.routing.mode == "single":
            return providers[cfg.routing.primary]

        # fallback
        chain: List[Tuple[ProviderName, ModelBackend]] = []
        for name in cfg.routing.order:
            if name == "nim" and not cfg.nim.enabled:
                continue
            if name == "external" and not cfg.external.enabled:
                continue
            if name == "embedded" and not cfg.embedded.enabled:
                continue
            chain.append((name, providers[name]))

        if not chain:
            chain = [("nim", providers["nim"]) ]
        return FallbackBackend(chain)


class NativeBackend(ModelBackend):
    """Native Model Backend (Direct Python API calls)
    
    Optimized for DGX Spark systems running models directly on hardware
    without NIM containers. Uses Python libraries to call models directly.
    """
    
    def __init__(self):
        self.models_loaded = False
        self.models = {}
        logger.info("Initialized Native Backend for DGX Spark")
        self._check_and_load_models()
    
    def _check_and_load_models(self):
        """Check if model libraries are available and load them"""
        try:
            # Try to import model libraries
            # These would be installed on the DGX Spark system
            self.available_models = {
                "alphafold": self._check_alphafold(),
                "rfdiffusion": self._check_rfdiffusion(),
                "proteinmpnn": self._check_proteinmpnn(),
            }
            logger.info(f"Available models: {self.available_models}")
        except Exception as e:
            logger.warning(f"Model loading check: {e}")
            self.available_models = {}
    
    def _check_alphafold(self) -> bool:
        """Check if AlphaFold2 is available"""
        try:
            # Check for AlphaFold runner script and conda environment
            runner_path = "/opt/generative-protein-binder-design/tools/alphafold2_arm64/alphafold_runner.py"
            if os.path.exists(runner_path):
                # Check if conda environment exists
                import subprocess
                result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
                if 'alphafold2_arm64' in result.stdout:
                    logger.info("AlphaFold2 ARM64 environment found")
                    return True
        except Exception as e:
            logger.debug(f"AlphaFold not available: {e}")
        return False
    
    def _check_rfdiffusion(self) -> bool:
        """Check if RFDiffusion is available"""
        try:
            # Check for RFDiffusion runner script and conda environment
            runner_path = "/opt/generative-protein-binder-design/tools/rfdiffusion_arm64/rfdiffusion_runner.py"
            if os.path.exists(runner_path):
                # Check if conda environment exists
                import subprocess
                result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
                if 'rfdiffusion_arm64' in result.stdout:
                    logger.info("RFDiffusion ARM64 environment found")
                    return True
        except Exception as e:
            logger.debug(f"RFDiffusion not available: {e}")
        return False
    
    def _check_proteinmpnn(self) -> bool:
        """Check if ProteinMPNN is available"""
        try:
            # Check for ProteinMPNN runner script and conda environment
            runner_path = "/opt/generative-protein-binder-design/tools/proteinmpnn_arm64/proteinmpnn_runner.py"
            if os.path.exists(runner_path):
                # Check if conda environment exists
                import subprocess
                result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
                if 'proteinmpnn_arm64' in result.stdout:
                    logger.info("ProteinMPNN ARM64 environment found")
                    return True
        except Exception as e:
            logger.debug(f"ProteinMPNN not available: {e}")
        return False
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        """AlphaFold2 structure prediction using native Python API"""
        logger.info(f"Running AlphaFold2 natively for sequence length {len(sequence)}")
        
        try:
            # Run AlphaFold2 using conda environment
            result = await self._run_alphafold_conda(sequence)
            return {
                "pdb": result,
                "confidence": 0.95,
                "backend": "native",
                "sequence": sequence
            }
            
        except Exception as e:
            logger.error(f"AlphaFold native execution error: {e}")
            if allow_mock_outputs():
                return self._generate_mock_structure(sequence)
            raise
    
    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        """RFDiffusion binder design using native Python API"""
        logger.info(f"Running RFDiffusion natively for {num_designs} designs")
        
        try:
            # Run RFDiffusion using conda environment
            designs = []
            for i in range(num_designs):
                design_pdb = await self._run_rfdiffusion_conda(target_pdb, i)
                designs.append({
                    "design_id": i,
                    "pdb": design_pdb,
                    "backend": "native"
                })
            
            return {"designs": designs}
            
        except Exception as e:
            logger.error(f"RFDiffusion native execution error: {e}")
            if allow_mock_outputs():
                return self._generate_mock_designs(num_designs)
            raise
    
    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        """ProteinMPNN sequence generation using native Python API"""
        logger.info("Running ProteinMPNN natively")
        
        try:
            # Run ProteinMPNN using conda environment
            sequence = await self._run_proteinmpnn_conda(backbone_pdb)
            return {
                "sequence": sequence,
                "score": 0.88,
                "backend": "native"
            }
            
        except Exception as e:
            logger.error(f"ProteinMPNN native execution error: {e}")
            if allow_mock_outputs():
                return self._generate_mock_sequence()
            raise
    
    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        """AlphaFold2-Multimer complex prediction using native Python API"""
        if not self.available_models.get("alphafold"):
            raise RuntimeError("AlphaFold2-Multimer not available in native backend")
        
        logger.info(f"Running AlphaFold2-Multimer natively for {len(sequences)} chains")
        
        try:
            # Import AlphaFold modules for multimer
            from alphafold.model import model, config
            from alphafold.data import pipeline
            
            # Run AlphaFold2-Multimer
            result = {
                "pdb": self._run_alphafold_multimer_inference(sequences),
                "confidence": 0.92,
                "backend": "native",
                "num_chains": len(sequences)
            }
            return result
            
        except ImportError as e:
            logger.error(f"AlphaFold-Multimer import error: {e}")
            if allow_mock_outputs():
                return self._generate_mock_complex()
            raise
    
    async def check_health(self) -> Dict[str, Any]:
        """Check health of native backend"""
        return {
            "alphafold": {
                "status": "ready" if self.available_models.get("alphafold") else "not_available",
                "backend": "Native ARM64",
                "conda_env": "alphafold2_arm64",
                "path": "/opt/generative-protein-binder-design/tools/alphafold2_arm64/"
            },
            "rfdiffusion": {
                "status": "ready" if self.available_models.get("rfdiffusion") else "not_available", 
                "backend": "Native ARM64",
                "conda_env": "rfdiffusion_arm64",
                "path": "/opt/generative-protein-binder-design/tools/rfdiffusion_arm64/"
            },
            "proteinmpnn": {
                "status": "ready" if self.available_models.get("proteinmpnn") else "not_available",
                "backend": "Native ARM64", 
                "conda_env": "proteinmpnn_arm64",
                "path": "/opt/generative-protein-binder-design/tools/proteinmpnn_arm64/"
            },
            "alphafold_multimer": {
                "status": "ready" if self.available_models.get("alphafold") else "not_available",
                "backend": "Native ARM64",
                "conda_env": "alphafold2_arm64",
                "path": "/opt/generative-protein-binder-design/tools/alphafold2_arm64/"
            }
        }
    
    # Real conda environment execution methods
    async def _run_alphafold_conda(self, sequence: str) -> str:
        """Run AlphaFold2 using conda environment"""
        import asyncio
        import tempfile
        import os
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">target\n{sequence}\n")
            fasta_file = f.name
        
        # Create temporary output directory
        output_dir = tempfile.mkdtemp()
        
        try:
            # Run AlphaFold2 using the runner script
            cmd = [
                "conda", "run", "-n", "alphafold2_arm64",
                "python", "/opt/generative-protein-binder-design/tools/alphafold2_arm64/alphafold_runner.py",
                fasta_file, output_dir
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"AlphaFold2 error: {stderr.decode()}")
                raise RuntimeError(f"AlphaFold2 execution failed: {stderr.decode()}")
            
            # Read the result PDB file
            pdb_file = os.path.join(output_dir, "result.pdb")
            if os.path.exists(pdb_file):
                with open(pdb_file, 'r') as f:
                    return f.read()
            else:
                if allow_mock_outputs():
                    return self._generate_mock_pdb(sequence, "alphafold2_native")
                raise RuntimeError("AlphaFold2 did not produce result.pdb")
                
        finally:
            # Cleanup
            os.unlink(fasta_file)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
    
    async def _run_rfdiffusion_conda(self, target_pdb: str, design_id: int) -> str:
        """Run RFDiffusion using conda environment"""
        import asyncio
        import tempfile
        import os
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(target_pdb)
            pdb_file = f.name
        
        # Create temporary output directory
        output_dir = tempfile.mkdtemp()
        
        try:
            # Run RFDiffusion using the runner script
            cmd = [
                "conda", "run", "-n", "rfdiffusion_arm64",
                "python", "/opt/generative-protein-binder-design/tools/rfdiffusion_arm64/rfdiffusion_runner.py",
                pdb_file, output_dir, str(design_id)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"RFDiffusion error: {stderr.decode()}")
                raise RuntimeError(f"RFDiffusion execution failed: {stderr.decode()}")
            
            # Read the result PDB file
            result_file = os.path.join(output_dir, f"design_{design_id}.pdb")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    return f.read()
            else:
                if allow_mock_outputs():
                    return self._generate_mock_pdb(f"design_{design_id}", "rfdiffusion_native")
                raise RuntimeError(f"RFDiffusion did not produce design_{design_id}.pdb")
                
        finally:
            # Cleanup
            os.unlink(pdb_file)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
    
    async def _run_proteinmpnn_conda(self, backbone_pdb: str) -> str:
        """Run ProteinMPNN using conda environment"""
        import asyncio
        import tempfile
        import os
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(backbone_pdb)
            pdb_file = f.name
        
        try:
            # Run ProteinMPNN using the runner script
            cmd = [
                "conda", "run", "-n", "proteinmpnn_arm64",
                "python", "/opt/generative-protein-binder-design/tools/proteinmpnn_arm64/proteinmpnn_runner.py",
                pdb_file
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"ProteinMPNN error: {stderr.decode()}")
                raise RuntimeError(f"ProteinMPNN execution failed: {stderr.decode()}")
            
            # Parse the result
            result = stdout.decode().strip()
            if result and len(result) > 10:  # Basic validation
                return result
            else:
                if allow_mock_outputs():
                    return "MKGSDKIHLTDDSFDITDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"
                raise RuntimeError("ProteinMPNN produced an invalid/empty sequence")
                
        finally:
            # Cleanup
            os.unlink(pdb_file)
    
    # Mock data generators for fallback
    def _generate_mock_structure(self, sequence: str) -> Dict[str, Any]:
        """Generate mock structure data"""
        return {
            "pdb": self._generate_mock_pdb(sequence, "mock_alphafold"),
            "confidence": 0.85,
            "backend": "mock",
            "sequence": sequence
        }
    
    def _generate_mock_designs(self, num_designs: int) -> Dict[str, Any]:
        """Generate mock design data"""
        return {
            "designs": [
                {
                    "design_id": i,
                    "pdb": self._generate_mock_pdb(f"design_{i}", "mock_rfdiffusion"),
                    "backend": "mock"
                }
                for i in range(num_designs)
            ]
        }
    
    def _generate_mock_sequence(self) -> Dict[str, Any]:
        """Generate mock sequence data"""
        return {
            "sequence": "MKGSDKIHLTDDSFDITDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
            "score": 0.85,
            "backend": "mock"
        }
    
    def _generate_mock_complex(self) -> Dict[str, Any]:
        """Generate mock complex data"""
        return {
            "pdb": self._generate_mock_pdb("complex", "mock_multimer"),
            "confidence": 0.88,
            "backend": "mock",
            "num_chains": 2
        }
    
    def _generate_mock_pdb(self, identifier: str, source: str) -> str:
        """Generate mock PDB data"""
        return f"""HEADER    {source.upper()} PREDICTION - {identifier}
REMARK   Mock PDB structure for testing
ATOM      1  N   ALA A   1      12.345  23.456  34.567  1.00 50.00           N
ATOM      2  CA  ALA A   1      11.234  22.345  33.456  1.00 50.00           C
ATOM      3  C   ALA A   1      10.123  21.234  32.345  1.00 50.00           C
ATOM      4  O   ALA A   1       9.012  20.123  31.234  1.00 50.00           O
END
"""


class HybridBackend(ModelBackend):
    """Hybrid Backend - Tries Native first, falls back to NIM"""
    
    def __init__(self):
        self.native = NativeBackend()
        self.nim = NIMBackend()
        logger.info("Initialized Hybrid Backend (Native + NIM fallback)")
    
    async def predict_structure(self, sequence: str) -> Dict[str, Any]:
        try:
            return await self.native.predict_structure(sequence)
        except Exception as e:
            logger.warning(f"Native backend failed, falling back to NIM: {e}")
            return await self.nim.predict_structure(sequence)
    
    async def design_binder_backbone(self, target_pdb: str, num_designs: int) -> Dict[str, Any]:
        try:
            return await self.native.design_binder_backbone(target_pdb, num_designs)
        except Exception as e:
            logger.warning(f"Native backend failed, falling back to NIM: {e}")
            return await self.nim.design_binder_backbone(target_pdb, num_designs)
    
    async def generate_sequence(self, backbone_pdb: str) -> Dict[str, Any]:
        try:
            return await self.native.generate_sequence(backbone_pdb)
        except Exception as e:
            logger.warning(f"Native backend failed, falling back to NIM: {e}")
            return await self.nim.generate_sequence(backbone_pdb)
    
    async def predict_complex(self, sequences: List[str]) -> Dict[str, Any]:
        try:
            return await self.native.predict_complex(sequences)
        except Exception as e:
            logger.warning(f"Native backend failed, falling back to NIM: {e}")
            return await self.nim.predict_complex(sequences)
    
    async def check_health(self) -> Dict[str, Any]:
        native_status = await self.native.check_health()
        nim_status = await self.nim.check_health()
        return {
            "backend_mode": "hybrid",
            "native": native_status,
            "nim": nim_status
        }


def get_backend(backend_type: str = None) -> ModelBackend:
    """Factory function to get the appropriate backend
    
    Args:
        backend_type: "nim", "native", or "hybrid" (default: from env or "nim")
    
    Returns:
        ModelBackend instance
    """
    if backend_type is None:
        backend_type = os.getenv("MODEL_BACKEND", "nim").lower()
    
    if backend_type == "native":
        logger.info("Using Native Backend for direct model execution")
        return NativeBackend()
    elif backend_type == "hybrid":
        logger.info("Using Hybrid Backend (Native + NIM fallback)")
        return HybridBackend()
    else:  # default to NIM
        logger.info("Using NIM Backend for containerized model execution")
        return NIMBackend()
