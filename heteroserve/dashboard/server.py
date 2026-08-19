"""Live dashboard: watch placement decisions happen in real time.

Runs a real cluster behind a small FastAPI app and streams every scheduler event
to the browser over SSE (chosen over WebSockets purely so the project needs no
extra dependency). You can retune the interconnect from a slider and watch the
migrate-vs-recompute decision flip while requests are in flight.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Request as HttpRequest
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from heteroserve.config import (
    ClusterConfig,
    KVConfig,
    LinkConfig,
    ModelConfig,
    WorkerConfig,
)
from heteroserve.sched.router import Request, Router

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights" / "gpt2"
STATIC = Path(__file__).resolve().parent / "static"

SAMPLE_PROMPTS = [
    "You are a careful assistant. Answer using only the provided context. "
    "The Eiffel Tower is located in the city of",
    "You are a careful assistant. Answer using only the provided context. "
    "The largest planet in our solar system is",
    "You are a careful assistant. Answer using only the provided context. "
    "Water boils at a temperature of",
]


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.router: Router | None = None
        self.tokenizer = None
        self.subscribers: list[asyncio.Queue] = []
        self.history: list[dict] = []
        self.started = time.time()

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        devices = self.args.devices.split(",")
        kv = KVConfig(block_size=16, num_blocks=self.args.num_blocks)
        cluster = ClusterConfig(
            model=ModelConfig() if self.args.model == "gpt2" else ModelConfig.tiny(),
            workers=[
                WorkerConfig(f"w{i}-{d.lower()}", device=d, kv=kv,
                             max_batch=8, max_prefill_tokens=256)
                for i, d in enumerate(devices)
            ],
            policy="cache_aware",
            link=LinkConfig(bandwidth_mbps=1000, latency_ms=2.0),
        )

        use_real = self.args.model == "gpt2" and WEIGHTS.exists()
        if use_real:
            from heteroserve.model.tokenizer import GPT2Tokenizer

            self.tokenizer = GPT2Tokenizer.from_dir(WEIGHTS)

        self.router = Router(cluster, weights_dir=WEIGHTS if use_real else None,
                             model_name=self.args.model, max_ctx=512)
        await self.router.start()
        self.router._event_hooks.append(self._on_event)
        await self.router.set_link(cluster.link)
        asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self.router:
            await self.router.stop()

    # -- event fan-out ------------------------------------------------------

    async def _on_event(self, meta: dict) -> None:
        kind = meta.get("kind")
        if kind in ("token",):
            return                       # too chatty for the feed
        if kind in ("cache_add", "cache_drop"):
            return
        await self.broadcast({"type": "event", "event": meta})

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            with contextlib.suppress(ValueError):
                self.subscribers.remove(q)

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            if not self.router:
                continue
            with contextlib.suppress(Exception):
                await self.router.poll_states()
                await self.broadcast({"type": "state", "state": self.snapshot()})

    def snapshot(self) -> dict:
        assert self.router
        snap = self.router.snapshot()
        snap["uptime"] = round(time.time() - self.started, 1)
        snap["recent"] = self.history[-40:]
        return snap

    # -- serving ------------------------------------------------------------

    async def generate(self, prompt: str, max_new_tokens: int) -> dict:
        assert self.router
        if self.tokenizer:
            ids = self.tokenizer.encode(prompt)
        else:
            ids = [(abs(hash(prompt[: i + 1])) % 40000) + 1 for i in range(len(prompt))]

        t0 = time.time()
        rec = await self.router.submit(
            Request(prompt_ids=ids, max_new_tokens=max_new_tokens, temperature=0.0)
        )
        text = self.tokenizer.decode(rec.output_ids) if self.tokenizer else ""

        entry = {
            "req_id": rec.req_id,
            "prompt": prompt[:90],
            "text": text,
            "worker": rec.worker_id,
            "prompt_tokens": rec.prompt_tokens,
            "cached": rec.cached_prefix_tokens,
            "computed": rec.prefill_tokens_computed,
            "ttft_ms": round(rec.ttft * 1000, 1),
            "e2e_ms": round(rec.e2e * 1000, 1),
            "migrated": rec.migrated,
            "migration_mb": round(rec.migration_bytes / 1e6, 2),
            "reason": rec.placement_reason,
            "t": t0,
        }
        self.history.append(entry)
        await self.broadcast({"type": "request", "request": entry})
        return entry


def build_app(dash: Dashboard) -> FastAPI:
    app = FastAPI(title="hetero-serve")

    @app.on_event("startup")
    async def _startup() -> None:
        await dash.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await dash.stop()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(dash.snapshot())

    @app.get("/api/events")
    async def events(request: HttpRequest) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        dash.subscribers.append(q)

        async def stream():
            try:
                yield f"data: {json.dumps({'type': 'state', 'state': dash.snapshot()})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(msg)}\n\n"
            finally:
                with contextlib.suppress(ValueError):
                    dash.subscribers.remove(q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/generate")
    async def generate(payload: dict) -> JSONResponse:
        prompt = payload.get("prompt") or SAMPLE_PROMPTS[0]
        n = int(payload.get("max_new_tokens", 16))
        return JSONResponse(await dash.generate(prompt, n))

    @app.post("/api/burst")
    async def burst(payload: dict) -> JSONResponse:
        n = int(payload.get("n", 8))
        prompts = [
            SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] + f" (variant {i})" for i in range(n)
        ]
        results = await asyncio.gather(
            *[dash.generate(p, int(payload.get("max_new_tokens", 12))) for p in prompts],
            return_exceptions=True,
        )
        ok = [r for r in results if isinstance(r, dict)]
        return JSONResponse({"submitted": n, "completed": len(ok)})

    @app.post("/api/policy")
    async def policy(payload: dict) -> JSONResponse:
        assert dash.router
        dash.router.cluster.policy = payload["policy"]
        return JSONResponse({"policy": dash.router.cluster.policy})

    @app.post("/api/link")
    async def link(payload: dict) -> JSONResponse:
        assert dash.router
        cfg = LinkConfig(
            bandwidth_mbps=float(payload.get("bandwidth_mbps", 1000)),
            latency_ms=float(payload.get("latency_ms", 2.0)),
        )
        await dash.router.set_link(cfg)
        return JSONResponse({"bandwidth_mbps": cfg.bandwidth_mbps,
                             "latency_ms": cfg.latency_ms})

    @app.post("/api/reset")
    async def reset() -> JSONResponse:
        assert dash.router
        await dash.router.reset()
        dash.history.clear()
        return JSONResponse({"ok": True})

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="")
    ap.add_argument("--model", default="gpt2", choices=["gpt2", "tiny"])
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=384)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not args.devices:
        try:
            import openvino as ov

            avail = ov.Core().available_devices
            args.devices = ",".join(d for d in ("GPU", "NPU", "CPU") if d in avail) or "CPU"
        except Exception:
            args.devices = "CPU"

    import uvicorn

    print(f"hetero-serve dashboard -> http://{args.host}:{args.port}   devices={args.devices}")
    uvicorn.run(build_app(Dashboard(args)), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
