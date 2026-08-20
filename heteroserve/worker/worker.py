"""A worker: one accelerator, one KV pool, one continuous-batching loop.

Each worker is a real OS process. It owns its device (CPU / Arc GPU / NPU via
OpenVINO, or the numpy reference engine), owns its slice of KV memory, and runs
iteration-level scheduling locally over whatever sequences the router has placed
on it. The router decides *where* work goes; the worker decides *when*.

Scheduling policy inside a worker is deliberately vLLM-shaped:

  * prefill has priority over decode (get TTFT down), but is chunked so one long
    prompt cannot stall every running sequence
  * decode runs the whole running set as one batch
  * running out of blocks preempts the newest sequence by recomputation rather
    than failing the request

The engine step runs in a thread executor so multi-megabyte KV migrations keep
flowing while the device is busy — numpy and OpenVINO both drop the GIL.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import LinkConfig, ModelConfig, WorkerConfig
from ..kv.blocks import BlockAllocator, OutOfBlocks, chain_hashes
from ..net.shaper import ShapedLink
from ..net.transport import Channel, connect, decode_array, encode_array
from . import protocol as P
from .protocol import SeqState


def softmax_1d(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def sample_token(logits: np.ndarray, temperature: float, top_k: int, rng) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    lg = logits.astype(np.float64) / temperature
    if top_k and 0 < top_k < lg.shape[0]:
        idx = np.argpartition(-lg, top_k)[:top_k]
        return int(rng.choice(idx, p=softmax_1d(lg[idx])))
    return int(rng.choice(lg.shape[0], p=softmax_1d(lg)))


@dataclass
class Sequence:
    seq_id: str
    prompt_ids: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0
    stop_at_eos: bool = True

    block_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    n_kv: int = 0                     # positions with KV resident
    state: str = SeqState.WAITING
    cached_prefix_tokens: int = 0     # tokens served by a prefix-cache hit
    prefill_tokens_computed: int = 0  # tokens we actually ran through the model
    preemptions: int = 0
    t_submit: float = 0.0
    t_admit: float = 0.0
    t_first_token: float = 0.0
    t_done: float = 0.0

    @property
    def ctx(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def prefill_target(self) -> int:
        """How far prefill must reach before decode can take over."""
        n = len(self.ctx)
        return n - 1 if self.output_ids else n

    @property
    def finished(self) -> bool:
        return len(self.output_ids) >= self.max_new_tokens


class Worker:
    def __init__(
        self,
        cfg: WorkerConfig,
        model_cfg: ModelConfig,
        weights_dir: Path | None,
        link: LinkConfig | None = None,
        seed: int = 0,
        eos_token_id: int = 50256,
        max_ctx: int = 512,
        rank: int = 0,
        world_size: int = 1,
        dist_url: str = "",
        dist_backend: str = "auto",
    ):
        self.max_ctx = max_ctx
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.weights_dir = weights_dir
        self.seed = seed
        self.eos_token_id = eos_token_id

        self.link = ShapedLink(link or LinkConfig())
        # Device-to-device transport for KV migration. Set up in start(); stays
        # None when there is nothing to pair with or the rendezvous fails, and
        # the shaped TCP path carries everything as before.
        self.rank = rank
        self.world_size = world_size
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.dist = None
        self._dist_pool = None
        # "off" (not configured) | "pending" | "ready" | "failed". The router
        # waits for every worker to leave "pending" before serving, so a request
        # is never routed while the transport is still being decided.
        self.dist_state = "pending" if (dist_url and world_size >= 2) else "off"
        self.alloc = self._new_allocator()
        self.engine = None
        self.engine_name = "?"
        self.paged_decode = False        # set by build_engine on the CUDA path

        self.seqs: dict[str, Sequence] = {}
        self.order: list[Sequence] = []
        self.peers: dict[str, dict] = {}

        self.router: Channel | None = None
        self.server: asyncio.AbstractServer | None = None
        self.port = cfg.port
        self._rng = np.random.default_rng(seed)
        self._state_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()

        # counters
        self.steps = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.device_busy_s = 0.0
        self.prefill_busy_s = 0.0
        self.decode_busy_s = 0.0
        self.tokens_generated = 0
        self.tokens_prefilled = 0
        self.tokens_saved_by_cache = 0
        self.migrations_in = 0
        self.migrations_out = 0
        self.bytes_migrated_out = 0
        self.bytes_migrated_in = 0

    def _new_allocator(self):
        """KV lives on the accelerator for CUDA, in host RAM otherwise.

        A GPU-resident pool is what lets the fused paged-attention kernel index
        the block table directly instead of having the host gather each
        sequence into a contiguous buffer first.
        """
        if self._wants_torch():
            from ..kv.torch_blocks import TorchBlockAllocator

            return TorchBlockAllocator(self.cfg.kv, self.model_cfg, device=self._torch_device())
        return BlockAllocator(self.cfg.kv, self.model_cfg)

    def _wants_torch(self) -> bool:
        return self.cfg.device.lower().startswith("cuda") or self.cfg.engine == "torch"

    def _torch_device(self) -> str:
        """`engine=torch` on a non-CUDA device means torch-on-CPU, which is how
        the whole CUDA code path gets exercised without a GPU."""
        d = self.cfg.device.lower()
        return d if d.startswith("cuda") else "cpu"

    # -- engine -------------------------------------------------------------

    def build_engine(self) -> None:
        from ..model.numpy_engine import NumpyEngine
        from ..model.weights import load_gpt2, synthetic_gpt2

        if self.weights_dir and Path(self.weights_dir).exists():
            weights, cfg = load_gpt2(Path(self.weights_dir))
            self.model_cfg = cfg
        else:
            weights = synthetic_gpt2(self.model_cfg, seed=1234)

        want = self.cfg.engine
        dev = self.cfg.device.upper()

        if self._wants_torch():
            import os

            from ..model.paged_attn import which_backend
            from ..model.torch_engine import TorchEngine

            tdev = self._torch_device()
            self.engine = TorchEngine(weights, self.model_cfg, device=tdev)
            backend = which_backend()
            # Take the fused path only when the kernel really compiled. The torch
            # fallback is correct but slower than gathering, so defaulting to it
            # would quietly make things worse and still look like "paged". The
            # env var forces it anyway, which is how the CPU tests cover this
            # code path end to end.
            self.paged_decode = backend == "cuda" or bool(
                os.environ.get("HETEROSERVE_FORCE_PAGED")
            )
            self.engine_name = f"torch:{tdev}[{backend}]"
            return

        if want in ("auto", "openvino") and dev in ("GPU", "NPU", "CPU"):
            try:
                from ..model.ov_engine import OpenVINOEngine

                self.engine = OpenVINOEngine(
                    weights, self.model_cfg, device=dev,
                    max_batch=self.cfg.max_batch,
                    bucket=self.cfg.max_prefill_tokens,
                    max_ctx=self.max_ctx,
                )
                built = self.engine.warmup()
                if built:
                    print(f"[{self.cfg.worker_id}] precompiled {len(built)} "
                          f"{dev} shape buckets", flush=True)
                self.engine_name = f"openvino:{dev}"
                return
            except Exception as exc:  # noqa: BLE001 - fall back, but say why
                if want == "openvino":
                    raise
                print(f"[{self.cfg.worker_id}] OpenVINO on {dev} unavailable ({exc}); using numpy")

        self.engine = NumpyEngine(weights, self.model_cfg, device=dev)
        self.engine_name = f"numpy:{dev}"

    # -- lifecycle ----------------------------------------------------------

    async def _setup_dist(self) -> None:
        """Join the KV process group, if one was configured.

        Failure here is not fatal: a worker that cannot rendezvous simply keeps
        migrating over shaped TCP, which is correct, just slower on hardware
        that could have done it directly.
        """
        if self.dist_state == "off":
            return
        import concurrent.futures

        from ..net.kvlink import DistLink

        self.dist = DistLink(
            rank=self.rank,
            world_size=self.world_size,
            init_method=self.dist_url,
            device=self._torch_device() if self._wants_torch() else "cpu",
            backend=self.dist_backend,
        )
        # One dedicated thread: torch.distributed point-to-point ops are
        # blocking and must not run on the event loop, and NCCL wants its
        # collectives issued in a consistent order from one thread.
        self._dist_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"kvlink-{self.cfg.worker_id}")
        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(self._dist_pool, self.dist.start)
        self.dist_state = "ready" if ok else "failed"
        if not ok:
            print(f"[{self.cfg.worker_id}] KV device link unavailable "
                  f"({self.dist.error}); using shaped TCP", flush=True)

    def _can_use_dist(self, peer: dict) -> bool:
        """Both ends on the group, both holding device-resident KV."""
        return (
            self.dist is not None
            and self.dist_state == "ready"
            and peer.get("rank") is not None
            and hasattr(self.alloc, "export_blocks_device")
        )

    async def _run_dist(self, fn):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._dist_pool, fn)

    async def start(self) -> int:
        self.build_engine()
        self.server = await asyncio.start_server(
            self._on_connection, self.cfg.host, self.cfg.port
        )
        self.port = self.server.sockets[0].getsockname()[1]
        asyncio.create_task(self._engine_loop())
        # Deliberately not awaited: the process group cannot form until every
        # worker exists, and the router only spawns the next one after this one
        # reports ready. Awaiting here deadlocks until the rendezvous times out
        # and silently drops everyone back to TCP.
        asyncio.create_task(self._setup_dist())
        return self.port

    async def run_forever(self) -> None:
        await self._stop.wait()
        if self.server:
            self.server.close()

    async def _on_connection(self, reader, writer) -> None:
        # Inbound channels are unshaped: the link budget models *egress*, and
        # the control plane (router <-> worker) is deliberately kept fast so the
        # benchmark isolates one variable — the cost of moving KV between
        # accelerators. Only the outbound peer connection in `_on_push_prefix`
        # pays the shaped cost.
        ch = Channel(reader, writer, link=None, name=f"{self.cfg.worker_id}<-peer")
        try:
            meta, blob = await ch.recv()
        except Exception:
            await ch.close()
            return

        kind = meta.get("kind")
        if kind == P.HELLO:
            self.router = ch
            await ch.send(
                {
                    "kind": P.HELLO_ACK,
                    "worker_id": self.cfg.worker_id,
                    "device": self.cfg.device,
                    "engine": self.engine_name,
                    "port": self.port,
                    "num_blocks": self.alloc.num_blocks,
                    "block_size": self.alloc.block_size,
                    "block_bytes": self.alloc.block_bytes(),
                    "max_batch": self.cfg.max_batch,
                }
            )
            await ch.serve(self._on_control)
            self.router = None
        elif kind == P.KV_PUSH:
            await self._on_kv_push(ch, meta, blob)
            await ch.serve(lambda m, b: self._on_kv_push(ch, m, b))
        else:
            await ch.close()

    # -- control plane ------------------------------------------------------

    async def _on_control(self, meta: dict, blob: bytes) -> None:
        kind = meta.get("kind")
        try:
            if kind == P.SUBMIT:
                await self._on_submit(meta)
            elif kind == P.PUSH_PREFIX:
                asyncio.create_task(self._on_push_prefix(meta))
            elif kind == P.SET_PEERS:
                self.peers = {p["worker_id"]: p for p in meta["peers"]}
            elif kind == P.SET_LINK:
                self.link.reconfigure(LinkConfig(**meta["link"]))
            elif kind == P.STATE_REQ:
                await self._emit(self._state_msg())
            elif kind == P.CANCEL:
                async with self._state_lock:
                    self._retire(meta["seq_id"], reason="cancelled")
            elif kind == P.RESET:
                async with self._state_lock:
                    self._reset()
            elif kind == P.SHUTDOWN:
                self._stop.set()
        except Exception as exc:  # noqa: BLE001
            await self._emit(
                {"kind": P.ERROR, "worker_id": self.cfg.worker_id,
                 "error": f"{exc}", "trace": traceback.format_exc()[-800:]}
            )

    async def _emit(self, msg: dict) -> None:
        if self.router is not None:
            try:
                await self.router.send(msg)
            except Exception:
                pass

    async def _on_submit(self, meta: dict) -> None:
        seq = Sequence(
            seq_id=meta["seq_id"],
            prompt_ids=list(meta["prompt_ids"]),
            max_new_tokens=int(meta.get("max_new_tokens", 32)),
            temperature=float(meta.get("temperature", 0.0)),
            top_k=int(meta.get("top_k", 0)),
            seed=int(meta.get("seed", 0)),
            stop_at_eos=bool(meta.get("stop_at_eos", True)),
            t_submit=float(meta.get("t_submit", time.time())),
        )

        async with self._state_lock:
            try:
                a = self.alloc.allocate(seq.prompt_ids)
            except OutOfBlocks:
                await self._emit(
                    {"kind": P.REJECTED, "seq_id": seq.seq_id,
                     "worker_id": self.cfg.worker_id, "reason": "kv_full"}
                )
                return

            seq.block_ids = a.block_ids
            # Keep at least one token to compute so we always have fresh logits.
            seq.cached_prefix_tokens = min(a.num_cached_tokens, len(seq.prompt_ids) - 1)
            seq.n_kv = seq.cached_prefix_tokens
            seq.state = SeqState.PREFILL if seq.n_kv else SeqState.WAITING
            seq.t_admit = time.time()
            self.tokens_saved_by_cache += seq.cached_prefix_tokens

            self.seqs[seq.seq_id] = seq
            self.order.append(seq)

        await self._emit(
            {"kind": P.ADMITTED, "seq_id": seq.seq_id, "worker_id": self.cfg.worker_id,
             "cached_prefix_tokens": seq.cached_prefix_tokens,
             "prompt_tokens": len(seq.prompt_ids), "t_admit": seq.t_admit}
        )
        self._wake.set()

    # -- data plane: KV migration ------------------------------------------

    async def _on_push_prefix(self, meta: dict) -> None:
        """Ship our cached blocks for `tokens` to a peer, paying the link cost."""
        target = self.peers.get(meta["target"])
        if target is None:
            await self._emit({"kind": P.PUSH_DONE, "ok": False,
                              "seq_id": meta.get("seq_id"), "reason": "unknown_peer"})
            return

        tokens = list(meta["tokens"])
        # The target may already hold the first `skip_blocks` blocks of this
        # prefix; shipping them again would just burn link budget.
        skip = int(meta.get("skip_blocks", 0))
        t0 = time.time()
        blocks: list[int] = []
        async with self._state_lock:
            matched, n_tokens = self.alloc.match_prefix(tokens)
            blocks = matched[skip:]
            for b in blocks:
                self.alloc.incref(b)         # pin so nothing can evict them
        # Serialise *outside* the lock. Holding it here meant every migration
        # queued behind a full engine step on both ends -- with GPT-2 that is
        # ~200 ms each way, which swamped the actual transfer and capped
        # effective migration throughput at ~20 MB/s no matter the link speed.
        use_dist = self._can_use_dist(target) and bool(blocks)
        payload = None
        if blocks:
            # On the fast path the payload stays on the device and never
            # crosses PCIe to host memory.
            payload = (self.alloc.export_blocks_device(blocks) if use_dist
                       else self.alloc.export_blocks(blocks))

        if payload is None:
            await self._emit({"kind": P.PUSH_DONE, "ok": False,
                              "seq_id": meta.get("seq_id"), "reason": "no_local_prefix"})
            return

        nbytes = 0
        try:
            ch = await connect(target["host"], target["port"], link=self.link,
                               name=f"{self.cfg.worker_id}->{target['worker_id']}")
            head = {"kind": P.KV_PUSH, "from": self.cfg.worker_id,
                    "tokens": tokens[:n_tokens], "n_tokens": n_tokens,
                    "first_block": skip, "seq_id": meta.get("seq_id")}

            if use_dist:
                # Metadata on the control socket, bulk over the device link. The
                # receiver posts a matching recv when it sees this frame.
                nbytes = payload.numel() * payload.element_size()
                head |= {"via": "dist", "src_rank": self.dist.rank,
                         "shape": list(payload.shape),
                         "dtype": str(payload.dtype).replace("torch.", ""),
                         "bytes": nbytes}
                await ch.send(head)
                await self._run_dist(
                    lambda: self.dist.send_tensor(payload, target["rank"]))
            else:
                desc, blob = encode_array(payload)
                nbytes = len(blob)
                head |= {"via": "tcp", "desc": desc}
                await ch.send(head, blob)

            ack, _ = await ch.recv()
            await ch.close()
            ok = bool(ack.get("ok"))
            self.migrations_out += 1
            self.bytes_migrated_out += nbytes
        except Exception as exc:  # noqa: BLE001
            ok = False
            await self._emit({"kind": P.ERROR, "worker_id": self.cfg.worker_id,
                              "error": f"push_prefix: {exc}"})
        finally:
            async with self._state_lock:
                for b in blocks:
                    self.alloc.decref(b)

        await self._emit(
            {"kind": P.PUSH_DONE, "ok": ok, "seq_id": meta.get("seq_id"),
             "target": meta["target"], "from": self.cfg.worker_id,
             "n_tokens": n_tokens, "bytes": nbytes,
             "via": "dist" if use_dist else "tcp",
             "elapsed": round(time.time() - t0, 4)}
        )

    async def _on_kv_push(self, ch: Channel, meta: dict, blob: bytes) -> None:
        if meta.get("kind") != P.KV_PUSH:
            return
        try:
            tokens = list(meta["tokens"])
            first_block = int(meta.get("first_block", 0))
            via_dist = meta.get("via") == "dist"

            if via_dist:
                # The sender has already posted its send; post the matching recv
                # and take delivery straight into device memory.
                payload = await self._run_dist(
                    lambda: self.dist.recv_tensor(
                        meta["shape"], meta["dtype"], int(meta["src_rank"]))
                )
                nbytes = int(meta.get("bytes", 0))
            else:
                payload = decode_array(meta["desc"], blob)
                nbytes = len(blob)

            # Reserve under the lock (cheap), copy outside it (expensive).
            async with self._state_lock:
                new_ids = self.alloc.reserve_blocks(int(payload.shape[0]))
            if via_dist:
                self.alloc.write_blocks_device(new_ids, payload)
            else:
                self.alloc.write_blocks(new_ids, payload)
            async with self._state_lock:
                self.alloc.adopt_migrated(
                    tokens, new_ids, meta["n_tokens"], first_index=first_block
                )
                # Nobody owns these yet — park them in the cache as evictable.
                for b in new_ids:
                    self.alloc.decref(b)
            self.migrations_in += 1
            self.bytes_migrated_in += nbytes
            await ch.send({"kind": P.KV_ACK, "ok": True, "blocks": len(new_ids)})
            await self._emit(
                {"kind": P.CACHE_ADD, "worker_id": self.cfg.worker_id,
                 "hashes": [str(h) for h in
                            chain_hashes(tokens[: meta["n_tokens"]], self.alloc.block_size)],
                 "source": "migration"}
            )
        except Exception as exc:  # noqa: BLE001
            await ch.send({"kind": P.KV_ACK, "ok": False, "error": str(exc)})

    # -- engine loop --------------------------------------------------------

    async def _engine_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            if not self._has_work():
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
                continue

            async with self._state_lock:
                events = await loop.run_in_executor(None, self._step)

            for ev in events:
                await self._emit(ev)
            await asyncio.sleep(0)   # let migrations and control traffic through

    def _has_work(self) -> bool:
        return any(s.state != SeqState.DONE for s in self.order)

    def _step(self) -> list[dict]:
        """One scheduler iteration. Runs off the event loop, holds the state lock."""
        t0 = time.perf_counter()
        events: list[dict] = []

        # Time the two phases separately. The router prices placement from
        # these numbers, and mixing them would make a device look slow simply
        # because it happened to be busy with the other phase.
        pending = [s for s in self.order if s.state in (SeqState.WAITING, SeqState.PREFILL)]
        if pending:
            events += self._prefill_chunk(pending[0])
            self.prefill_steps += 1
            self.prefill_busy_s += time.perf_counter() - t0
        else:
            running = [s for s in self.order if s.state == SeqState.RUNNING]
            if not running:
                return events
            events += self._decode(running[: self.cfg.max_batch])
            self.decode_steps += 1
            self.decode_busy_s += time.perf_counter() - t0

        self.steps += 1
        self.device_busy_s += time.perf_counter() - t0
        return events

    # -- phases -------------------------------------------------------------

    def _prefill_chunk(self, seq: Sequence) -> list[dict]:
        events: list[dict] = []
        ctx = seq.ctx
        target = seq.prefill_target
        start = seq.n_kv
        end = min(start + self.cfg.max_prefill_tokens, target)
        if end <= start:
            seq.state = SeqState.RUNNING
            return events

        toks = np.asarray(ctx[start:end], dtype=np.int64)

        # Reserve the blocks before computing anything: the fused path writes
        # straight into the pool, and running out afterwards would waste the
        # whole chunk's compute.
        try:
            self.alloc.grow(seq.block_ids, end)
        except OutOfBlocks:
            return self._preempt(seq, events)

        if self.paged_decode:
            # Fused: the chunk's K/V goes into the pool and attention walks the
            # block table from there, so the cached prefix is never gathered.
            logits = self.engine.prefill_paged(toks, start, seq.block_ids, self.alloc)
        else:
            past_k = past_v = None
            if start:
                past_k, past_v = self.alloc.gather_kv(seq.block_ids, start)
            logits, k, v = self.engine.prefill(toks, start, past_k, past_v)
            self.alloc.write_kv(seq.block_ids, start, k, v)

        seq.n_kv = end
        seq.prefill_tokens_computed += end - start
        self.tokens_prefilled += end - start
        seq.state = SeqState.PREFILL

        if seq.n_kv >= target:
            # Publish the completed blocks so other sequences (and other
            # workers) can reuse this prefix.
            self.alloc.register_full_blocks(ctx[: seq.n_kv], seq.block_ids)
            hashes = chain_hashes(ctx[: seq.n_kv], self.alloc.block_size)
            if hashes:
                events.append(
                    {"kind": P.CACHE_ADD, "worker_id": self.cfg.worker_id,
                     "hashes": [str(h) for h in hashes], "source": "prefill"}
                )

            if not seq.output_ids:
                rng = np.random.default_rng(seq.seed)
                tok = sample_token(logits, seq.temperature, seq.top_k, rng)
                seq.output_ids.append(tok)
                seq.t_first_token = time.time()
                self.tokens_generated += 1
                events.append(self._token_event(seq, tok, first=True))
                if self._maybe_finish(seq, tok, events):
                    return events
            seq.state = SeqState.RUNNING

        return events

    def _decode(self, batch: list[Sequence]) -> list[dict]:
        events: list[dict] = []

        # Make room before touching the device.
        keep: list[Sequence] = []
        for seq in batch:
            try:
                self.alloc.grow(seq.block_ids, seq.n_kv + 1)
                keep.append(seq)
            except OutOfBlocks:
                self._preempt(seq, events)
        if not keep:
            return events

        toks = np.array([s.ctx[s.n_kv] for s in keep], dtype=np.int64)
        pos = np.array([s.n_kv for s in keep], dtype=np.int64)

        if self.paged_decode:
            # Fused path: attention walks each sequence's block table inside the
            # kernel and the new K/V is written straight into the pool, so there
            # is no gather and nothing to write back here.
            logits = self.engine.decode_batch_paged(
                toks, pos, [s.block_ids for s in keep], pos, self.alloc
            )
            k_new = v_new = None
        else:
            pasts = [self.alloc.gather_kv(s.block_ids, s.n_kv) for s in keep]
            logits, k_new, v_new = self.engine.decode_batch(
                toks, pos, [p[0] for p in pasts], [p[1] for p in pasts]
            )

        for i, seq in enumerate(keep):
            if k_new is not None:
                self.alloc.write_kv(seq.block_ids, seq.n_kv, k_new[i], v_new[i])
            seq.n_kv += 1
            rng = np.random.default_rng(seq.seed + seq.n_kv)
            tok = sample_token(logits[i], seq.temperature, seq.top_k, rng)
            seq.output_ids.append(tok)
            self.tokens_generated += 1
            if not seq.t_first_token:
                seq.t_first_token = time.time()
            events.append(self._token_event(seq, tok))
            self._maybe_finish(seq, tok, events)

        return events

    # -- bookkeeping --------------------------------------------------------

    def _token_event(self, seq: Sequence, tok: int, first: bool = False) -> dict:
        return {
            "kind": P.TOKEN, "seq_id": seq.seq_id, "worker_id": self.cfg.worker_id,
            "token": int(tok), "index": len(seq.output_ids) - 1,
            "first": first or len(seq.output_ids) == 1, "t": time.time(),
        }

    def _maybe_finish(self, seq: Sequence, tok: int, events: list[dict]) -> bool:
        done = seq.finished or (seq.stop_at_eos and tok == self.eos_token_id)
        if not done:
            return False
        seq.state = SeqState.DONE
        seq.t_done = time.time()
        # Register everything we generated too — enables multi-turn reuse.
        self.alloc.register_full_blocks(seq.ctx[: seq.n_kv], seq.block_ids)
        events.append(
            {
                "kind": P.FINISHED, "seq_id": seq.seq_id, "worker_id": self.cfg.worker_id,
                "output_ids": [int(t) for t in seq.output_ids],
                "prompt_tokens": len(seq.prompt_ids),
                "generated_tokens": len(seq.output_ids),
                "cached_prefix_tokens": seq.cached_prefix_tokens,
                "prefill_tokens_computed": seq.prefill_tokens_computed,
                "preemptions": seq.preemptions,
                "t_submit": seq.t_submit, "t_admit": seq.t_admit,
                "t_first_token": seq.t_first_token, "t_done": seq.t_done,
            }
        )
        self._release(seq)
        return True

    def _preempt(self, victim: Sequence, events: list[dict]) -> list[dict]:
        """Recompute-style preemption: drop the KV, keep the tokens, requeue."""
        running = [s for s in self.order if s.state == SeqState.RUNNING]
        target = running[-1] if running else victim
        self.alloc.free_sequence(target.block_ids)
        target.block_ids = []
        target.n_kv = 0
        target.state = SeqState.WAITING
        target.preemptions += 1
        try:
            a = self.alloc.allocate(target.ctx)
            target.block_ids = a.block_ids
            target.n_kv = min(a.num_cached_tokens, target.prefill_target)
        except OutOfBlocks:
            pass
        events.append(
            {"kind": P.PREEMPTED, "seq_id": target.seq_id,
             "worker_id": self.cfg.worker_id, "count": target.preemptions}
        )
        return events

    def _release(self, seq: Sequence) -> None:
        if seq.block_ids:
            self.alloc.free_sequence(seq.block_ids)
            seq.block_ids = []
        self.order = [s for s in self.order if s.seq_id != seq.seq_id]
        self.seqs.pop(seq.seq_id, None)

    def _retire(self, seq_id: str, reason: str) -> None:
        seq = self.seqs.get(seq_id)
        if seq:
            seq.state = SeqState.DONE
            self._release(seq)

    def _reset(self) -> None:
        for seq in list(self.order):
            self._release(seq)
        self.order.clear()
        self.seqs.clear()
        self.alloc = self._new_allocator()
        self.steps = self.prefill_steps = self.decode_steps = 0
        self.device_busy_s = 0.0
        self.prefill_busy_s = 0.0
        self.decode_busy_s = 0.0
        self.tokens_generated = self.tokens_prefilled = self.tokens_saved_by_cache = 0
        self.migrations_in = self.migrations_out = 0
        self.bytes_migrated_in = self.bytes_migrated_out = 0

    def _state_msg(self) -> dict:
        return {
            "kind": P.STATE,
            "worker_id": self.cfg.worker_id,
            "device": self.cfg.device,
            "engine": self.engine_name,
            "kv": self.alloc.snapshot(),
            "queued": sum(1 for s in self.order if s.state in (SeqState.WAITING, SeqState.PREFILL)),
            "running": sum(1 for s in self.order if s.state == SeqState.RUNNING),
            "steps": self.steps,
            "prefill_steps": self.prefill_steps,
            "decode_steps": self.decode_steps,
            "device_busy_s": round(self.device_busy_s, 4),
            "prefill_busy_s": round(self.prefill_busy_s, 4),
            "decode_busy_s": round(self.decode_busy_s, 4),
            "tokens_generated": self.tokens_generated,
            "tokens_prefilled": self.tokens_prefilled,
            "tokens_saved_by_cache": self.tokens_saved_by_cache,
            "migrations_in": self.migrations_in,
            "migrations_out": self.migrations_out,
            "bytes_migrated_in": self.bytes_migrated_in,
            "bytes_migrated_out": self.bytes_migrated_out,
            "link": self.link.stats.as_dict(),
            "kvlink": ({**self.dist.info(), "state": self.dist_state} if self.dist
                       else {"backend": "tcp", "state": self.dist_state}),
            "t": time.time(),
        }


# ---------------------------------------------------------------------------
# process entry point
# ---------------------------------------------------------------------------


async def _amain(args) -> None:
    from ..config import KVConfig

    cfg = WorkerConfig(
        worker_id=args.worker_id,
        device=args.device,
        engine=args.engine,
        kv=KVConfig(block_size=args.block_size, num_blocks=args.num_blocks),
        max_batch=args.max_batch,
        max_prefill_tokens=args.max_prefill_tokens,
        port=args.port,
    )
    model_cfg = ModelConfig() if args.model == "gpt2" else ModelConfig.tiny()
    weights = Path(args.weights) if args.weights and args.model == "gpt2" else None

    w = Worker(cfg, model_cfg, weights, LinkConfig(), seed=args.seed,
               max_ctx=args.max_ctx,
               rank=args.rank, world_size=args.world_size,
               dist_url=args.dist_url, dist_backend=args.dist_backend)
    port = await w.start()
    # The parent reads this line to learn the port.
    print(f"WORKER_READY {args.worker_id} {port} {w.engine_name}", flush=True)
    await w.run_forever()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", dest="worker_id", required=True)
    ap.add_argument("--device", default="CPU")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--weights", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--block-size", dest="block_size", type=int, default=16)
    ap.add_argument("--num-blocks", dest="num_blocks", type=int, default=512)
    ap.add_argument("--max-batch", dest="max_batch", type=int, default=8)
    ap.add_argument("--max-prefill-tokens", dest="max_prefill_tokens", type=int, default=256)
    ap.add_argument("--max-ctx", dest="max_ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", dest="world_size", type=int, default=1)
    ap.add_argument("--dist-url", dest="dist_url", default="",
                    help="torch.distributed rendezvous, e.g. tcp://127.0.0.1:29677")
    ap.add_argument("--dist-backend", dest="dist_backend", default="auto",
                    help="auto | nccl | gloo")
    asyncio.run(_amain(ap.parse_args()))


if __name__ == "__main__":
    main()
