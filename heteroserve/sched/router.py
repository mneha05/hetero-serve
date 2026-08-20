"""The control plane: places requests, and decides when KV should move.

The interesting decision lives in `_place_cache_aware`. For each candidate
worker it compares two futures:

    stay        pay to recompute the prefix locally
    migrate     pay to drag the prefix across the link, then recompute the rest

Both are priced in seconds using per-device speeds the router *measures* at
startup (an Arc GPU and an NPU do not prefill at the same rate, so a hardcoded
constant would be fiction), plus the link's bandwidth/latency budget. The
crossover between those two terms is the whole experiment: at 10 Gbps moving
18 MB of KV is nearly free, at 50 Mbps it is slower than recomputing from
scratch, and somewhere in between the right answer flips.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ClusterConfig, LinkConfig, WorkerConfig
from ..kv.blocks import chain_hashes
from ..metrics import RequestRecord
from ..net.transport import Channel, connect
from ..worker import protocol as P

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Request:
    prompt_ids: list[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    seed: int = 0
    tag: str = ""
    req_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Placement:
    worker_id: str
    donor_id: str | None = None
    skip_blocks: int = 0
    hit_tokens: int = 0
    reason: str = ""
    est_cost: float = 0.0


@dataclass
class WorkerHandle:
    cfg: WorkerConfig
    ch: Channel
    proc: asyncio.subprocess.Process | None = None
    device: str = "CPU"
    engine: str = ""
    host: str = "127.0.0.1"
    port: int = 0
    num_blocks: int = 0
    block_size: int = 16
    block_bytes: int = 0
    max_batch: int = 8

    # live view, refreshed from STATE messages and local bookkeeping
    queued: int = 0
    running: int = 0
    kv_util: float = 0.0
    inflight: int = 0
    pending_prefill_tokens: int = 0
    hashes: set[str] = field(default_factory=set)
    last_state: dict = field(default_factory=dict)

    # measured performance model (seconds)
    t_prefill_per_token: float = 1.5e-3
    t_decode_step: float = 3.0e-2

    @property
    def worker_id(self) -> str:
        return self.cfg.worker_id

    def queue_cost(self) -> float:
        """Rough seconds of work already committed to this device.

        First-order estimate: the prefill tokens still owed, plus one decode
        step per in-flight sequence (decode is batched, so each extra sequence
        costs roughly one marginal step). Uses `inflight`, which the router
        maintains itself, rather than `running`, which only refreshes when
        someone polls a worker for state and is therefore stale mid-run.
        """
        return (
            self.pending_prefill_tokens * self.t_prefill_per_token
            + self.inflight * self.t_decode_step
        )


class Router:
    def __init__(
        self,
        cluster: ClusterConfig,
        weights_dir: Path | None = None,
        model_name: str = "gpt2",
        verbose: bool = False,
        max_ctx: int = 512,
    ):
        self.max_ctx = max_ctx
        self.cluster = cluster
        self.weights_dir = weights_dir
        self.model_name = model_name
        self.verbose = verbose

        self.workers: dict[str, WorkerHandle] = {}
        self.index: dict[str, set[str]] = {}     # prefix hash -> worker ids
        self.records: dict[str, RequestRecord] = {}
        self._futures: dict[str, asyncio.Future] = {}
        # req_id -> (request, workers that already rejected it). Needed so a
        # rejection can actually be re-dispatched rather than just recorded.
        self._pending: dict[str, tuple] = {}
        self._push_futures: dict[str, asyncio.Future] = {}
        self._rr = 0
        self._event_hooks: list = []
        self.migrations = 0
        self.migration_bytes = 0
        self.migration_s = 0.0
        self.last_push: dict = {}      # most recent PUSH_DONE, for tests/telemetry
        # Bytes committed to the wire but not yet delivered. The placement cost
        # model queues new transfers behind this, so a burst of individually
        # cheap migrations cannot all price themselves against an idle link.
        self.inflight_migration_bytes = 0

    def log(self, *a) -> None:
        if self.verbose:
            print("[router]", *a, flush=True)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        # Every worker gets a rank up front so the KV process group can rendezvous
        # while they boot. A group of one has nothing to migrate to, so we do not
        # bother forming one.
        self._ranks = {w.worker_id: i for i, w in enumerate(self.cluster.workers)}
        self._world = len(self.cluster.workers)
        if self._world >= 2 and self.cluster.kv_transport != "tcp":
            from ..net.kvlink import default_init_method

            self._dist_url = default_init_method(self.cluster.dist_port)
        else:
            self._dist_url = ""

        for wcfg in self.cluster.workers:
            await self._spawn(wcfg)
        await self._broadcast_peers()
        await self._await_kv_links()
        self.log(f"cluster up: {[f'{h.worker_id}({h.engine})' for h in self.workers.values()]}")

    async def _spawn(self, wcfg: WorkerConfig) -> None:
        args = [
            sys.executable, "-m", "heteroserve.worker.worker",
            "--worker-id", wcfg.worker_id,
            "--device", wcfg.device,
            "--engine", wcfg.engine,
            "--model", self.model_name,
            "--block-size", str(wcfg.kv.block_size),
            "--num-blocks", str(wcfg.kv.num_blocks),
            "--max-batch", str(wcfg.max_batch),
            "--max-prefill-tokens", str(wcfg.max_prefill_tokens),
            "--max-ctx", str(self.max_ctx),
            "--seed", str(self.cluster.seed),
            "--rank", str(self._ranks[wcfg.worker_id]),
            "--world-size", str(self._world),
            "--dist-url", self._dist_url,
            "--dist-backend", self.cluster.kv_transport
            if self.cluster.kv_transport in ("nccl", "gloo") else "auto",
        ]
        if self.weights_dir:
            args += ["--weights", str(self.weights_dir)]

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )

        port = None
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise RuntimeError(f"worker {wcfg.worker_id} died before becoming ready")
            text = line.decode(errors="replace").rstrip()
            if text.startswith("WORKER_READY"):
                port = int(text.split()[2])
                break
            self.log(f"{wcfg.worker_id}: {text}")

        asyncio.create_task(self._drain(proc, wcfg.worker_id))

        ch = await connect(wcfg.host, port, name=f"router->{wcfg.worker_id}")
        await ch.send({"kind": P.HELLO})
        ack, _ = await ch.recv()

        h = WorkerHandle(
            cfg=wcfg, ch=ch, proc=proc,
            device=ack["device"], engine=ack["engine"], host=wcfg.host, port=port,
            num_blocks=ack["num_blocks"], block_size=ack["block_size"],
            block_bytes=ack["block_bytes"], max_batch=ack["max_batch"],
        )
        self.workers[wcfg.worker_id] = h
        asyncio.create_task(ch.serve(self._on_event))

    async def _drain(self, proc, wid: str) -> None:
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            self.log(f"{wid}: {line.decode(errors='replace').rstrip()}")

    async def _await_kv_links(self, timeout: float = 25.0) -> None:
        """Block until every worker's KV transport has settled.

        Without this a request can be routed while a worker is still deciding
        whether it has a device link, so the same migration would take different
        paths run to run -- which is exactly the kind of nondeterminism that
        makes a benchmark untrustworthy.
        """
        if not self._dist_url:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            await self.poll_states()
            await asyncio.sleep(0.2)
            states = [
                (h.last_state.get("kvlink") or {}).get("state", "pending")
                for h in self.workers.values()
            ]
            if states and all(s != "pending" for s in states):
                ready = sum(1 for s in states if s == "ready")
                backends = {(h.last_state.get("kvlink") or {}).get("backend")
                            for h in self.workers.values()}
                self.log(f"KV transport: {ready}/{len(states)} on the device link "
                         f"({', '.join(sorted(str(b) for b in backends))})")
                return
        self.log("KV transport: rendezvous timed out; continuing on shaped TCP")

    async def _broadcast_peers(self) -> None:
        peers = [
            {"worker_id": h.worker_id, "host": h.host, "port": h.port,
             "rank": self._ranks.get(h.worker_id)}
            for h in self.workers.values()
        ]
        for h in self.workers.values():
            await h.ch.send({"kind": P.SET_PEERS, "peers": peers})

    async def set_link(self, link: LinkConfig) -> None:
        self.cluster.link = link
        payload = {
            "bandwidth_mbps": link.bandwidth_mbps,
            "latency_ms": link.latency_ms,
            "jitter_ms": link.jitter_ms,
            "loss": link.loss,
        }
        for h in self.workers.values():
            await h.ch.send({"kind": P.SET_LINK, "link": payload})

    async def reset(self) -> None:
        """Clear all worker state between benchmark runs."""
        for h in self.workers.values():
            await h.ch.send({"kind": P.RESET})
            h.hashes.clear()
            h.inflight = h.queued = h.running = 0
            h.pending_prefill_tokens = 0
        self.index.clear()
        self.records.clear()
        self._pending.clear()
        self.migrations = 0
        self.migration_bytes = 0
        self.migration_s = 0.0
        self.inflight_migration_bytes = 0
        await asyncio.sleep(0.05)

    async def stop(self) -> None:
        for h in self.workers.values():
            with contextlib.suppress(Exception):
                await h.ch.send({"kind": P.SHUTDOWN})
        await asyncio.sleep(0.1)
        for h in self.workers.values():
            with contextlib.suppress(Exception):
                await h.ch.close()
            if h.proc and h.proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    h.proc.terminate()
        for h in self.workers.values():
            if h.proc:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(h.proc.wait(), timeout=5)

    # -- placement ----------------------------------------------------------

    def _block_hashes(self, prompt_ids: list[int]) -> list[str]:
        bs = next(iter(self.workers.values())).block_size if self.workers else 16
        return [str(h) for h in chain_hashes(prompt_ids, bs)]

    def _hits(self, hashes: list[str]) -> dict[str, int]:
        """Blocks of the prompt each worker already holds, counted from the front."""
        out: dict[str, int] = {}
        for wid, h in self.workers.items():
            n = 0
            for hh in hashes:
                if hh in h.hashes:
                    n += 1
                else:
                    break
            out[wid] = n
        return out

    def place(self, prompt_ids: list[int]) -> Placement:
        policy = self.cluster.policy
        if policy == "round_robin":
            ids = list(self.workers)
            wid = ids[self._rr % len(ids)]
            self._rr += 1
            return Placement(wid, reason="round_robin")

        hashes = self._block_hashes(prompt_ids)
        hits = self._hits(hashes)

        if policy == "least_loaded":
            wid = min(self.workers, key=lambda w: (self.workers[w].queue_cost(), w))
            return Placement(wid, hit_tokens=hits[wid] * self.workers[wid].block_size,
                             reason="least_loaded")

        if policy == "prefix_affinity":
            wid = max(self.workers, key=lambda w: (hits[w], -self.workers[w].queue_cost()))
            return Placement(wid, hit_tokens=hits[wid] * self.workers[wid].block_size,
                             reason=f"affinity hit={hits[wid]}blk")

        return self._place_cache_aware(prompt_ids, hits)

    def _place_cache_aware(self, prompt_ids: list[int], hits: dict[str, int]) -> Placement:
        prompt_len = len(prompt_ids)
        link = self.cluster.link
        best: Placement | None = None

        for wid, h in self.workers.items():
            bs = h.block_size
            local_blocks = hits[wid]
            local_tokens = local_blocks * bs
            queue = h.queue_cost()

            # Option A: run here, recompute whatever we don't already hold.
            cost = queue + max(0, prompt_len - local_tokens) * h.t_prefill_per_token
            cand = Placement(
                wid, hit_tokens=local_tokens, est_cost=cost,
                reason=f"local hit {local_tokens}tok",
            )

            # Option B: pull the missing part of the prefix from the best donor.
            if self.cluster.enable_migration:
                donors = [d for d in self.workers if d != wid]
                if donors:
                    donor = max(donors, key=lambda d: hits[d])
                    if hits[donor] > local_blocks:
                        delta_blocks = hits[donor] - local_blocks
                        mig_bytes = delta_blocks * h.block_bytes
                        # Price this transfer *behind whatever is already on the
                        # wire*. Costing it against an idle link is how the first
                        # version of this scheduler wrecked its own tail latency:
                        # at marginal bandwidth it green-lit a burst of
                        # migrations that each looked cheap in isolation, then
                        # they all queued on the same egress token bucket and
                        # p99 TTFT went to 3.3 s. See README "what went wrong".
                        mig_s = link.transfer_seconds(
                            mig_bytes + self.inflight_migration_bytes
                        )
                        donor_tokens = hits[donor] * bs
                        alt = (
                            queue + mig_s
                            + max(0, prompt_len - donor_tokens) * h.t_prefill_per_token
                        )
                        if alt < cost:
                            cost = alt
                            cand = Placement(
                                wid, donor_id=donor, skip_blocks=local_blocks,
                                hit_tokens=donor_tokens, est_cost=alt,
                                reason=(
                                    f"migrate {delta_blocks}blk "
                                    f"({mig_bytes/1e6:.1f}MB, est {mig_s*1e3:.0f}ms) "
                                    f"from {donor}"
                                ),
                            )

            if best is None or cand.est_cost < best.est_cost:
                best = cand

        assert best is not None
        return best

    # -- submission ---------------------------------------------------------

    async def submit(self, req: Request) -> RequestRecord:
        fut = await self.submit_nowait(req)
        return await fut

    async def submit_nowait(self, req: Request, worker_id: str | None = None) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._futures[req.req_id] = fut

        placement = (
            Placement(worker_id, reason="pinned") if worker_id else self.place(req.prompt_ids)
        )
        rec = RequestRecord(
            req_id=req.req_id,
            worker_id=placement.worker_id,
            prompt_tokens=len(req.prompt_ids),
            placement_reason=placement.reason,
            t_submit=time.time(),
        )
        self.records[req.req_id] = rec
        self._pending[req.req_id] = (req, set())

        asyncio.create_task(self._dispatch(req, placement, rec))
        return fut

    async def _dispatch(self, req: Request, placement: Placement, rec: RequestRecord) -> None:
        h = self.workers[placement.worker_id]

        if placement.donor_id:
            ok = await self._migrate(req, placement, rec)
            if not ok:
                rec.placement_reason += " (migration failed, recomputing)"

        h.inflight += 1
        # Track the exact amount we queued so the decrement can match it. Using
        # the full prompt length here and subtracting it later would drive the
        # counter to zero and silently destroy the load signal.
        rec.queued_prefill_tokens = max(0, len(req.prompt_ids) - placement.hit_tokens)
        h.pending_prefill_tokens += rec.queued_prefill_tokens
        await h.ch.send(
            {
                "kind": P.SUBMIT,
                "seq_id": req.req_id,
                "prompt_ids": req.prompt_ids,
                "max_new_tokens": req.max_new_tokens,
                "temperature": req.temperature,
                "top_k": req.top_k,
                "seed": req.seed,
                "t_submit": rec.t_submit,
            }
        )

    async def _migrate(self, req: Request, placement: Placement, rec: RequestRecord) -> bool:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._push_futures[req.req_id] = fut
        donor = self.workers[placement.donor_id]

        t0 = time.time()
        reserved = placement.hit_tokens // donor.block_size * donor.block_bytes
        self.inflight_migration_bytes += reserved
        await donor.ch.send(
            {
                "kind": P.PUSH_PREFIX,
                "seq_id": req.req_id,
                "tokens": req.prompt_ids,
                "target": placement.worker_id,
                "skip_blocks": placement.skip_blocks,
            }
        )
        try:
            result = await asyncio.wait_for(fut, timeout=120)
        except asyncio.TimeoutError:
            self._push_futures.pop(req.req_id, None)
            self.inflight_migration_bytes = max(0, self.inflight_migration_bytes - reserved)
            return False

        self.inflight_migration_bytes = max(0, self.inflight_migration_bytes - reserved)
        elapsed = time.time() - t0
        if result.get("ok"):
            rec.migrated = True
            rec.migration_bytes = int(result.get("bytes", 0))
            rec.migration_s = elapsed
            self.migrations += 1
            self.migration_bytes += rec.migration_bytes
            self.migration_s += elapsed
            return True
        return False

    # -- events -------------------------------------------------------------

    async def _on_event(self, meta: dict, blob: bytes) -> None:
        kind = meta.get("kind")
        wid = meta.get("worker_id", "")
        h = self.workers.get(wid)

        if kind == P.CACHE_ADD and h is not None:
            for hh in meta["hashes"]:
                h.hashes.add(hh)
                self.index.setdefault(hh, set()).add(wid)

        elif kind == P.CACHE_DROP and h is not None:
            for hh in meta["hashes"]:
                h.hashes.discard(hh)
                s = self.index.get(hh)
                if s:
                    s.discard(wid)

        elif kind == P.ADMITTED:
            rec = self.records.get(meta["seq_id"])
            if rec:
                rec.worker_id = wid
                rec.cached_prefix_tokens = meta["cached_prefix_tokens"]
                rec.t_admit = meta["t_admit"]

        elif kind == P.TOKEN:
            rec = self.records.get(meta["seq_id"])
            if rec and meta.get("first") and not rec.t_first_token:
                rec.t_first_token = meta["t"]
                if h:
                    h.pending_prefill_tokens = max(
                        0, h.pending_prefill_tokens - rec.queued_prefill_tokens
                    )
                    rec.queued_prefill_tokens = 0

        elif kind == P.PREEMPTED:
            rec = self.records.get(meta["seq_id"])
            if rec:
                rec.preemptions = meta["count"]

        elif kind == P.REJECTED:
            await self._handle_rejection(meta)
            return

        elif kind == P.FINISHED:
            self._finish(meta, h)

        elif kind == P.PUSH_DONE:
            self.last_push = meta
            fut = self._push_futures.pop(meta.get("seq_id", ""), None)
            if fut and not fut.done():
                fut.set_result(meta)

        elif kind == P.STATE and h is not None:
            h.last_state = meta
            h.queued = meta["queued"]
            h.running = meta["running"]
            h.kv_util = meta["kv"]["utilisation"]
            self._update_from_device_time(h, meta)

        elif kind == P.ERROR:
            print(f"[router] worker error {wid}: {meta.get('error')}", flush=True)

        for hook in self._event_hooks:
            with contextlib.suppress(Exception):
                await hook(meta)

    def _finish(self, meta: dict, h: WorkerHandle | None) -> None:
        rec = self.records.get(meta["seq_id"])
        if rec is None:
            return
        rec.worker_id = meta["worker_id"]
        rec.generated_tokens = meta["generated_tokens"]
        rec.output_ids = list(meta.get("output_ids", []))
        rec.cached_prefix_tokens = meta["cached_prefix_tokens"]
        rec.prefill_tokens_computed = meta["prefill_tokens_computed"]
        rec.preemptions = meta["preemptions"]
        rec.t_admit = meta["t_admit"] or rec.t_admit
        rec.t_first_token = meta["t_first_token"] or rec.t_first_token
        rec.t_done = meta["t_done"]

        if h is not None:
            h.inflight = max(0, h.inflight - 1)
            if rec.queued_prefill_tokens:
                h.pending_prefill_tokens = max(
                    0, h.pending_prefill_tokens - rec.queued_prefill_tokens
                )
                rec.queued_prefill_tokens = 0
            self._update_perf_model(h, rec)

        self._pending.pop(meta["seq_id"], None)
        fut = self._futures.pop(meta["seq_id"], None)
        if fut and not fut.done():
            fut.set_result(rec)

    def _update_from_device_time(self, h: WorkerHandle, state: dict) -> None:
        """Price a device from how long *it* was busy, not from request latency.

        Deriving ms/token from (first_token - admit) folds in queueing, and
        deriving ms/step from inter-token gaps folds in prefill contention, so
        both move with load rather than with the hardware. Calibration numbers
        swung 4-6x run to run until this replaced them. The worker times each
        phase around its own engine call; queueing is modelled separately in
        `queue_cost()`.
        """
        toks = state.get("tokens_prefilled", 0)
        pre_s = state.get("prefill_busy_s", 0.0)
        if toks > 32 and pre_s > 0:
            h.t_prefill_per_token = pre_s / toks

        steps = state.get("decode_steps", 0)
        dec_s = state.get("decode_busy_s", 0.0)
        if steps > 4 and dec_s > 0:
            h.t_decode_step = dec_s / steps

    def _update_perf_model(self, h: WorkerHandle, rec: RequestRecord) -> None:
        """Fallback estimate from request latency, used only until a worker has
        reported enough device time for `_update_from_device_time` to take over."""
        if h.last_state.get("tokens_prefilled", 0) > 32:
            return
        alpha = 0.3
        if rec.prefill_tokens_computed > 0 and rec.t_first_token and rec.t_admit:
            per_tok = (rec.t_first_token - rec.t_admit) / rec.prefill_tokens_computed
            if 0 < per_tok < 1.0:
                h.t_prefill_per_token = (1 - alpha) * h.t_prefill_per_token + alpha * per_tok
        if rec.generated_tokens > 1 and rec.t_done and rec.t_first_token:
            per_step = (rec.t_done - rec.t_first_token) / (rec.generated_tokens - 1)
            if 0 < per_step < 10.0:
                h.t_decode_step = (1 - alpha) * h.t_decode_step + alpha * per_step

    async def _handle_rejection(self, meta: dict) -> None:
        """A worker ran out of KV at admission — actually re-dispatch elsewhere.

        Recording the new worker without re-sending the request leaves the
        caller awaiting a future that nothing will ever resolve, which is a
        hang rather than an error. Every worker that says no is remembered so
        we cannot bounce a request between the same two devices forever.
        """
        req_id = meta["seq_id"]
        rec = self.records.get(req_id)
        entry = self._pending.get(req_id)
        if rec is None or entry is None:
            return

        req, refused = entry
        refused.add(meta["worker_id"])
        h = self.workers.get(meta["worker_id"])
        if h is not None:
            h.inflight = max(0, h.inflight - 1)
            h.pending_prefill_tokens = max(
                0, h.pending_prefill_tokens - rec.queued_prefill_tokens
            )
            rec.queued_prefill_tokens = 0

        alternatives = [w for w in self.workers if w not in refused]
        if not alternatives:
            self._pending.pop(req_id, None)
            fut = self._futures.pop(req_id, None)
            if fut and not fut.done():
                fut.set_exception(
                    RuntimeError(f"no worker could admit {req_id}: KV pool exhausted")
                )
            return

        target = min(alternatives, key=lambda w: self.workers[w].queue_cost())
        rec.worker_id = target
        rec.placement_reason += f" -> requeued on {target} (kv_full)"
        self.log(f"{req_id} rejected by {meta['worker_id']}, retrying on {target}")
        hits = self._hits(self._block_hashes(req.prompt_ids))
        await self._dispatch(
            req,
            Placement(target, hit_tokens=hits.get(target, 0) * self.workers[target].block_size,
                      reason="retry"),
            rec,
        )

    # -- housekeeping -------------------------------------------------------

    async def poll_states(self) -> None:
        for h in self.workers.values():
            with contextlib.suppress(Exception):
                await h.ch.send({"kind": P.STATE_REQ})

    async def calibrate(
        self, prompt_len: int = 96, gen: int = 6, seed: int = 99, batch: int = 0
    ) -> None:
        """Measure each device's real prefill/decode speed before serving.

        Heterogeneous devices differ by more than an order of magnitude, and the
        placement cost model is only as good as these numbers.

        Calibration drives `batch` concurrent sequences per worker rather than
        one, and that detail matters more than it looks. The NPU runs static
        shape buckets, so a decode step with one sequence costs the same as a
        decode step with eight -- measuring it with a single request made it
        look ~4x slower than it actually is under load (212 ms/step vs ~55), and
        the router then steered work away from a perfectly good accelerator.
        Measure a device in the regime it will actually run in.

        Each worker gets distinct random prompts so nothing lands in a shared
        prefix cache; worker state is wiped afterwards.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        futs = []
        for wid in self.workers:
            n = batch or max(1, self.workers[wid].max_batch)
            for b in range(n):
                ids = rng.integers(1000, 40000, size=prompt_len).tolist()
                req = Request(prompt_ids=[int(t) for t in ids], max_new_tokens=gen,
                              tag="calibration", req_id=f"cal-{wid}-{b}")
                futs.append(await self.submit_nowait(req, worker_id=wid))

        await asyncio.gather(*futs, return_exceptions=True)
        # Read the device-level timers the workers accumulated during the load.
        await self.poll_states()
        await asyncio.sleep(0.35)
        for wid, h in self.workers.items():
            self.log(
                f"{wid} [{h.engine}] prefill {h.t_prefill_per_token*1e3:.2f} ms/tok, "
                f"decode {h.t_decode_step*1e3:.1f} ms/step"
            )
        await self.reset()

    def snapshot(self) -> dict:
        return {
            "policy": self.cluster.policy,
            "link": {
                "bandwidth_mbps": self.cluster.link.bandwidth_mbps,
                "latency_ms": self.cluster.link.latency_ms,
            },
            "migrations": self.migrations,
            "migration_bytes": self.migration_bytes,
            "workers": [
                {
                    "worker_id": h.worker_id,
                    "device": h.device,
                    "engine": h.engine,
                    "queued": h.queued,
                    "running": h.running,
                    "kv_util": h.kv_util,
                    "cached_blocks": len(h.hashes),
                    "t_prefill_per_token_ms": round(h.t_prefill_per_token * 1e3, 3),
                    "t_decode_step_ms": round(h.t_decode_step * 1e3, 2),
                    "state": h.last_state,
                }
                for h in self.workers.values()
            ],
        }
