"""Device-to-device KV transport, so a migration stops bouncing through host RAM.

The default path serialises a block payload to host memory and pushes it over a
shaped TCP socket. That is the right thing when you are *studying* the
migrate-vs-recompute crossover -- you can dial the link to 50 Mbps and watch the
decision flip, which no real interconnect lets you do. It is the wrong thing on
a multi-GPU node, where it adds two PCIe crossings and a memcpy to a transfer the
hardware could have done directly.

So there are two transports, and they are not rivals:

    shaped TCP    GPU -> host -> socket -> host -> GPU.  Bandwidth is a knob.
                  What the benchmark sweeps.
    dist (NCCL)   GPU -> GPU, straight across NVLink or PCIe peer-to-peer.
                  Bandwidth is whatever the hardware gives you, and the cost
                  model reads it from a probe rather than a config file.

Coordination stays on the existing TCP peer connection: the sender ships a small
metadata frame ("N blocks, this shape, this dtype, from rank R"), the receiver
posts a matching `recv`, and only the bulk payload crosses the fast path. That
keeps the protocol change small and means the two transports differ in exactly
one place -- who carries the bytes.

`gloo` is used as the backend when CUDA is absent, which is what makes the whole
path testable on a machine with no GPU: the rendezvous, the metadata handshake,
the shapes and the byte-exactness are all identical, and only the wire underneath
changes. Swapping to NCCL is one string.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


def torch_distributed_available() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available()


def preferred_backend(device: str | None = None) -> str:
    """NCCL when there is a CUDA device to use it on, gloo otherwise."""
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        return "gloo"
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        if getattr(dist, "is_nccl_available", lambda: False)():
            return "nccl"
    return "gloo"


@dataclass
class LinkStats:
    sends: int = 0
    recvs: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    send_seconds: float = 0.0
    recv_seconds: float = 0.0

    @property
    def achieved_bytes_per_sec(self) -> float:
        if self.send_seconds <= 0:
            return 0.0
        return self.bytes_sent / self.send_seconds

    def as_dict(self) -> dict:
        return {
            "sends": self.sends,
            "recvs": self.recvs,
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "achieved_mbps": round(self.achieved_bytes_per_sec * 8 / 1e6, 1),
        }


class DistLink:
    """A torch.distributed process group used only for point-to-point KV moves.

    Deliberately not used for anything collective. Migration is inherently
    pairwise -- one worker has a prefix another one wants -- and a collective
    would force every worker to participate in a transfer that concerns two of
    them.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        init_method: str,
        device: str = "cpu",
        backend: str = "auto",
        timeout_s: float = 30.0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.init_method = init_method
        self.device = device
        self.backend = preferred_backend(device) if backend == "auto" else backend
        self.timeout_s = timeout_s
        self.ready = False
        self.error: Optional[str] = None
        self.stats = LinkStats()
        self._torch = None
        self._dist = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> bool:
        """Join the process group. Returns False (with `error` set) rather than
        raising: a failed rendezvous must degrade to the TCP path, not take the
        worker down."""
        if self.world_size < 2:
            self.error = "world_size < 2: nothing to migrate to"
            return False
        try:
            import datetime

            import torch
            import torch.distributed as dist
        except ImportError as exc:
            self.error = f"torch.distributed unavailable: {exc}"
            return False

        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.backend,
                    init_method=self.init_method,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=datetime.timedelta(seconds=self.timeout_s),
                )
            if self.backend == "nccl":
                torch.cuda.set_device(self.device)
            self._torch = torch
            self._dist = dist
            self.ready = True
            return True
        except Exception as exc:  # noqa: BLE001 - degrade, do not crash
            self.error = f"{type(exc).__name__}: {str(exc).splitlines()[-1][:200]}"
            return False

    def shutdown(self) -> None:
        if not self.ready or self._dist is None:
            return
        try:
            if self._dist.is_initialized():
                self._dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        self.ready = False

    # -- point to point -----------------------------------------------------

    def send_tensor(self, tensor, dst_rank: int) -> float:
        """Blocking send. Call from a worker thread, never the event loop."""
        if not self.ready:
            raise RuntimeError(f"link not ready: {self.error}")
        t = tensor.contiguous()
        if self.backend == "nccl":
            t = t.to(self.device, non_blocking=False)
        t0 = time.perf_counter()
        self._dist.send(t, dst=dst_rank)
        if self.backend == "nccl":
            self._torch.cuda.synchronize(self.device)   # send is async on CUDA
        dt = time.perf_counter() - t0
        self.stats.sends += 1
        self.stats.bytes_sent += t.numel() * t.element_size()
        self.stats.send_seconds += dt
        return dt

    def recv_tensor(self, shape, dtype, src_rank: int):
        """Blocking receive into a freshly allocated tensor of the given shape."""
        if not self.ready:
            raise RuntimeError(f"link not ready: {self.error}")
        torch = self._torch
        dev = self.device if self.backend == "nccl" else "cpu"
        buf = torch.empty(tuple(shape), dtype=_as_dtype(torch, dtype), device=dev)
        t0 = time.perf_counter()
        self._dist.recv(buf, src=src_rank)
        if self.backend == "nccl":
            torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        self.stats.recvs += 1
        self.stats.bytes_recv += buf.numel() * buf.element_size()
        self.stats.recv_seconds += dt
        return buf

    # -- calibration --------------------------------------------------------

    def probe_bandwidth(self, peer_rank: int, nbytes: int = 8 << 20,
                        iters: int = 3) -> float:
        """Measure achieved bytes/sec to a peer.

        The whole point of the fast path is that its bandwidth is a property of
        the hardware rather than a number in a config file, so the scheduler
        should measure it and price migrations against the result. Sender side
        only; the peer must be running `probe_bandwidth_responder`.
        """
        if not self.ready:
            return 0.0
        torch = self._torch
        dev = self.device if self.backend == "nccl" else "cpu"
        payload = torch.empty(nbytes // 2, dtype=torch.float16, device=dev)
        best = 0.0
        for _ in range(iters):
            dt = self.send_tensor(payload, peer_rank)
            if dt > 0:
                best = max(best, nbytes / dt)
        return best

    def probe_bandwidth_responder(self, peer_rank: int, nbytes: int = 8 << 20,
                                  iters: int = 3) -> None:
        if not self.ready:
            return
        torch = self._torch
        for _ in range(iters):
            self.recv_tensor((nbytes // 2,), torch.float16, peer_rank)

    def info(self) -> dict:
        return {
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "ready": self.ready,
            "error": self.error,
            **self.stats.as_dict(),
        }


def _as_dtype(torch, dtype):
    if isinstance(dtype, str):
        return getattr(torch, dtype.replace("torch.", ""))
    return dtype


def default_init_method(port: int | None = None) -> str:
    port = port or int(os.environ.get("HETEROSERVE_DIST_PORT", "29677"))
    host = os.environ.get("HETEROSERVE_DIST_HOST", "127.0.0.1")
    return f"tcp://{host}:{port}"
