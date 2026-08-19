# hetero-serve

A KV-cache-aware LLM serving scheduler that runs across three *different* accelerators
at once — an Intel Arc GPU, an NPU, and CPU cores — and decides, per request, whether
it is cheaper to **ship a KV cache across the network** or to **recompute it from scratch**.

Real GPT-2 weights. Real worker processes. Real TCP sockets with a real token-bucket
bandwidth shaper. Every number in this README was measured on the laptop it was
written on.

---

## The question I actually wanted to answer

I spent a long time working on modems, where the whole job is knowing when moving
bytes is cheaper than not moving them. Serving LLMs turns out to have the same shape
of problem hiding inside it.

When a request arrives that shares a long prefix with an earlier one — a system
prompt, a RAG document, an earlier turn of a conversation — its KV cache may already
exist, but on the *wrong* accelerator. You have three options:

1. run it on the accelerator that has the cache, even if that device is busy
2. run it elsewhere and recompute the prefix from scratch
3. run it elsewhere and drag the KV cache over the interconnect

Option 3 is a bandwidth-vs-compute trade, and I wanted to find where it flips.
For GPT-2 124M at fp16, one token of KV cache is **36 KiB across all layers**
(`2 × 12 layers × 768 dims × 2 bytes`), so a 512-token prefix is **18.9 MB**. At
10 Gbps that's 15 ms and obviously worth moving. At 50 Mbps it's 3.0 seconds and
obviously not. The interesting part is the middle, and what the wrong answer costs
you at p99.

---

## The hardware I had (and didn't have)

I do not own a multi-GPU box. I have a Dell 14 Plus with a Lunar Lake chip. But that
chip has **three independently targetable compute devices**, which turns out to be a
more interesting problem than a homogeneous GPU cluster, because they are not
interchangeable:

| device | hardware | what it's good at |
|---|---|---|
| `GPU` | Intel Arc 140V (8 GB, integrated) | fastest prefill; fastest decode at long context |
| `NPU` | Intel AI Boost | fastest decode at *short* context, then loses to shape padding |
| `CPU` | Core Ultra 7 256V, 8 threads | steady, no compile cost, no padding tax |

Each worker is a real OS process bound to one device. `scripts/probe_devices.py`
compiles real graphs to find out what each one supports, rather than trusting docs:

```
device   name                                        dynamic shapes  >4D tensors
CPU      Intel(R) Core(TM) Ultra 7 256V                         yes          yes
GPU      Intel(R) Arc(TM) 140V GPU (8GB) (iGPU)                 yes          yes
NPU      Intel(R) AI Boost                                       NO           NO
```

Those two `NO`s shaped the whole engine design (see [The NPU tax](#the-npu-tax)).

The same script then times GPT-2 124M on each of them — 128-token prefill, batch-4
decode, checked against the numpy reference:

| engine | prefill 128 tok | decode batch-4 | vs numpy | argmax matches numpy |
|---|---:|---:|---:|:--:|
| numpy (reference) | 707 ms | 428 ms | 1.0× | — |
| OpenVINO CPU | 69.8 ms | 31.6 ms | 10.1× | yes |
| **Arc 140V GPU** | **21.6 ms** | 30.1 ms | 32.8× | yes |
| **NPU** | 20.8 ms | **21.0 ms** | 34.1× | yes |

All three agree with the reference implementation on the predicted token. The
OpenVINO paths run fp16 internally, so logits differ in the third decimal — which is
why the correctness tests assert on argmax and on paged-vs-contiguous equivalence
rather than on bit equality.

---

## What it's made of

```
                    ┌──────────────────────────────────────┐
   requests ───────▶│  ROUTER  (control plane)             │
                    │                                      │
                    │  • global prefix directory           │
                    │      block-chain hash → {workers}    │
                    │  • cost model, priced in seconds     │
                    │      stay    = queue + recompute     │
                    │      migrate = queue + transfer      │
                    │                + recompute remainder │
                    │  • device speeds measured, not       │
                    │    guessed                           │
                    └───┬──────────────┬──────────────┬────┘
                        │              │              │      control plane (fast)
              ┌─────────▼───┐  ┌───────▼─────┐  ┌─────▼─────┐
              │  worker 0   │  │  worker 1   │  │ worker 2  │
              │  Arc GPU    │  │  NPU        │  │ CPU       │
              │             │  │             │  │           │
              │ paged KV    │  │ paged KV    │  │ paged KV  │
              │ continuous  │  │ continuous  │  │ continuous│
              │  batching   │  │  batching   │  │  batching │
              └──────┬──────┘  └──────┬──────┘  └─────┬─────┘
                     └────────────────┴───────────────┘
                        data plane: KV blocks over TCP,
                        through a token-bucket shaper
                        (bandwidth + latency + jitter)
```

**Paged KV cache** (`heteroserve/kv/blocks.py`). Fixed 16-token blocks, refcounted,
with vLLM-style automatic prefix sharing. Blocks are hashed by *chain* — block *i*'s
hash covers every token from position 0 through the end of block *i* — so a hash match
is a genuine shared-prefix match and not a coincidence. Unreferenced blocks stay
cached and are evicted LRU only under pressure. Because a prefix is a *set of blocks*
rather than a contiguous slab, it can be sliced, shared, and shipped.

**The engines** (`heteroserve/model/`). A numpy GPT-2 that serves as the correctness
oracle, and an OpenVINO graph built op-by-op that runs on any of the three devices.
The graph is hand-built rather than imported from ONNX because the KV cache has to be
an **input and an output** of the model: the scheduler owns the cache, so the engine
must accept whatever the block allocator gathered and hand the new K/V back every step.

**The network** (`heteroserve/net/`). Workers talk over real TCP. Outgoing KV transfers
pass through a token bucket sized to the configured bandwidth, then a propagation
delay. Concurrent transfers **contend for the same bucket**, which matters more than
I expected (below). When the benchmark says 2.3 s went into moving KV, the process
really did sit there for 2.3 s.

**The scheduler.** The router picks *where*; each worker decides *when*, with
prefill-priority chunked continuous batching and recompute-preemption when it runs out
of blocks.

---

## Results

48 requests over 6 shared 256-token prefixes drawn Zipf (so a few prefixes are hot),
Poisson arrivals at 6/s, 16 generated tokens each. Every configuration runs 3 times
and the records are pooled — 144 requests per row. `±` is the standard deviation of
throughput across the three repeats.

| policy | link | tok/s | ± | TTFT p50 | TTFT p99 | **E2E p50** | E2E p95 | cache hit | migrations | MB moved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| round robin | 50 Mbps | 57.0 | 0.9 | 133 ms | 478 ms | 2.87 s | 5.91 s | 62.9% | 0 | 0 |
| least loaded | 50 Mbps | 60.5 | 0.8 | 114 ms | **477 ms** | 2.20 s | 4.56 s | 62.9% | 0 | 0 |
| prefix affinity | 50 Mbps | 56.4 | 2.2 | 136 ms | 531 ms | 3.60 s | 5.74 s | 78.7% | 0 | 0 |
| cache aware | 50 Mbps | 56.8 | 0.8 | 138 ms | 508 ms | 2.71 s | 5.12 s | 64.8% | 0 | 0 |
| cache aware | 200 Mbps | 61.8 | 1.5 | **104 ms** | 1450 ms | 1.96 s | 4.56 s | 70.5% | 5 | 47 |
| cache aware | 1 Gbps | 60.1 | 0.6 | 106 ms | 785 ms | 1.95 s | **4.45 s** | 76.8% | 22 | 208 |
| cache aware | 10 Gbps | 59.8 | 1.5 | 113 ms | 674 ms | **1.94 s** | 4.77 s | **79.4%** | 27 | 255 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/ttft-by-policy-dark.png">
  <img alt="Latency by policy" src="results/figures/ttft-by-policy-light.png">
</picture>

**Throughput barely moves.** 56–62 tok/s across every policy, and the spread is only a
couple of standard deviations. At this arrival rate the cluster is saturated, and
routing decides *whose* latency is good, not how much total work gets done. I expected
a bigger throughput story and did not get one; saying so is more useful than picking
the one column where the gap looks impressive.

**The real win is end-to-end p50: 3.60 s → 1.94 s, a 1.85× improvement** over prefix
affinity. Cache-aware routing gets there by having it both ways — it reaches the
*highest* cache hit rate (79.4%, above affinity's 78.7%) while still spreading load,
because migration lets a device receive a prefix instead of having to be the device
that already owned it.

**No policy wins everything.** Plain least-loaded has the best TTFT tail (477 ms).
Cache-aware at 10 Gbps is worse at p99 (674 ms) because every migrated request pays
the transfer on its critical path. If your SLO is first-token latency, the boring
policy is a completely reasonable answer.

### Where the migration crossover actually is

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/bandwidth-crossover-dark.png">
  <img alt="Migration crossover vs bandwidth" src="results/figures/bandwidth-crossover-light.png">
</picture>

At **50 Mbps the scheduler refuses to migrate at all** — 0 out of 144 requests — which
is the correct call: moving a 9.4 MB prefix costs ~1.5 s against ~130 ms to recompute
it. As bandwidth rises it migrates more (5 → 22 → 27) and end-to-end latency improves.

The ugly point in the middle is the honest one. **At 200 Mbps, p99 TTFT spikes to
1.45 s** — worse than never migrating. That is the marginal regime: a handful of
transfers look individually affordable, get approved, and then collide on the link and
on the receiving worker. Being *nearly* fast enough to migrate is worse than being
clearly too slow.

### Migration is still 30× more expensive than the link model predicts

| link | payload | modelled | measured | effective rate |
|---|---:|---:|---:|---:|
| 200 Mbps | 9.44 MB | 380 ms | 1226 ms | 8 MB/s |
| 1 Gbps | 9.44 MB | 78 ms | 462 ms | 20 MB/s |
| 10 Gbps | 9.44 MB | 9.5 ms | 290 ms | 33 MB/s |

Effective throughput plateaus around 20–33 MB/s no matter how fast the link is, so
above a few hundred Mbps **migration is not bandwidth-bound at all**. Two causes, both
measured:

- **Head-of-line blocking** behind the engine's own step lock — see [bug #4](#4-i-held-the-engine-lock-through-a-multi-megabyte-memcpy).
  On an *idle* cluster the same transfer runs at 143 MB/s; the gap is contention, not
  the wire.
- **It sits on the critical path.** The request waits for its KV before it is even
  admitted. Overlapping the transfer with prefill of the uncached remainder is the
  obvious fix and is not implemented.

### Where a decode step really goes

`scripts/profile_decode.py`, batch 8, 288-token context, 84.9 MB of KV touched per step:

| device | gather | engine | write | total | gather share |
|---|---:|---:|---:|---:|---:|
| Arc GPU | 44.0 ms | 89.3 ms | 0.4 ms | 133.7 ms | 33% |
| NPU | 47.4 ms | 192.1 ms | 0.5 ms | 240.0 ms | 20% |
| CPU | 50.5 ms | 120.6 ms | 0.5 ms | 171.5 ms | 29% |
| numpy | 44.2 ms | 440.3 ms | 0.4 ms | 484.9 ms | 9% |

A third of a GPU decode step is **host-side numpy paging**, not accelerator time: my
attention gathers each sequence's blocks into contiguous tensors before calling the
engine, because OpenVINO cannot index the block pool directly. A fused paged-attention
kernel reads the block table in place and never pays this. It is the single biggest
inefficiency in the project, and it is a consequence of building paging *around* a
graph runtime rather than inside a kernel.

Note the NPU inverting here: fastest at 128-token context (27 ms) but slowest at 288
(192 ms), because 288 pads up to a 512 bucket and it computes ~1.8× the attention it
needs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/device-split-dark.png">
  <img alt="Work distribution across accelerators" src="results/figures/device-split-light.png">
</picture>

---

## What went wrong (the useful part)

Four things I got wrong. Every one was caught by measuring, not by reading the code —
which is the actual argument for building the benchmark harness before trusting the
scheduler.

### 1. I priced migrations against an idle link

The first cost model asked "how long does 18 MB take at this bandwidth?" and compared
that to recompute. Individually each migration looked cheap. Collectively they were
not: six of them landed on the same egress token bucket at once and queued. Tail
latency at 200 Mbps went to **3.3 s p99** — worse than the naive policy that never
migrates at all.

The fix is one term. A migration is now priced behind whatever is already in flight:

```python
mig_s = link.transfer_seconds(mig_bytes + self.inflight_migration_bytes)
```

That alone took p99 at 200 Mbps from 3.31 s to 1.82 s, and cut the number of
migrations it green-lit from 6 to 4. Congestion you cause yourself is still
congestion — obvious in hindsight, and exactly the mistake a bandwidth-naive
scheduler makes.

### 2. My "measured" device speeds were measuring the queue

The router prices placement using per-device prefill and decode costs, which it
measures at startup. I derived them from request latency: ms/token from
`first_token − admit`, ms/step from the gap between output tokens.

Both are contaminated. The first folds in queueing delay; the second folds in
prefill contention from other sequences on the same worker. So the "device speed"
moved with load instead of with the hardware, and calibration swung **4–6× between
runs of identical code** — NPU decode measured 55 ms/step one run and 212 ms/step the
next, which was enough to make the router abandon a perfectly good accelerator.

Now each worker times its own engine calls, split by phase, and the router divides
device-busy-time by work done. Queueing is modelled separately, where it belongs.

### 3. A rejected request hung forever

When a worker's KV pool is full it refuses admission, and the router is supposed to
try elsewhere. It updated its bookkeeping to point at the new worker — and never
actually sent the request. The caller waited on a future nothing would ever resolve.

Now it re-dispatches for real and remembers which workers already refused, so a
request cannot bounce between two full devices forever. Two regression tests cover it:
one asserts an impossible request raises promptly instead of hanging, the other that a
request refused by a full worker still completes somewhere else.

### 4. I held the engine lock through a multi-megabyte memcpy

Migrations were running at ~20 MB/s regardless of link speed. My first guess was the
serialisation path — the 6D transpose in `export_blocks` looked like an obvious
suspect. I benchmarked it before changing anything, and I was wrong: the whole
export → `tobytes` → import round trip is **5.8 ms for 9.4 MB**, and the transpose I
suspected is actually the *faster* of the two layouts I tried.

The real cause was a lock. Both `export_blocks` and `import_blocks` ran while holding
the worker's state lock — the same lock the engine holds for the duration of a decode
step. With GPT-2 a step is ~200 ms, and a migration waits out a step on the donor
*and* on the receiver, which is almost exactly the 424 ms observed. A controlled
idle-vs-loaded experiment confirmed it.

The fix is to hold the lock only for metadata (match, incref, reserve) and do the bulk
copy outside it. That is safe precisely because of how the allocator works: reserved
and pinned blocks have a non-zero refcount, so nothing else can claim or evict them,
and numpy drops the GIL for a copy that size. Result: 10 Gbps migrations 424 ms → 290 ms,
end-to-end p50 3.67 s → 1.94 s.

The lesson I'd keep: *profile before optimising* applies to your own confident
hypotheses too. I nearly rewrote a memory layout that was already fine.

---

## The NPU tax

The NPU rejects dynamic shapes *and* rejects tensors above 4D. Both constraints are
visible in the design:

- **Past KV is one 4D tensor per layer** (24 inputs, 24 outputs) instead of one tidy
  6D tensor. CPU and GPU would happily take the 6D version; the NPU would not, and
  one graph for all three devices is worth the verbosity.
- **Shapes are bucketed and compiled ahead of time.** Every bucket the scheduler could
  hit is compiled at startup (~15–20 s each, then cached to disk), because discovering
  a new bucket *mid-run* would inject a 20-second stall straight into a latency
  measurement.
- **Padding is real waste.** A prefill of 24 tokens still runs a 256-wide graph, and a
  decode step with 3 sequences costs the same as one with 8.

That last point caused bug #2 above: measuring NPU decode with a single request
measures the fully-padded batch, making it look ~4× slower than it is under load.

---

## Run it

```bash
pip install -r requirements.txt
python run_demo.py              # downloads GPT-2 (~550 MB) on first run
```

`run_demo.py` boots a cluster on whatever devices exist and walks through a cold
request, a warm one, the migrate-vs-recompute decision at two bandwidths, and a real
migration verified to produce byte-identical output.

```bash
# what can this machine actually do?
python scripts/probe_devices.py

# live dashboard: retune the interconnect while requests are in flight
python -m heteroserve.dashboard.server        # -> http://127.0.0.1:8000

# the full benchmark
python -m heteroserve.bench.sweep --repeats 3
python -m heteroserve.bench.plot

# tests
python -m pytest
```

Everything runs on CPU alone if OpenVINO is missing — the numpy engine is slower but
identical in behaviour, and the whole test suite uses it.

### The dashboard

Worker cards with live KV utilisation, a request feed showing what was cached versus
recomputed with the router's actual reasoning, a TTFT chart coloured by cache hit /
migration / cold, and a bandwidth slider. Drag the slider down and watch migrations
stop happening.

---

## Tests

18 tests, no mocks — the distributed ones spawn real worker processes and talk over
real sockets.

The load-bearing ones:

- **`test_paged_matches_contiguous`** — a sequence run through chunked prefill with its
  KV in blocks produces the same logits as one contiguous pass. Everything else
  (sharing, migration, preemption) rests on this.
- **`test_migrated_kv_produces_identical_output`** — a prefix moved across the network
  yields byte-identical generated tokens. Migration must be a performance decision,
  never a semantic one.
- **`test_bandwidth_actually_constrains_the_wire`** — a 10× slower link makes the same
  transfer measurably slower, proving the shaper affects the socket and not just the
  cost model.
- **`test_migrate_vs_recompute_flips_with_bandwidth`** — same prompt, same cluster
  state, 50 Mbps vs 10 Gbps, opposite decisions.
- **`test_oversized_prompt_fails_fast_instead_of_hanging`** — regression for bug #3.

---

## Honest limitations

- **Migration is synchronous.** A request waits for its KV to arrive before it is even
  admitted, so the transfer sits on the critical path. Overlapping it with the prefill
  of the uncached remainder is the obvious next thing, and would move the crossover
  meaningfully.
- **One machine.** The "network" is loopback plus a shaper. That models bandwidth and
  latency honestly but not packet loss, reordering, or a real NIC's behaviour under
  load.
- **GPT-2 124M.** Small enough that prefill is cheap relative to an 18 MB transfer,
  which pushes the crossover to higher bandwidth than a 7B model would. The KV/token
  ratio is what drives the trade, and it grows with model depth.
- **p99 on 144 samples** is still only the second-worst request. Treat p50 and p95 as
  solid and p99 as directional.
- **The control plane is unshaped** — only worker-to-worker KV transfers pay the link
  budget. That isolates the variable under study but is not what a real deployment
  would experience.

---

## Layout

```
heteroserve/
  config.py            models, KV geometry, link budgets, cluster shape
  metrics.py           TTFT / TPOT / E2E percentiles
  kv/blocks.py         paged allocator, chain hashing, eviction, migration I/O
  model/
    fetch.py           HuggingFace download + a dependency-free safetensors reader
    tokenizer.py       GPT-2 byte-level BPE, from scratch
    numpy_engine.py    reference transformer, paged KV
    ov_engine.py       OpenVINO graph for CPU / GPU / NPU
  net/
    shaper.py          token bucket + propagation delay
    transport.py       length-prefixed framing over TCP
  worker/worker.py     one device, one KV pool, continuous batching
  sched/router.py      prefix directory, cost model, placement, migration
  bench/               workload generation, sweep harness, charts
  dashboard/           live view
```
