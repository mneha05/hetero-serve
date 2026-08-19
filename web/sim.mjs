/**
 * The scheduler, with no DOM in sight.
 *
 * A faithful port of the Python: 16-token paged blocks, chain-hashed prefix
 * matching, refcounts and LRU eviction, chunked prefill, continuous batching,
 * four routing policies, and the contention-aware migrate-vs-recompute cost
 * model priced against a token-bucket link.
 *
 * Kept free of the DOM on purpose -- `sim.test.mjs` drives every control path
 * through this module under Node, so "does the button work" is a test rather
 * than a click.
 *
 * What is NOT here is GPT-2. A 124M-parameter model will not run in a browser
 * tab, so generated text is a stand-in and step costs come from timings
 * measured on real hardware.
 */

export const BLOCK = 16;              // tokens per block, as in the Python
export const KV_PER_TOK = 36864;      // GPT-2 124M, fp16, all layers
export const LATENCY_S = 0.002;       // measured one-way link latency
export const GEN_TOKENS = 14;
export const PREFILL_CHUNK = 256;

export const CLUSTERS = {
  t4: {
    label: "2 × Tesla T4",
    note: "CUDA · v3 fused kernel",
    workers: [
      { id: "cuda:0", name: "Tesla T4", sub: "v3 kernel", pre: 0.59, dec: 52 },
      { id: "cuda:1", name: "Tesla T4", sub: "v3 kernel", pre: 0.59, dec: 52 },
    ],
  },
  intel: {
    label: "Arc + NPU + CPU",
    note: "heterogeneous · OpenVINO",
    workers: [
      { id: "GPU", name: "Arc 140V", sub: "iGPU", pre: 0.38, dec: 134 },
      { id: "NPU", name: "AI Boost", sub: "static shapes", pre: 0.62, dec: 240 },
      { id: "CPU", name: "Ultra 7", sub: "8 threads", pre: 1.38, dec: 172 },
    ],
  },
};

export const POLICIES = ["cache_aware", "prefix_affinity", "least_loaded", "round_robin"];

/* ---------------------------------------------------------------- tokens -- */

/** Deterministic, and crucially prefix-preserving: two prompts that start with
 *  the same words produce the same leading token ids, so prefix caching behaves
 *  the way it would with a real BPE tokenizer. */
export function tokenize(text) {
  const words = String(text).trim().toLowerCase().match(/[a-z0-9']+|[^\sa-z0-9]/g) || [];
  const ids = [];
  for (const w of words) {
    let h = 2166136261;
    for (let i = 0; i < w.length; i++) { h ^= w.charCodeAt(i); h = Math.imul(h, 16777619); }
    ids.push((h >>> 0) % 50000);
    if (w.length > 6) ids.push(((h >>> 7) ^ 0x9e37) % 50000);   // long words split
  }
  return ids;
}

/** Chain hash per full block: block i's hash covers tokens 0..end-of-i, so a
 *  match is a genuine shared prefix and not a coincidental block collision. */
export function chainHashes(ids) {
  const out = [];
  let prev = 0x811c9dc5;
  for (let b = 0; b + BLOCK <= ids.length; b += BLOCK) {
    let h = prev;
    for (let i = b; i < b + BLOCK; i++) { h ^= ids[i]; h = Math.imul(h, 16777619); }
    prev = h >>> 0;
    out.push(prev);
  }
  return out;
}

/* ------------------------------------------------------------- allocator -- */

export class Allocator {
  constructor(nBlocks) {
    this.n = nBlocks;
    this.free = [];
    for (let i = nBlocks - 1; i >= 0; i--) this.free.push(i);
    this.ref = new Int32Array(nBlocks);
    this.byHash = new Map();      // hash -> block
    this.hashOf = new Map();      // block -> hash
    this.evictable = new Map();   // hash -> true, insertion order == LRU
    this.recent = new Set();      // blocks touched since the last render
    this.evictions = 0;
  }

  get used() { return this.n - this.free.length - this.evictable.size; }
  get utilisation() { return this.used / this.n; }

  take() {
    if (this.free.length) return this.free.pop();
    if (this.evictable.size) {                       // evict least recently used
      const h = this.evictable.keys().next().value;
      this.evictable.delete(h);
      const b = this.byHash.get(h);
      this.byHash.delete(h);
      this.hashOf.delete(b);
      this.evictions++;
      return b;
    }
    return -1;                                       // pool exhausted
  }

  incref(b) {
    if (this.ref[b] === 0) {
      const h = this.hashOf.get(b);
      if (h !== undefined) this.evictable.delete(h);
    }
    this.ref[b]++;
  }

  decref(b) {
    if (--this.ref[b] > 0) return;
    this.ref[b] = 0;
    const h = this.hashOf.get(b);
    // Keep the contents cached: it may be a useful prefix for someone later.
    if (h !== undefined && this.byHash.get(h) === b) this.evictable.set(h, true);
    else this.free.push(b);
  }

  /** Longest cached prefix, in blocks. Stops at the first miss. */
  matchPrefix(hashes) {
    let n = 0;
    for (const h of hashes) { if (!this.byHash.has(h)) break; n++; }
    return n;
  }

  allocate(ids, hashes) {
    const need = Math.ceil(ids.length / BLOCK);
    const hit = Math.min(this.matchPrefix(hashes), need);
    const blocks = [];
    for (let i = 0; i < hit; i++) {
      const b = this.byHash.get(hashes[i]);
      this.incref(b);
      blocks.push(b);
    }
    while (blocks.length < need) {
      const b = this.take();
      if (b < 0) { blocks.forEach(x => this.decref(x)); return null; }
      this.ref[b] = 1;
      blocks.push(b);
      this.recent.add(b);
    }
    return { blocks, cachedTokens: hit * BLOCK };
  }

  /** Publish completed blocks so other sequences can share them. */
  publish(hashes, blocks) {
    for (let i = 0; i < hashes.length && i < blocks.length; i++) {
      if (this.byHash.has(hashes[i])) continue;
      this.byHash.set(hashes[i], blocks[i]);
      this.hashOf.set(blocks[i], hashes[i]);
    }
  }

  release(blocks) { blocks.forEach(b => this.decref(b)); }

  /** Migrated blocks arrive already populated and unowned: cached, evictable. */
  adopt(hashes, upto) {
    let n = 0;
    for (let i = 0; i < upto && i < hashes.length; i++) {
      if (this.byHash.has(hashes[i])) continue;
      const b = this.take();
      if (b < 0) break;
      this.byHash.set(hashes[i], b);
      this.hashOf.set(b, hashes[i]);
      this.evictable.set(hashes[i], true);
      this.recent.add(b);
      n++;
    }
    return n;
  }

  /** 'used' | 'cached' | 'free', for rendering. */
  stateOf(b) {
    if (this.ref[b] > 0) return "used";
    return this.hashOf.has(b) ? "cached" : "free";
  }
}

/* ---------------------------------------------------------------- worker -- */

export class Worker {
  constructor(spec, nBlocks) {
    Object.assign(this, spec);
    this.alloc = new Allocator(nBlocks);
    this.queue = [];
    this.running = [];
    this.busyUntil = 0;
    this.phase = "idle";
    this.pendingPrefill = 0;
    this.generated = 0;
    this.saved = 0;
    this.migIn = 0;
  }
  /** Seconds of work already committed here -- the queue term of the cost model. */
  get load() {
    return this.pendingPrefill * this.pre / 1000 + this.running.length * this.dec / 1000;
  }
}

/* ------------------------------------------------------------------- sim -- */

export class Sim {
  constructor({ cluster = "t4", policy = "cache_aware", bandwidthMbps = 1000,
                blocksPerWorker = 96 } = {}) {
    this.blocksPerWorker = blocksPerWorker;
    this.policy = policy;
    this.bandwidthMbps = bandwidthMbps;
    this.setCluster(cluster);
  }

  setCluster(key) {
    this.clusterKey = key;
    this.workers = CLUSTERS[key].workers.map(s => new Worker(s, this.blocksPerWorker));
    this.reset(true);
  }

  reset(keepWorkers = false) {
    if (!keepWorkers) {
      this.workers = CLUSTERS[this.clusterKey].workers.map(
        s => new Worker(s, this.blocksPerWorker));
    }
    this.t = 0;
    this.rr = 0;
    this.seq = 0;
    this.done = [];
    this.log = [];
    this.inFlight = [];
    this.wire = { busyUntil: 0, bytes: 0, count: 0, from: 0, until: 0 };
  }

  setPolicy(p) {
    if (!POLICIES.includes(p)) throw new Error(`unknown policy: ${p}`);
    this.policy = p;
  }
  setBandwidth(mbps) { this.bandwidthMbps = mbps; }

  transferSeconds(bytes) { return LATENCY_S + bytes / (this.bandwidthMbps * 1e6 / 8); }

  /* -- placement: the cost model ------------------------------------------ */

  place(ids, hashes) {
    const W = this.workers;

    if (this.policy === "round_robin") {
      const w = W[this.rr++ % W.length];
      return { w, donor: null, hitTok: 0, why: "round robin · cache ignored" };
    }

    const hits = W.map(w => w.alloc.matchPrefix(hashes));

    if (this.policy === "least_loaded") {
      let b = 0;
      W.forEach((w, i) => { if (w.load < W[b].load) b = i; });
      return { w: W[b], donor: null, hitTok: hits[b] * BLOCK, why: "least loaded" };
    }

    if (this.policy === "prefix_affinity") {
      let b = 0;
      W.forEach((w, i) => {
        if (hits[i] > hits[b] || (hits[i] === hits[b] && w.load < W[b].load)) b = i;
      });
      return { w: W[b], donor: null, hitTok: hits[b] * BLOCK,
               why: `affinity · ${hits[b]} blk resident` };
    }

    // cache_aware: price staying against migrating, both in seconds
    let best = null;
    W.forEach((w, i) => {
      const localTok = hits[i] * BLOCK;
      const stay = w.load + Math.max(0, ids.length - localTok) * w.pre / 1000;
      let cand = { w, donor: null, hitTok: localTok, cost: stay, bytes: 0,
                   why: `local hit ${localTok} tok` };

      let d = -1;
      W.forEach((_, j) => { if (j !== i && (d < 0 || hits[j] > hits[d])) d = j; });
      if (d >= 0 && hits[d] > hits[i]) {
        const deltaBlocks = hits[d] - hits[i];
        const bytes = deltaBlocks * BLOCK * KV_PER_TOK;
        // Queue behind whatever is already on the wire. Pricing against an idle
        // link is the bug that wrecked p99 in the Python version.
        const inflight = Math.max(0, this.wire.busyUntil - this.t);
        const mig = this.transferSeconds(bytes) + inflight;
        const donorTok = hits[d] * BLOCK;
        const alt = w.load + mig + Math.max(0, ids.length - donorTok) * w.pre / 1000;
        if (alt < cand.cost) {
          cand = { w, donor: W[d], hitTok: donorTok, cost: alt, bytes,
                   why: `migrate ${deltaBlocks} blk · ${(bytes / 1e6).toFixed(1)} MB · ~${Math.round(mig * 1000)} ms` };
        }
      }
      if (!best || cand.cost < best.cost) best = cand;
    });
    return best;
  }

  /* -- submission --------------------------------------------------------- */

  submit(text) {
    const ids = tokenize(text);
    if (!ids.length) return null;
    const hashes = chainHashes(ids);
    const p = this.place(ids, hashes);

    const req = {
      id: ++this.seq, text, ids, hashes, w: p.w, why: p.why,
      tSubmit: this.t, tFirst: 0, tDone: 0,
      migrated: false, bytes: 0, cached: 0, computed: 0,
      nKv: 0, out: 0, state: "queued",
    };

    const admit = () => {
      const a = req.w.alloc.allocate(ids, hashes);
      if (!a) { req.state = "rejected"; req.why += " · KV pool full"; return; }
      req.blocks = a.blocks;
      req.cached = Math.min(a.cachedTokens, ids.length - 1);
      req.nKv = req.cached;
      req.w.pendingPrefill += ids.length - req.cached;
      req.w.saved += req.cached;
      req.w.queue.push(req);
      req.state = "prefill";
    };

    if (p.donor && p.bytes) {
      // The transfer sits on the request's critical path, as in the Python.
      const start = Math.max(this.t, this.wire.busyUntil);
      const dur = this.transferSeconds(p.bytes);
      this.wire.busyUntil = start + dur;
      this.wire.from = start;
      this.wire.until = start + dur;
      this.wire.bytes += p.bytes;
      this.wire.count++;
      req.migrated = true;
      req.bytes = p.bytes;
      req.state = "moving";
      this.inFlight.push({
        at: this.wire.busyUntil,
        run: () => {
          p.w.alloc.adopt(hashes, Math.ceil(p.hitTok / BLOCK));
          p.w.migIn++;
          admit();
        },
      });
    } else {
      admit();
    }

    this.log.unshift(req);
    if (this.log.length > 40) this.log.pop();
    return req;
  }

  /* -- one step ----------------------------------------------------------- */

  tick(dt) {
    this.t += dt;

    for (let i = this.inFlight.length - 1; i >= 0; i--) {
      if (this.t >= this.inFlight[i].at) { this.inFlight[i].run(); this.inFlight.splice(i, 1); }
    }

    for (const w of this.workers) {
      if (this.t < w.busyUntil) continue;
      w.phase = "idle";

      // prefill first, chunked, so one long prompt cannot stall every decode
      const pend = w.queue[0];
      if (pend) {
        const chunk = Math.min(PREFILL_CHUNK, pend.ids.length - pend.nKv);
        if (chunk > 0) {
          w.busyUntil = this.t + chunk * w.pre / 1000;
          w.phase = "prefill";
          pend.nKv += chunk;
          pend.computed += chunk;
          w.pendingPrefill = Math.max(0, w.pendingPrefill - chunk);
          if (pend.nKv >= pend.ids.length) {
            w.alloc.publish(pend.hashes, pend.blocks);
            pend.tFirst = w.busyUntil;
            pend.out = 1;
            pend.state = "decoding";
            w.queue.shift();
            w.running.push(pend);
          }
          continue;
        }
        w.queue.shift();
      }

      // otherwise decode the whole running set as one batch
      if (w.running.length) {
        w.busyUntil = this.t + w.dec / 1000;
        w.phase = "decode";
        for (let i = w.running.length - 1; i >= 0; i--) {
          const r = w.running[i];
          r.out++;
          w.generated++;
          if (r.out >= GEN_TOKENS) {
            r.tDone = w.busyUntil;
            r.state = "done";
            w.alloc.publish(r.hashes, r.blocks);
            w.alloc.release(r.blocks);
            w.running.splice(i, 1);
            this.done.push(r);
          }
        }
      }
    }
  }

  /** Advance by `seconds` in small steps -- used by tests to run to completion. */
  run(seconds, step = 0.005) {
    const end = this.t + seconds;
    while (this.t < end) this.tick(step);
  }

  get busy() {
    return this.workers.some(w => w.queue.length || w.running.length) || this.inFlight.length > 0;
  }

  metrics() {
    const d = this.done;
    const p50 = (a) => {
      if (!a.length) return null;
      const s = a.slice().sort((x, y) => x - y);
      return s[Math.floor(s.length * 0.5)];
    };
    const prompt = d.reduce((s, r) => s + r.ids.length, 0);
    const cached = d.reduce((s, r) => s + r.cached, 0);
    return {
      served: d.length,
      ttft: p50(d.map(r => r.tFirst - r.tSubmit)),
      e2e: p50(d.map(r => r.tDone - r.tSubmit)),
      hitRate: prompt ? cached / prompt : 0,
      saved: cached,
      migrations: this.wire.count,
      bytes: this.wire.bytes,
      wireActive: this.t < this.wire.until,
      wireProgress: this.wire.until > this.wire.from
        ? Math.max(0, Math.min(1, (this.t - this.wire.from) / (this.wire.until - this.wire.from)))
        : 0,
    };
  }
}
