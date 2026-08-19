/**
 * Every control on the page, tested as behaviour rather than clicked.
 *
 *   node --test web/sim.test.mjs
 *
 * Each UI control maps to exactly one Sim method, so driving the methods is
 * driving the buttons. The DOM wiring in app.js does nothing but call these.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  Allocator, BLOCK, CLUSTERS, GEN_TOKENS, KV_PER_TOK, POLICIES, Sim,
  chainHashes, tokenize,
} from "./sim.mjs";

const SYS = "You are a careful assistant. Answer using only the provided context. ";
const A = SYS + "What is the capital of France?";
const B = SYS + "What is the largest planet in the solar system?";
const C = "Translate the following passage into plain English for a general audience.";

/* ------------------------------------------------------------- tokenizer -- */

test("tokenizer is deterministic and preserves shared prefixes", () => {
  assert.deepEqual(tokenize(A), tokenize(A));
  const a = tokenize(A), b = tokenize(B);
  const shared = tokenize(SYS).length;
  assert.ok(shared > BLOCK, "system prompt must span at least one block");
  assert.deepEqual(a.slice(0, shared), b.slice(0, shared));
  assert.notDeepEqual(a.slice(shared), b.slice(shared));
  assert.equal(tokenize("").length, 0);
  assert.equal(tokenize("   ").length, 0);
});

test("chain hashes are prefix-sensitive, not just content-sensitive", () => {
  const ids = tokenize(A);
  const h1 = chainHashes(ids);
  const h2 = chainHashes(tokenize(B));
  assert.equal(h1[0], h2[0], "same first block must hash the same");

  // identical block content, different history => different chain hash
  const tail = ids.slice(0, BLOCK);
  const x = chainHashes([...tail, ...tail]);
  const y = chainHashes([...tail.map(v => v ^ 1), ...tail]);
  assert.notEqual(x[1], y[1]);
});

/* ------------------------------------------------------------- allocator -- */

test("allocator reuses a cached prefix instead of reallocating", () => {
  const al = new Allocator(64);
  const ids = tokenize(A);
  const hs = chainHashes(ids);

  const first = al.allocate(ids, hs);
  assert.equal(first.cachedTokens, 0);
  al.publish(hs, first.blocks);
  const usedAfterFirst = al.used;

  const idsB = tokenize(B);
  const second = al.allocate(idsB, chainHashes(idsB));
  assert.ok(second.cachedTokens >= BLOCK, "second prompt should hit the cache");
  assert.ok(al.used <= usedAfterFirst + Math.ceil(idsB.length / BLOCK));

  al.release(first.blocks);
  al.release(second.blocks);
  assert.equal(al.used, 0, "everything released returns to free/cached");
});

test("allocator evicts least-recently-used blocks under pressure", () => {
  const al = new Allocator(4);                    // 64 tokens of capacity
  const ids = Array.from({ length: 64 }, (_, i) => i);
  const a = al.allocate(ids, chainHashes(ids));
  al.publish(chainHashes(ids), a.blocks);
  al.release(a.blocks);
  assert.equal(al.evictions, 0);

  const other = Array.from({ length: 64 }, (_, i) => 9000 + i);
  const b = al.allocate(other, chainHashes(other));
  assert.ok(b, "must succeed by evicting");
  assert.ok(al.evictions > 0, "eviction should have happened");
});

test("allocator refuses when the pool genuinely cannot fit the request", () => {
  const al = new Allocator(2);
  const ids = Array.from({ length: 500 }, (_, i) => i);
  assert.equal(al.allocate(ids, chainHashes(ids)), null);
  assert.equal(al.used, 0, "a failed allocation must leave no residue");
});

/* ----------------------------------------------------------- send button -- */

test("Send: a request runs to completion and reports sane timings", () => {
  const sim = new Sim();
  const r = sim.submit(A);
  assert.ok(r);
  sim.run(20);
  assert.equal(r.state, "done");
  assert.equal(r.out, GEN_TOKENS);
  assert.ok(r.tFirst > r.tSubmit, "TTFT must be positive");
  assert.ok(r.tDone > r.tFirst, "decode must take time");
  assert.equal(sim.metrics().served, 1);
});

test("Send twice: the second request hits the cache and skips prefill", () => {
  const sim = new Sim({ policy: "prefix_affinity" });
  const cold = sim.submit(A);
  sim.run(20);
  const warm = sim.submit(B);
  sim.run(20);

  assert.equal(cold.cached, 0);
  assert.ok(warm.cached >= BLOCK, `expected a cache hit, got ${warm.cached}`);
  assert.ok(warm.computed < cold.computed,
    `warm should compute less prefill (${warm.computed} vs ${cold.computed})`);
  assert.ok(sim.metrics().hitRate > 0);
});

test("a fresh prefix is cold again", () => {
  const sim = new Sim({ policy: "prefix_affinity" });
  sim.submit(A); sim.run(20);
  const other = sim.submit(C); sim.run(20);
  assert.equal(other.cached, 0);
});

/* ---------------------------------------------------------- burst button -- */

test("Burst: eight concurrent requests all complete", () => {
  const sim = new Sim();
  const reqs = [];
  for (let i = 0; i < 8; i++) reqs.push(sim.submit(`${A} variant ${i}`));
  sim.run(60);
  assert.equal(reqs.filter(r => r.state === "done").length, 8);
  assert.equal(sim.metrics().served, 8);
  assert.ok(!sim.busy, "nothing should still be in flight");
});

test("Burst spreads work across workers under a load-aware policy", () => {
  const sim = new Sim({ policy: "least_loaded" });
  for (let i = 0; i < 8; i++) sim.submit(`${A} variant ${i}`);
  sim.run(60);
  const used = new Set(sim.done.map(r => r.w.id));
  assert.ok(used.size > 1, "least_loaded should use more than one worker");
});

/* ---------------------------------------------------------- reset button -- */

test("Clear caches: wipes state, and the next request is cold again", () => {
  const sim = new Sim({ policy: "prefix_affinity" });
  sim.submit(A); sim.run(20);
  assert.ok(sim.metrics().served > 0);

  sim.reset();
  const m = sim.metrics();
  assert.equal(m.served, 0);
  assert.equal(m.migrations, 0);
  assert.equal(sim.t, 0);
  assert.equal(sim.log.length, 0);
  assert.ok(sim.workers.every(w => w.alloc.used === 0));

  const after = sim.submit(B);
  sim.run(20);
  assert.equal(after.cached, 0, "caches were cleared, so this must be cold");
});

/* -------------------------------------------------------- policy dropdown -- */

test("every policy places requests and completes them", () => {
  for (const policy of POLICIES) {
    const sim = new Sim({ policy });
    for (let i = 0; i < 6; i++) sim.submit(`${A} variant ${i}`);
    sim.run(60);
    assert.equal(sim.metrics().served, 6, `${policy} did not finish all requests`);
    assert.ok(sim.done.every(r => r.why), `${policy} should explain its placement`);
  }
});

test("round robin ignores the cache; affinity follows it", () => {
  const rr = new Sim({ policy: "round_robin" });
  const af = new Sim({ policy: "prefix_affinity" });
  for (const sim of [rr, af]) {
    for (let i = 0; i < 6; i++) { sim.submit(`${A} v${i}`); sim.run(6); }
    sim.run(40);
  }
  assert.equal(new Set(rr.done.map(r => r.w.id)).size, 2, "round robin must alternate");
  assert.equal(new Set(af.done.map(r => r.w.id)).size, 1, "affinity must concentrate");
  assert.ok(af.metrics().hitRate > rr.metrics().hitRate,
    "affinity should achieve a better hit rate than cache-blind routing");
});

test("setPolicy rejects an unknown policy rather than failing silently", () => {
  const sim = new Sim();
  assert.throws(() => sim.setPolicy("nonsense"), /unknown policy/);
});

/* ----------------------------------------------------- bandwidth slider -- */

test("a slow link suppresses migration; a fast one enables it", () => {
  const warm = (sim) => {                     // pin a long prefix onto one worker
    for (let i = 0; i < 3; i++) { sim.submit(`${A} v${i}`); sim.run(8); }
    sim.run(20);
  };
  const load = (sim) => {                     // then make that worker unattractive
    for (let i = 0; i < 10; i++) sim.submit(`${A} follow ${i}`);
    sim.run(60);
  };

  const slow = new Sim({ policy: "cache_aware", bandwidthMbps: 10 });
  warm(slow); load(slow);

  const fast = new Sim({ policy: "cache_aware", bandwidthMbps: 100000 });
  warm(fast); load(fast);

  assert.equal(slow.metrics().migrations, 0,
    "at 10 Mbps moving KV should never beat recomputing");
  assert.ok(fast.metrics().migrations > 0,
    "at 100 Gbps the scheduler should choose to move KV");
});

test("transfer cost is serialisation plus a latency floor", () => {
  const bytes = 16 * BLOCK * KV_PER_TOK;               // 9.4 MB, 16 blocks
  const sim = new Sim({ bandwidthMbps: 1000 });

  const at1G = sim.transferSeconds(bytes);
  sim.setBandwidth(10000);
  const at10G = sim.transferSeconds(bytes);

  // The serialisation term scales exactly with bandwidth...
  const ser1G = at1G - 0.002, ser10G = at10G - 0.002;
  assert.ok(Math.abs(ser1G / ser10G - 10) < 0.01, "serialisation should scale 10x");

  // ...but total time does not, because latency is a floor. This is why the
  // crossover stops moving once the link is fast enough: past that point you
  // are paying for a round trip, not for bytes.
  assert.ok(at1G / at10G < 10, "latency floor must damp the total speedup");
  assert.ok(at10G > 0.002, "never faster than one-way latency");

  sim.setBandwidth(1e9);
  assert.ok(sim.transferSeconds(bytes) < 0.0021, "at absurd bandwidth, latency is all that is left");
});

/* ------------------------------------------------------- cluster dropdown -- */

test("both clusters boot and serve", () => {
  for (const key of Object.keys(CLUSTERS)) {
    const sim = new Sim({ cluster: key });
    assert.equal(sim.workers.length, CLUSTERS[key].workers.length);
    sim.submit(A);
    sim.run(60);
    assert.equal(sim.metrics().served, 1, `${key} failed to serve`);
  }
});

test("switching cluster rebuilds the workers and clears state", () => {
  const sim = new Sim({ cluster: "t4" });
  sim.submit(A); sim.run(20);
  sim.setCluster("intel");
  assert.equal(sim.workers.length, 3);
  assert.equal(sim.metrics().served, 0);
  assert.ok(sim.workers.every(w => w.alloc.used === 0));
  sim.submit(A); sim.run(60);
  assert.equal(sim.metrics().served, 1);
});

/* ------------------------------------------------------------- integrity -- */

test("KV accounting is exact: 36 KiB per token", () => {
  assert.equal(KV_PER_TOK, 36864);
  const sim = new Sim();
  const bytes = 16 * BLOCK * KV_PER_TOK;            // 16 blocks
  assert.equal(bytes, 9437184);
  assert.ok(Math.abs(sim.transferSeconds(bytes) - (0.002 + bytes / 125e6)) < 1e-9);
});

test("a rejected request is reported, not silently dropped", () => {
  const sim = new Sim({ blocksPerWorker: 1 });      // 16 tokens of capacity
  const r = sim.submit(A);
  sim.run(20);
  assert.equal(r.state, "rejected");
  assert.match(r.why, /KV pool full/);
});

test("the simulator settles: no request is left in flight", () => {
  const sim = new Sim({ policy: "cache_aware", bandwidthMbps: 5000 });
  for (let i = 0; i < 12; i++) sim.submit(`${A} v${i}`);
  sim.run(120);
  assert.ok(!sim.busy);
  assert.equal(sim.metrics().served, 12);
  assert.ok(sim.done.every(r => r.tDone >= r.tFirst && r.tFirst >= r.tSubmit),
    "timestamps must be monotonic");
});

test("metrics are well-formed before anything has run", () => {
  const m = new Sim().metrics();
  assert.equal(m.served, 0);
  assert.equal(m.ttft, null);
  assert.equal(m.e2e, null);
  assert.equal(m.hitRate, 0);
  assert.equal(m.bytes, 0);
});
