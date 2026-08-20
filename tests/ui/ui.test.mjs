/**
 * Drive the real page in a real DOM and click every control.
 *
 * Loads the built artifact bundle (a classic <script>, so jsdom will execute it
 * -- jsdom does not run <script type="module">), then dispatches genuine events
 * and asserts the page reacted. This tests the wiring between the buttons and
 * the scheduler, which the sim tests deliberately do not cover.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

// jsdom is a dev-only dependency: skip rather than fail when it is absent.
let JSDOM = null;
try { ({ JSDOM } = await import("jsdom")); } catch { /* not installed */ }

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUNDLE = process.env.BUNDLE
  || path.join(HERE, "..", "..", "build", "artifact.html");
const SKIP = JSDOM
  ? false
  : { skip: "jsdom not installed — run `npm --prefix tests/ui ci` first" };

function boot() {
  const html = readFileSync(BUNDLE, "utf-8");
  const dom = new JSDOM(`<!doctype html><html><body>${html}</body></html>`, {
    runScripts: "dangerously",
    pretendToBeVisual: true,   // gives us requestAnimationFrame
  });
  const { window } = dom;
  const $ = (id) => window.document.getElementById(id);
  // let the boot code and a few animation frames run
  const settle = () => new Promise(r => window.setTimeout(r, 40));
  return { window, $, settle, doc: window.document, close: () => window.close() };
}

const click = (el) => el.dispatchEvent(new el.ownerDocument.defaultView.MouseEvent(
  "click", { bubbles: true }));
const change = (el, v) => {
  el.value = v;
  el.dispatchEvent(new el.ownerDocument.defaultView.Event("change", { bubbles: true }));
};
const input = (el, v) => {
  el.value = v;
  el.dispatchEvent(new el.ownerDocument.defaultView.Event("input", { bubbles: true }));
};

test("page boots without throwing and renders the workers", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  assert.equal($("workers").children.length, 2, "two worker cards for the default cluster");
  assert.match($("cNote").textContent, /Tesla T4/);
  assert.equal($("hServed").textContent, "0");
  assert.match($("stream").textContent, /Send a prompt/);
  assert.match($("tokN").textContent, /^\d+ tok$/, "token count filled on boot");
});

test("Send: clicking actually submits and the stream updates", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  click($("send"));
  await settle();
  assert.doesNotMatch($("stream").textContent, /Send a prompt —/, "placeholder replaced");
  assert.match($("stream").textContent, /Tesla T4/, "request row names its worker");
});

test("Send twice: the UI reports a cache hit", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, window, settle } = ctx;
  await settle();
  click($("send"));
  for (let i = 0; i < 40; i++) { await settle(); }        // let it finish
  input($("prompt"),
    "You are a careful assistant. Answer using only the provided context. What is the largest planet?");
  click($("send"));
  await settle();
  assert.match($("stream").textContent, /cache hit/, "second request should show a cache hit");
});

test("Burst: fires eight requests", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  click($("burst"));
  for (let i = 0; i < 20; i++) await settle();
  const rows = $("stream").querySelectorAll(".r").length;
  assert.ok(rows >= 8, `expected at least 8 rows, saw ${rows}`);
});

test("Clear: resets the stream and the counters", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  click($("send"));
  for (let i = 0; i < 30; i++) await settle();
  assert.notEqual($("hServed").textContent, "0");
  click($("reset"));
  await settle();
  assert.equal($("hServed").textContent, "0");
  assert.match($("stream").textContent, /Caches cleared/);
});

test("Policy dropdown: every option is accepted without error", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle, window } = ctx;
  await settle();
  const errors = [];
  window.addEventListener("error", e => errors.push(e.message));
  for (const opt of [...$("policy").options]) {
    change($("policy"), opt.value);
    click($("send"));
    await settle();
  }
  for (let i = 0; i < 20; i++) await settle();
  assert.deepEqual(errors, [], "no runtime errors while switching policy");
  assert.ok(+$("hServed").textContent > 0);
});

test("Interconnect slider: updates the label and the wire readout", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  input($("bw"), "0");
  await settle();
  const slow = $("bwV").textContent;
  assert.match(slow, /Mbps/);
  input($("bw"), "100");
  await settle();
  const fast = $("bwV").textContent;
  assert.match(fast, /Gbps/);
  assert.notEqual(slow, fast);
  assert.match($("wireT").textContent, /Gbps|Mbps/);
});

test("Cluster dropdown: switching rebuilds the worker cards", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  assert.equal($("workers").children.length, 2);
  change($("cluster"), "intel");
  await settle();
  assert.equal($("workers").children.length, 3, "Intel cluster has three workers");
  assert.match($("cNote").textContent, /Arc/);
  assert.match($("stream").textContent, /Cluster switched/);
  click($("send"));
  await settle();
  assert.match($("stream").textContent, /Arc 140V|AI Boost|Ultra 7/);
});

test("Sim speed slider updates its label", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  input($("speed"), "17");
  await settle();
  assert.equal($("spV").textContent, "17×");
});

test("Preset buttons: all four submit their prompt", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  const steps = $("steps").querySelectorAll("button");
  assert.equal(steps.length, 4, "four presets");
  for (const b of steps) {
    click(b);
    await settle();
  }
  for (let i = 0; i < 40; i++) await settle();
  assert.ok(+$("hServed").textContent >= 3, "presets should have served requests");
});

test("the results table rendered its rows", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, settle } = ctx;
  await settle();
  const rows = $("kRows").querySelectorAll("tr");
  assert.equal(rows.length, 5);
  assert.match($("kRows").textContent, /55\.4%/, "the headline number is present");
});

test("no uncaught errors during a full drive of the page", SKIP, async (t) => {
  const ctx = boot();
  t.after(() => ctx.close());
  const { $, window, settle } = ctx;
  const errors = [];
  window.addEventListener("error", e => errors.push(e.message));
  window.addEventListener("unhandledrejection", e => errors.push(String(e.reason)));
  await settle();

  click($("send"));
  click($("burst"));
  input($("bw"), "20");
  change($("policy"), "round_robin");
  for (let i = 0; i < 25; i++) await settle();
  change($("cluster"), "intel");
  change($("policy"), "cache_aware");
  input($("bw"), "95");
  click($("burst"));
  for (let i = 0; i < 40; i++) await settle();
  click($("reset"));
  await settle();

  assert.deepEqual(errors, [], `page threw: ${errors.join(" | ")}`);
});
