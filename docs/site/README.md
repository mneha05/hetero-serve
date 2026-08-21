# Portfolio site — hetero-serve card

`index.html` here is **the live nehamahesh.netlify.app page with the hetero-serve card
added**, ready to deploy as-is.

## Why the whole file and not a patch

That site is not built from any repository — its markup matches nothing in the account
and a code search across every repo (public and private, root and `public/`, `dist/`,
`src/`, `site/`, `app/`) turns up nothing. It was deployed by hand, which means **the
deployed file is the source**. So the safe move is to hand back a complete file rather
than a snippet to splice into something that may not exist anywhere else.

## Deploy it

1. Download [`index.html`](index.html) (raw, then save).
2. Put it alone in a folder.
3. Drag that folder onto the **Deploys** tab of the Netlify project.

That is the same path the current site took, so nothing about the setup changes.

**Better, if you want this to stop being manual:** create a repo with `index.html` in it,
then in Netlify go to *Site configuration → Build & deploy → Link to a Git repository*.
After that every push deploys, and this file can be edited in version control like
everything else.

## What changed

- **hetero-serve added as `/ 01`**, and the six existing cards renumbered to `/ 02`–`/ 07`.
  Nothing else in the page was touched — same classes (`proj reveal sheen metric knum kind
  stk howbox howcap`), same voice, same structure.
- The card carries the animated hero and the architecture diagram straight from
  `raw.githubusercontent.com`, so there are no assets to copy and they update whenever the
  repo does.

## One thing to decide

The existing **`/ 02` Inference Engine — CUDA · from scratch** card overlaps heavily with
this one: both describe fused attention and a paged KV cache in hand-written CUDA. Sitting
next to each other they invite the question of whether they are the same project. Worth
either merging them, cutting one, or making the distinction explicit — hetero-serve is a
*multi-device scheduler* whose kernels exist because it profiled itself, which is a
different claim from *an inference engine written from scratch*.

## Regenerating the images

```bash
python scripts/make_gif.py        # -> docs/hero.gif        (45 KB, 6.4 s loop)
python scripts/make_diagram.py    # -> docs/architecture.png
```

`docs/site-card.html` holds just the `<article>` on its own, if you would rather paste it
into a source file you keep somewhere else.
