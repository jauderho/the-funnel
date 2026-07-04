# AESTHETIC_CONTRACT.md — "Cyberdeck"

> **Purpose.** This is a binding design contract for the **Cyberdeck** visual style. You (the implementing model — Claude, Codex, or otherwise) did not author this visual direction; it is fixed here and already shipped in the reference implementation. Your job is to produce UI that looks as if it came from a single, opinionated senior designer building a technical instrument panel. Do not improvise the aesthetic. Do not "modernize" the palette. Follow the rules; deviate only where this document explicitly grants latitude.
>
> **How to use.** Read the entire file before writing any UI code. Treat the token block in §3 as the single source of truth for color, type, and surface values. Reference tokens by name (`--bg2`, `--sky`, `--rose`…) — never invent a new color. When in doubt, choose the more restrained option. End every UI task by running the **Compliance Checklist** (§10) against your own output.
>
> **Provenance.** This language was extracted from the shipped `jauderho/synoanalyzer` `index.html` (a single self-contained file) and named **Cyberdeck**. That file is the reference implementation — when this document and the code disagree, flag it; don't silently pick one.

---

## 0. Design philosophy (the "why," in five lines)

1. **Instrument panel, not dashboard fluff.** A deep-space navy canvas with data rendered like telemetry — dense, precise, calm under load. The screen should read as a piece of equipment, not a brochure.
2. **Two voices: sans for chrome, mono for data.** Outfit carries everything operational (labels, headings, body, numbers); JetBrains Mono is reserved for anything *literal* — file paths, IDs, destinations, commands, code. **This split is the identity.**
3. **Cyan is structural; the spectrum is semantic.** One structural accent — sky/cyan — owns chrome, active states, focus, and links. A disciplined multi-hue set (green/amber/rose/violet/teal/orange) carries *meaning* (severity, category, status) — never decoration.
4. **Glow earns the "cyber."** A single sky→violet gradient (the logo), soft neon-tinted shadows, a frosted-blur header, and a faint radial ambient glow are the sanctioned flourishes. Everything else stays flat and quiet.
5. **Dark-first, legible-always.** Dark is the native theme; light is a tuned equal, never an afterthought. WCAG AA contrast is a hard floor — tertiary text was deliberately tuned to clear 4.5:1 in *both* themes.

---

## 1. Non-negotiables (violating any of these is a failed task)

- **Default to dark mode** with a working light/dark toggle, persisted via `localStorage`, applied as `data-theme="dark|light"` on `<html>`. (Cyberdeck ships as standalone single-file apps — `localStorage` is expected here.)
- **Tokens only.** All colors come from the §3 `[data-theme]` blocks. No new hex, no off-palette hue. Per-item "series" colors come only from the sanctioned categorical array (§3).
- **Two fonts, strictly split.** **Outfit** (sans) for all chrome/UI/labels/headings/numbers; **JetBrains Mono** for literal data only (paths, IDs, destinations, filenames, commands, inline code, the logo glyph). Never set a label/heading/body in mono; never set a path/ID/command in sans. Forbidden as primary faces: Inter, Roboto, Arial, SF Pro, Geist, any serif.
- **Sky is structural, not semantic.** `--sky` is reserved for chrome: section ticks, the active-tab ring, focus, links, primary buttons. It is **not** a category color and **not** a status. Meaning is carried by green/amber/rose/violet/teal/orange.
- **Legibility floor.** No text below **11px**; body base is **15px**. Aligned numeric readouts use `tabular-nums`. (The reference shipped a full pass purging all 9–10px text.)
- **Accessibility floor:** WCAG AA contrast (≥4.5:1 for text) in *both* themes — verify computed pairs, don't eyeball; color is never the sole signal (pair it with a glyph, pill, border, or label); visible hover/focus on every interactive element; comfortable hit targets (icon buttons ≥36px).
- **Self-contained, zero-build.** One HTML file with an inline `<style>`; no framework, no bundler, no runtime dependency beyond Google Fonts. **All data-viz is hand-built CSS/SVG — no chart library.**

---

## 2. Typography

```
UI / chrome / labels / headings / numbers:  'Outfit', sans-serif                 (300, 400, 500, 600, 700)
Literal data / paths / IDs / code / glyph:   'JetBrains Mono', monospace          (400, 500, 600)
```

Loaded via a Google Fonts `<link>`:
`https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap`

| Role | Face | Size | Weight | Notes |
|---|---|---|---|---|
| Hero readout (gauge score) | Outfit | 38px | 700 | line-height 1; the dominant answer-number (§7) |
| Big metric (`ov-big`) | Outfit | 30px | 700 | trailing unit in 16px/600 `--t3` |
| Stat-card value | Outfit | 26px | 700 | line-height 1.1; colored only when it carries meaning |
| Drift hero number | Outfit | 20px | 700 | |
| Card / panel heading | Outfit | 14–16px | 600 | |
| Wordmark | Outfit | 16px | 600 | |
| **Section title (eyebrow)** | Outfit | 12px | 600 | uppercase, tracking .04em, `--t2`, preceded by a 3px `--sky` tick |
| Tab | Outfit | 14px | 500 (600 active) | |
| Body / detail | Outfit | 13px | 400 | line-height 1.5–1.6 |
| Field label / caption | Outfit | 11px | 600 | uppercase, tracking .03em, `--t3` |
| Badge / pill / chip | Outfit | 11px | 500–700 | radius-full |
| Severity pill | Outfit | 11px | 700 | uppercase, tracking .06em |
| **Mono data** (path / dest / ID) | JetBrains Mono | 11–13px | 400 | `--t2` |
| **Mono code chip** | JetBrains Mono | 11px | 400 | `--bg1`/`--bg2` fill, 1px border, radius 4–5px, often `--sky` text |

Mono is for *data you could copy-paste*; sans is for *everything you read*. If a string is a label about data, it's sans; if it's the data itself, it's mono.

---

## 3. Design tokens — single source of truth

Two `[data-theme]` blocks. **Dark first, light second.** Every accent ships with a translucent `-D` fill (dark `.12` / light `.13`) used for pill/badge/chip backgrounds and soft icon tiles.

```
/* Backgrounds — cool, layered ink → navy */
                  DARK        LIGHT
bg        #07091a    /   #f0f4fc     page base (near-black navy / cool paper)
bg1       #0c1226    /   #e6edf8     recessed: header, tab-bar, tracks, chips, inset wells
bg2       #111930    /   #ffffff     cards / primary surface
bg3       #162040    /   #f8faff     raised: hover, active tab
border    #1c2d4a    /   #cdd8ee
borderL   #162440    /   #dce6f5     quiet inner divider (note: darker than `border` in dark)

/* Text — never pure #fff / #000 */
t1        #e8eeff    /   #0f1b35     primary
t2        #8ba3c5    /   #4a6585     secondary
t3        #7a93b4    /   #4f6580     muted labels — TUNED to clear AA (≈5.5:1 dark / ≈5:1 light)

/* Structural accent — sky/cyan (chrome, active, focus, links, primary) */
sky       #38bdf8    /   #0284c7
skyD      rgba(56,189,248,.12) / rgba(2,132,199,.13)

/* Semantic + categorical — each with a matching -D fill (dark .12 / light .13) */
green     #34d399    /   #059669     good / success
amber     #fbbf24    /   #d97706     warning / medium severity
rose      #fb7185    /   #e11d48     bad / high severity / danger
purple    #a78bfa    /   #7c3aed     category (e.g. media) + the gradient partner to sky
teal      #2dd4bf    /   #0d9488     category (e.g. local / apps)
orange    #fb923c    /   #ea580c     category (e.g. cloud)

glow      rgba(56,189,248,.05) / rgba(2,132,199,.04)   radial ambient glow

/* Per-item categorical series (assign by index; bright, for dots/bars/accent strips) */
['#38bdf8','#34d399','#a78bfa','#fbbf24','#fb7185','#2dd4bf','#fb923c','#e879f9','#a3e635','#f43f5e']
```

**Surface ladder:** `bg` (page) → `bg1` (recessed wells, tracks, chips, header, tab-bar) → `bg2` (cards) → `bg3` (raised/hover/active). Borders: `border` for card edges, `borderL` for quiet inner dividers.

**The `-D` rule:** a pill/badge/icon-tile is `background: var(--hueD)` + `color: var(--hue)` + (optional) `1px solid var(--hue)`. Neutral chips use `--bg1` + `--border` + `--t2`.

---

## 4. Spacing & layout

- **4px grid**; lean on the 8 / 10 / 12 / 14 / 18 / 20 / 24 / 26 / 32 rhythm. Card padding 12–24px.
- **Radius scale:** 8px (icon buttons, active tab, logo glyph) · 9–10px (buttons, icon tiles) · 12px (compact rows + the segmented tab-bar) · 14px (stat cards, summaries, popovers) · 16px (primary cards, coverage wrap, hero) · 20px (the drop-zone hero) · **100px** (pills/badges/chips) · 4–6px (inline code & flag chips).
- **Header:** sticky, `top:0`, height 56px, `--bg1` with a 1px bottom `border` and **`backdrop-filter: blur(8px)`**; 28px horizontal padding; `justify-content: space-between` with a left group (wordmark + brand-link icon) and a right group (icon buttons). On mobile, padding drops to 16px.
- **Main:** `max-width: 1280px`, centered, padding `32px 28px` (→ `20px 16px` ≤900px).
- **Grids:** content cards use `repeat(auto-fill, minmax(320px, 1fr))` (or 248–280px for denser cards); fixed summary rows use `repeat(5,1fr)` collapsing to 3 (≤900px) then 2 (≤600px). Row gaps 8–14px.
- **Equal-height rows:** let grid stretch cells; don't opt out. Adjacent cards in a row share height.
- **No crowding:** absolutely-positioned popovers/tooltips are `position: fixed`, viewport-clamped, and flip above their anchor when there's no room below — verify clearance against bounding rects, not by eye.
- **Large-list performance:** repeated rows (`.task-card`, `.ret-row`, `.integ-row`, `.dir-card`) use `content-visibility:auto` + a `contain-intrinsic-size` estimate so thousands render instantly; **restore `content-visibility:visible` inside `@media print`** so nothing is skipped on paper.
- Verify reflow at **390 / 768 / 1280px**.

---

## 5. Components (match these recipes — don't invent parallel ones)

- **Card** — `--bg2`, 1px `border`, radius 14–16; hover lifts `translateY(-2px)` + soft shadow `0 8px 32px rgba(0,0,0,.15)`. Category/severity emphasis via a **3px top accent strip** (absolute, the card's hue) or a **3px colored left border**.
- **Stat card** — uppercase 11px/600 `--t3` label · 26px/700 value (color *only* when meaningful: `--green`/`--rose`) · 12px `--t2` meta.
- **Segmented tab-bar** — container `--bg1`, radius 12, 1px border, 5px pad; tabs radius 8, 14px/500 `--t2`. Hover → `--bg2` + `--t1`. **Active → `--bg3` + 600 + a sky inset ring** `box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--sky) 50%, transparent), 0 2px 8px rgba(0,0,0,.3)`.
- **Section title** — 12px/600 uppercase `--t2`, preceded by a **3px × 14px rounded `--sky` tick** (`::before`).
- **Pills / badges / chips** — radius-full, 11px; status/category use the `-D` fill + solid hue text (+ optional 1px hue border); neutral chips use `--bg1` + `border` + `--t2`. **Severity pills**: 11px/700, uppercase, tracking .06em.
- **Severity ramp (canonical):** `high → rose`, `medium → amber`, `low → sky`, `good → green` — expressed together as a left-border + a soft `-D` icon tile + a pill. (Note `low` is the one place sky reads as a level, paired with an icon — still chrome-adjacent, not a category.)
- **Category/type colors:** assign consistently (the reference uses home=sky, media=purple, vm=rose, backup=amber, apps=teal, general=`--t2`; cloud=orange, local=teal). Type badges = `-D` fill + hue text.
- **Icon tile** — 34–36px rounded square (radius 10), `-D` hue fill, emoji/glyph centered.
- **Icon button** — 36×36, radius 8, `--bg2` + border; hover → `--bg3` + `--t1` + `--sky` border.
- **Button** (`.btn`) — Outfit 13/500, `--bg2`, 1px border, radius 9; hover → sky border + sky text + `--skyD` fill. `.btn-primary` = `--skyD` fill + sky border/text. `.btn-danger` hover = rose.
- **Data table** — wrapper `--bg2` + radius 16 + `overflow-x:auto`; header row `--bg1`, 12px/600, slight tracking; cells 10–13px with `borderL` row rules; **path cells in mono**; status cells use a 22px round `cov-check` (`-D` fill for yes, `--bg1` @ .4 opacity for no).
- **Popover / tooltip** — `position:fixed`, `--bg2`, 1px border, radius 12–14, shadow `0 16px 48px rgba(0,0,0,.35)`; `pointer-events:none` until "pinned" (then `auto`); contains mono code chips. Auto-positioned + viewport-clamped.
- **Drop-zone hero** — dashed 2px border card, radius 20, a radial `--glow` `::before`; hover/drag → solid `--sky` border + `translateY(-2px)` + sky-tinted shadow `0 16px 48px rgba(56,189,248,.08)`.
- **Empty state** — centered 13px `--t3`, 40px padding; or a richer empty card (big 38px glyph + 16px heading + muted line + one `.btn-primary`).
- **Icons** — emoji used as section/severity/category glyphs (the reference's idiom); if a vector family is needed, use one inline-SVG set at 1.5–2px stroke. Keep it to one family.

---

## 6. Motion

- Chrome transitions: **0.2–0.3s ease** on color/background/border/transform. Theme switch animates `background`/`color` at 0.3s.
- **Signature hovers:** cards lift `translateY(-2px)` + soft shadow; list/suggestion rows nudge `translateX(3px)`; icon buttons & tabs shift background/border/text.
- **Fills animate, numbers don't.** Gauge arcs and track/progress bars animate `width` at ~0.4s ease. There is **no digit-roll / odometer** here — Cyberdeck animates the bar, not the number.
- Tab content swaps instantly via display toggle — no heavy entrance animation.
- **No bounce, no parallax, no decorative motion.**
- Honor `prefers-reduced-motion: reduce` → near-zero durations, no transforms. (This is a required floor for new work.)

---

## 7. The lead readout (mandatory for results)

Every Cyberdeck view leads with the answer, in two tiers:

1. **A stats bar** — a fixed grid (≈5 across) of stat cards: uppercase muted label, 26px/700 value (colored only when meaningful), 12px meta line. This is the at-a-glance summary strip directly under the header.
2. **A hero readout** below it — the dominant figure the user came for, typically a **donut score gauge** (§8) paired with a breakdown, or a single big metric (`ov-big`, 30px/700 with a muted unit suffix).

**Color discipline:** the hero number stays `--t1` unless its grade/sign is the meaning (a risk/loss may take `--rose`, a healthy score `--green`, mid `--amber`). Sky is never used to color a value — it's chrome.

**Breathing room:** hero cards hold ≥16px vertical padding; the number never touches a card edge; line-height ~1.0–1.1 is fine for the geometric Outfit numerals (unlike serif faces).

---

## 8. Data visualization (hand-built CSS/SVG — no library)

Cyberdeck draws its own viz. No Recharts, no D3, no canvas lib.

1. **Donut score gauge** — two concentric SVG `<circle>`s on a `viewBox="0 0 120 120"`, `transform: rotate(-90deg)`: a track ring in `--bg1` and an arc in the grade hue, `stroke-width:12`, `stroke-linecap:round`, the arc fraction set via `stroke-dasharray`/`stroke-dashoffset`. Centered overlay: 38px/700 score + a muted "out of N" + a grade label colored by band (`≥80 green` / `≥50 amber` / `else rose`). The arc animates by transitioning `stroke-dashoffset`/width.
2. **Track / breakdown bars** — a `--bg1` track (5–8px tall, radius 4) with a hue fill whose `width:%` animates at 0.4s. Lay out as `label | track | value%` grids; values in `tabular-nums`.
3. **Segmented matrices** — coverage/status grids built from table cells with round status tokens (`-D` fill ✓ / dim `--bg1` —); never a heatmap of arbitrary colors.
4. **Series coloring** — per-item dots/strips/bars pull from the categorical array (§3) by index. Two adjacent series must be **distinct hues, not tints** of one.
5. **Direct labeling.** Put the value on the bar/row (right-aligned `tabular-nums`), not in a separate legend. Status is always also a glyph (✓ / — / ⚠), never color alone.
6. **Quiet chrome.** Tracks and dividers stay in `--bg1`/`--borderL`; the data (fills, hues) is the only loud thing. No 3D, no shadows on bars, no gradient fills except the sanctioned ambient glow.
7. **Tooltips/popovers** match the §5 popover recipe: `--bg2`, 1px border, radius 12–14, mono for any literal values.

---

## 9. Anti-patterns — DO NOT

- ❌ Cross the font split: mono for labels/headings/body, or sans for paths/IDs/commands/code. The split is sacred.
- ❌ Introduce a hex outside the §3 blocks, or a new accent hue. Series colors come only from the sanctioned array.
- ❌ Use `--sky` as a category or a status. Sky is chrome (section ticks, active tab, focus, links, primary). Meaning lives in green/amber/rose/violet/teal/orange.
- ❌ Text below 11px; body below 15px; pure `#000`/`#fff` text or fills.
- ❌ Decorative or rainbow gradients. Gradient is limited to the sky→violet logo glyph and the radial ambient `--glow`.
- ❌ Glow everything. Glow/neon shadow is reserved for the logo, active states, and the upload hero; ordinary cards stay flat with quiet `rgba(0,0,0,…)` shadows.
- ❌ Pull in a chart library, UI framework, or build step. Cyberdeck is one self-contained HTML file; viz is hand-built CSS/SVG.
- ❌ Ship a dark-only (or light-only) build, or let either theme fail AA contrast.
- ❌ Skip the print palette override — printing the dark theme wastes ink and is unreadable.
- ❌ Animate numbers with an odometer/digit-roll (that belongs to a different style), add bounce/parallax, or ignore `prefers-reduced-motion`.
- ❌ Convey status by color alone (always add a glyph/pill/label).
- ❌ Forget to restore `content-visibility` in print, leaving rows blank on paper.

---

## 10. Compliance checklist (run before declaring done)

- [ ] Dark is default; light/dark toggle works and persists to `localStorage` (`data-theme` on `<html>`); both themes checked deliberately (contrast, glows, borders, the `-D` fills on white).
- [ ] Zero hex outside the §3 blocks; series colors from the sanctioned array; `--sky` used only structurally.
- [ ] Outfit for all chrome/UI/numbers; JetBrains Mono **only** for literal data (paths/IDs/destinations/commands/code/logo glyph); no crossover either direction.
- [ ] No text below 11px; body 15px; `tabular-nums` on aligned figures.
- [ ] `--t1`/`--t2`/`--t3` each clear WCAG AA (≥4.5:1) on `--bg`/`--bg1`/`--bg2`/`--bg3` in **both** themes — verified via computed contrast, not eyeballed.
- [ ] Signature elements present where appropriate: sky→violet gradient logo + glow, frosted (`blur`) header, section-title sky tick, active-tab sky inset ring (`color-mix`).
- [ ] Severity ramp (high rose / med amber / low sky / good green) and category hues applied consistently as glyph + pill + border + text — never color alone.
- [ ] Surface ladder respected: `bg2` cards, `bg1` recessed wells/tracks/chips, `bg3` raised/hover/active; radii from the §4 scale.
- [ ] Hero readout leads results (stats bar + donut gauge / big metric); value colored only when meaningful; never sky.
- [ ] Viz is hand-built CSS/SVG; gauge arcs + track bars animate `width`; no chart lib; distinct hues not tints; values directly labeled.
- [ ] `@media print` overrides both theme blocks to a forced light ink palette, hides chrome (`.btn-icon`, `.tab-bar`, buttons), expands all `.tab-content`, un-grids/stacks cards, `break-inside:avoid`, and restores `content-visibility:visible`; `document.title` set to a report title around `window.print()`.
- [ ] `prefers-reduced-motion` honored; hover/focus/empty states present; icon buttons ≥36px.
- [ ] Single self-contained `index.html`; zero build/deps beyond Google Fonts; verified at 390 / 768 / 1280px.

---

## 11. Where the tokens live

Cyberdeck is **single-file by design** — there is no shared package and no per-app duplication to keep in sync.

| Concern | Home |
|---|---|
| Color/surface/text tokens | inline `<style>` → `[data-theme="dark"]` / `[data-theme="light"]` blocks at the top of `index.html` |
| Per-item series palette | the `TASK_COLORS` JS array (the §3 categorical set) |
| Print palette | the `@media print` block (re-declares both `[data-theme]` blocks; later source order + equal specificity overrides the active theme) |
| Fonts | the Google Fonts `<link>` in `<head>` |
| Theme state | persisted in `localStorage`; applied as `data-theme` on `<html>` |

Reference implementation: `jauderho/synoanalyzer` `index.html`. A local static server (e.g. `python3 -m http.server`) is enough to preview — there is nothing to build.
