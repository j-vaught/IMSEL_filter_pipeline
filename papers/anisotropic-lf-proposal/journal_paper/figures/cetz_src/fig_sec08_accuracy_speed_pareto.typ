#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let rose = rgb("#CC2E40")

#cetz.canvas({
  import cetz.draw: *

  // Plot dimensions
  let pw = 11.0
  let ph = 7.5
  let ox = 2.0
  let oy = 1.0

  // Log scale x: 0.3 to 30000 ms
  // We map log10(x) from -0.3 to 4.5
  let lx-min = -0.3
  let lx-max = 4.5

  // Y: 0.60 to 0.95
  let y-min = 0.60
  let y-max = 0.95

  let tx(v) = ox + (calc.log(v, base: 10) - lx-min) / (lx-max - lx-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  // === AXES ===
  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox + pw/2, oy - 0.75), text(fill: black90, size: 10pt)[Runtime per image (ms)])

  // Y-axis label
  content(
    (ox - 1.5, oy + ph/2),
    text(fill: black90, size: 10pt)[ODS F-score (BIPED v1)],
    angle: 90deg,
  )

  // X ticks (log scale)
  let x-tick-vals = (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
  let x-tick-labels = ("0.5", "1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k")
  for (i, v) in x-tick-vals.enumerate() {
    let x = tx(v)
    if x >= ox and x <= ox + pw {
      line((x, oy), (x, oy - 0.1), stroke: 0.5pt + black70)
      content((x, oy - 0.35), text(fill: black70, size: 7pt)[#x-tick-labels.at(i)])
      // light grid
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
    }
  }

  // Y ticks
  let y-ticks = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.1, y), stroke: 0.5pt + black70)
    let label = str(calc.round(v, digits: 2))
    content((ox - 0.55, y), text(fill: black70, size: 7pt)[#label])
    line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
  }

  // === DATA POINTS ===
  let r-small = 0.1
  let r-large = 0.14

  // Helper: point with label
  let pt(x-val, y-val, label, col, rad, label-anchor, dx, dy) = {
    let px = tx(x-val)
    let py = ty(y-val)
    circle((px, py), radius: rad, fill: col, stroke: 0.5pt + col.darken(20%))
    content((px + dx, py + dy), text(fill: black90, size: 7pt)[#label], anchor: label-anchor)
  }

  // Classical (Black70)
  pt(0.5, 0.761, [Sobel], black70, r-small, "east", -0.15, -0.18)
  pt(0.5, 0.772, [Prewitt], black70, r-small, "east", -0.15, 0.18)
  pt(0.8, 0.756, [Kirsch], black70, r-small, "west", 0.15, -0.18)
  pt(1.0, 0.752, [Gauss. Deriv], black70, r-small, "west", 0.15, -0.2)

  // ASF variants (Garnet, larger)
  pt(12, 0.812, [ASF-poly point], garnet, r-large, "east", -0.2, 0.2)
  pt(45, 0.810, [ASF-rect], garnet, r-large, "west", 0.2, -0.2)
  pt(132, 0.802, [ASF-poly line], garnet, r-large, "west", 0.2, 0.0)

  // DL models (Atlantic)
  pt(30, 0.851, [TEED], atlantic, r-small, "west", 0.15, 0.15)
  pt(35, 0.836, [PIDINet], atlantic, r-small, "west", 0.15, -0.15)
  pt(80, 0.903, [DexiNed], atlantic, r-small, "east", -0.15, -0.2)
  pt(80, 0.908, [NBED], atlantic, r-small, "east", -0.15, 0.15)
  pt(15000, 0.909, [DiffusionEdge], atlantic, r-small, "east", -0.15, 0.15)

  // === PARETO FRONTIER ===
  // Non-dominated points (moving right and up): Prewitt -> ASF-poly point -> TEED -> NBED/DexiNed -> DiffusionEdge
  // Actually: lowest runtime with highest F: Prewitt(0.5, 0.772), ASF-poly(12, 0.812), TEED(30, 0.851), NBED(80, 0.908), DiffusionEdge(15000, 0.909)
  let pareto = ((0.5, 0.772), (12, 0.812), (30, 0.851), (80, 0.908), (15000, 0.909))
  for i in range(pareto.len() - 1) {
    let (x1, y1) = pareto.at(i)
    let (x2, y2) = pareto.at(i + 1)
    line(
      (tx(x1), ty(y1)),
      (tx(x2), ty(y2)),
      stroke: (paint: rose, thickness: 1pt, dash: "dashed"),
    )
  }

  // === LEGEND ===
  let lx = ox + pw - 3.2
  let ly = oy + 1.6
  let ls = 0.4

  rect((lx - 0.15, ly + 0.25), (lx + 3.1, ly - 4*ls - 0.1), fill: white, stroke: 0.4pt + black30)

  // Classical
  circle((lx + 0.1, ly), radius: r-small, fill: black70, stroke: 0.5pt + black70.darken(20%))
  content((lx + 0.35, ly), text(fill: black90, size: 7.5pt)[Classical], anchor: "west")

  // ASF
  circle((lx + 0.1, ly - ls), radius: r-large, fill: garnet, stroke: 0.5pt + garnet.darken(20%))
  content((lx + 0.35, ly - ls), text(fill: black90, size: 7.5pt)[ASF (ours)], anchor: "west")

  // DL
  circle((lx + 0.1, ly - 2*ls), radius: r-small, fill: atlantic, stroke: 0.5pt + atlantic.darken(20%))
  content((lx + 0.35, ly - 2*ls), text(fill: black90, size: 7.5pt)[Deep learning], anchor: "west")

  // Pareto
  line((lx - 0.05, ly - 3*ls), (lx + 0.25, ly - 3*ls), stroke: (paint: rose, thickness: 1pt, dash: "dashed"))
  content((lx + 0.35, ly - 3*ls), text(fill: black90, size: 7.5pt)[Pareto frontier], anchor: "west")
})
