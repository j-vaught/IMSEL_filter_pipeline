#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")
#let white = rgb("#FFFFFF")

// Data extracted from ablation_metrics.json: WVF with Ns=18, degrees 2, 3, 4, 5
// Format: (Np, ODS)

#let d2-data = (
  (8, 0.7947), (10, 0.7975), (12, 0.8166), (15, 0.8229), (20, 0.8379),
  (25, 0.8418), (30, 0.8388), (40, 0.8316), (50, 0.8213), (65, 0.8036),
  (80, 0.7909), (100, 0.7693), (130, 0.7409), (160, 0.7163), (200, 0.6924),
  (250, 0.6744), (300, 0.6569), (400, 0.6251), (500, 0.6003),
)

#let d3-data = (
  (10, 0.7045), (12, 0.7060), (15, 0.7398), (20, 0.7558), (25, 0.7948),
  (30, 0.8018), (40, 0.8219), (50, 0.8257), (65, 0.8365), (80, 0.8399),
  (100, 0.8348), (130, 0.8169), (160, 0.8028), (200, 0.7893), (250, 0.7660),
  (300, 0.7428), (400, 0.7077), (500, 0.6850),
)

#let d4-data = (
  (15, 0.7060), (20, 0.7558), (25, 0.7975), (30, 0.8004), (40, 0.8127),
  (50, 0.8255), (65, 0.8363), (80, 0.8399), (100, 0.8348), (130, 0.8171),
  (160, 0.8028), (200, 0.7892), (250, 0.7662), (300, 0.7428), (400, 0.7077),
  (500, 0.6850),
)

#let d5-data = (
  (25, 0.7250), (30, 0.7272), (40, 0.7580), (50, 0.7749), (65, 0.8038),
  (80, 0.8175), (100, 0.8285), (130, 0.8356), (160, 0.8356), (200, 0.8276),
  (250, 0.8142), (300, 0.8028), (400, 0.7740), (500, 0.7504),
)

#cetz.canvas({
  import cetz.draw: *

  // Plot dimensions
  let pw = 10.0
  let ph = 6.5
  let ox = 1.6
  let oy = 1.0

  // X range: 0 to 520
  let x-min = 0
  let x-max = 520

  // Y range: 0.58 to 0.86
  let y-min = 0.58
  let y-max = 0.86

  let tx(v) = ox + (v - x-min) / (x-max - x-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  // === AXES ===
  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox + pw / 2, oy - 0.7), text(fill: black90, size: 10pt)[Support size $N_p$])

  // Y-axis label
  content(
    (ox - 1.2, oy + ph / 2),
    text(fill: black90, size: 10pt)[ODS F-score],
    angle: 90deg,
  )

  // X ticks
  let x-tick-vals = (0, 100, 200, 300, 400, 500)
  for v in x-tick-vals {
    let x = tx(v)
    line((x, oy), (x, oy - 0.1), stroke: 0.5pt + black70)
    content((x, oy - 0.35), text(fill: black70, size: 8pt)[#v])
    if v > 0 {
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
    }
  }

  // Y ticks
  let y-ticks = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.1, y), stroke: 0.5pt + black70)
    let label = str(calc.round(v, digits: 2))
    content((ox - 0.5, y), text(fill: black70, size: 8pt)[#label])
    line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
  }

  // === HELPER FUNCTIONS ===
  let draw-curve(data, col, line-style) = {
    for i in range(data.len() - 1) {
      let (x1, y1) = data.at(i)
      let (x2, y2) = data.at(i + 1)
      line((tx(x1), ty(y1)), (tx(x2), ty(y2)), stroke: line-style)
    }
  }

  let draw-markers(data, col, rad) = {
    for (x, y) in data {
      circle((tx(x), ty(y)), radius: rad, fill: col, stroke: 0.5pt + col.darken(20%))
    }
  }

  // === CURVES ===
  let lw = 1.5pt
  let marker-r = 0.08

  // d=2: garnet, solid
  draw-curve(d2-data, garnet, (paint: garnet, thickness: lw))
  draw-markers(d2-data, garnet, marker-r)

  // d=3: atlantic, solid
  draw-curve(d3-data, atlantic, (paint: atlantic, thickness: lw))
  draw-markers(d3-data, atlantic, marker-r)

  // d=4: congaree, solid
  draw-curve(d4-data, congaree, (paint: congaree, thickness: lw))
  draw-markers(d4-data, congaree, marker-r)

  // d=5: horseshoe, solid
  draw-curve(d5-data, horseshoe, (paint: horseshoe, thickness: lw))
  draw-markers(d5-data, horseshoe, marker-r)

  // === LEGEND ===
  let lx = ox + pw - 2.4
  let ly = oy + 1.4
  let ls = 0.42

  rect((lx - 0.15, ly + 0.25), (lx + 2.3, ly - 4 * ls - 0.1), fill: white, stroke: 0.4pt + black30)

  // d=2
  line((lx, ly), (lx + 0.4, ly), stroke: (paint: garnet, thickness: lw))
  circle((lx + 0.2, ly), radius: marker-r, fill: garnet, stroke: 0.5pt + garnet.darken(20%))
  content((lx + 0.55, ly), text(fill: black90, size: 8pt)[$d = 2$], anchor: "west")

  // d=3
  line((lx, ly - ls), (lx + 0.4, ly - ls), stroke: (paint: atlantic, thickness: lw))
  circle((lx + 0.2, ly - ls), radius: marker-r, fill: atlantic, stroke: 0.5pt + atlantic.darken(20%))
  content((lx + 0.55, ly - ls), text(fill: black90, size: 8pt)[$d = 3$], anchor: "west")

  // d=4
  line((lx, ly - 2 * ls), (lx + 0.4, ly - 2 * ls), stroke: (paint: congaree, thickness: lw))
  circle((lx + 0.2, ly - 2 * ls), radius: marker-r, fill: congaree, stroke: 0.5pt + congaree.darken(20%))
  content((lx + 0.55, ly - 2 * ls), text(fill: black90, size: 8pt)[$d = 4$], anchor: "west")

  // d=5
  line((lx, ly - 3 * ls), (lx + 0.4, ly - 3 * ls), stroke: (paint: horseshoe, thickness: lw))
  circle((lx + 0.2, ly - 3 * ls), radius: marker-r, fill: horseshoe, stroke: 0.5pt + horseshoe.darken(20%))
  content((lx + 0.55, ly - 3 * ls), text(fill: black90, size: 8pt)[$d = 5$], anchor: "west")
})
