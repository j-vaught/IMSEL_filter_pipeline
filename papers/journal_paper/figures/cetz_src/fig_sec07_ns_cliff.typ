#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let congaree  = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let rose      = rgb("#CC2E40")
#let black90   = rgb("#363636")
#let black70   = rgb("#5C5C5C")
#let black50   = rgb("#A2A2A2")
#let black30   = rgb("#C7C7C7")
#let white     = rgb("#FFFFFF")

// Data extracted from outputs/BSDS500_image_ablation/low_ns_results.json
// d=2 polynomial degree, ODS vs Ns for various Np values
#let data-np15 = (
  (2, 0.630550),
  (3, 0.838924),
  (4, 0.838609),
  (5, 0.839864),
  (6, 0.838924),
  (7, 0.839884),
  (8, 0.839476),
  (9, 0.839657),
  (12, 0.840167),
  (18, 0.839657),
  (36, 0.840104),
)

#let data-np25 = (
  (2, 0.618062),
  (3, 0.848957),
  (4, 0.845911),
  (5, 0.847781),
  (6, 0.848957),
  (7, 0.847567),
  (8, 0.847960),
  (9, 0.847323),
  (12, 0.847831),
  (18, 0.847323),
  (36, 0.847475),
)

#let data-np50 = (
  (2, 0.622518),
  (3, 0.857384),
  (4, 0.860273),
  (5, 0.859584),
  (6, 0.857384),
  (7, 0.859556),
  (8, 0.861149),
  (9, 0.859752),
  (12, 0.859638),
  (18, 0.859752),
  (36, 0.859585),
)

#let data-np100 = (
  (2, 0.615958),
  (3, 0.825537),
  (4, 0.827831),
  (5, 0.826832),
  (6, 0.825537),
  (7, 0.826587),
  (8, 0.827803),
  (9, 0.826835),
  (12, 0.827443),
  (18, 0.826835),
  (36, 0.826852),
)

#cetz.canvas({
  import cetz.draw: *

  // Plot dimensions
  let pw = 8.0
  let ph = 5.5
  let ox = 1.6
  let oy = 1.0

  // X range: 2 to 36 (Ns)
  let x-min = 1
  let x-max = 38

  // Y range: 0.58 to 0.88
  let y-min = 0.58
  let y-max = 0.88

  let tx(v) = ox + (v - x-min) / (x-max - x-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  // Axes
  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox + pw/2, oy - 0.7), text(fill: black90, size: 10pt)[$N_s$ (number of orientations)])

  // Y-axis label
  content(
    (ox - 1.15, oy + ph/2),
    text(fill: black90, size: 10pt)[ODS F-score],
    angle: 90deg,
  )

  // X ticks
  let x-ticks = (2, 3, 6, 9, 12, 18, 24, 30, 36)
  for v in x-ticks {
    let x = tx(v)
    line((x, oy), (x, oy - 0.1), stroke: 0.5pt + black70)
    content((x, oy - 0.35), text(fill: black70, size: 8pt)[#v])
    line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
  }

  // Y ticks
  let y-ticks = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.1, y), stroke: 0.5pt + black70)
    let label = str(calc.round(v, digits: 2))
    content((ox - 0.45, y), text(fill: black70, size: 8pt)[#label])
    line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
  }

  // Draw curves
  let draw-curve(data, col, thick) = {
    for i in range(data.len() - 1) {
      let (x1, y1) = data.at(i)
      let (x2, y2) = data.at(i + 1)
      line((tx(x1), ty(y1)), (tx(x2), ty(y2)), stroke: thick + col)
    }
    for pt in data {
      let (x, y) = pt
      circle((tx(x), ty(y)), radius: 0.08, fill: col, stroke: 0.4pt + col.darken(20%))
    }
  }

  draw-curve(data-np15, garnet, 1.2pt)
  draw-curve(data-np25, atlantic, 1.2pt)
  draw-curve(data-np50, congaree, 1.2pt)
  draw-curve(data-np100, horseshoe, 1.2pt)

  // Vertical annotation line at Ns=3
  let x3 = tx(3)
  line((x3, oy), (x3, oy + ph), stroke: (paint: rose, thickness: 0.8pt, dash: "dashed"))

  // Legend
  let lx = ox + pw - 2.6
  let ly = oy + 1.8
  let ls = 0.38

  rect((lx - 0.15, ly + 0.22), (lx + 2.5, ly - 4*ls - 0.15), fill: white, stroke: 0.4pt + black30)

  // Np=15
  line((lx, ly), (lx + 0.35, ly), stroke: 1.2pt + garnet)
  circle((lx + 0.175, ly), radius: 0.06, fill: garnet, stroke: none)
  content((lx + 0.5, ly), text(fill: black90, size: 8pt)[$N_p = 15$], anchor: "west")

  // Np=25
  line((lx, ly - ls), (lx + 0.35, ly - ls), stroke: 1.2pt + atlantic)
  circle((lx + 0.175, ly - ls), radius: 0.06, fill: atlantic, stroke: none)
  content((lx + 0.5, ly - ls), text(fill: black90, size: 8pt)[$N_p = 25$], anchor: "west")

  // Np=50
  line((lx, ly - 2*ls), (lx + 0.35, ly - 2*ls), stroke: 1.2pt + congaree)
  circle((lx + 0.175, ly - 2*ls), radius: 0.06, fill: congaree, stroke: none)
  content((lx + 0.5, ly - 2*ls), text(fill: black90, size: 8pt)[$N_p = 50$], anchor: "west")

  // Np=100
  line((lx, ly - 3*ls), (lx + 0.35, ly - 3*ls), stroke: 1.2pt + horseshoe)
  circle((lx + 0.175, ly - 3*ls), radius: 0.06, fill: horseshoe, stroke: none)
  content((lx + 0.5, ly - 3*ls), text(fill: black90, size: 8pt)[$N_p = 100$], anchor: "west")
})
