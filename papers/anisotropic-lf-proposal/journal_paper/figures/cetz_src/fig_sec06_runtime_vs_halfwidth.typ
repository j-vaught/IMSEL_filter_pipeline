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

#cetz.canvas({
  import cetz.draw: *

  // Plot dimensions
  let pw = 9.0   // plot width
  let ph = 6.5   // plot height
  let ox = 1.5   // origin x
  let oy = 0.8   // origin y

  // Data ranges
  let x-min = 0
  let x-max = 15
  let y-min = 0
  let y-max = 10

  // Transform functions
  let tx(v) = ox + (v - x-min) / (x-max - x-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  // === AXES ===
  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90, mark: (end: "stealth", fill: black90, scale: 0.4))
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90, mark: (end: "stealth", fill: black90, scale: 0.4))

  // X-axis label
  content((ox + pw/2, oy - 0.65), text(fill: black90, size: 10pt)[Line half-width $m$])

  // Y-axis label
  content(
    (ox - 1.1, oy + ph/2),
    text(fill: black90, size: 10pt)[Runtime per image (s)],
    angle: 90deg,
  )

  // X ticks
  let x-ticks = (0, 1, 2, 5, 7, 10, 14)
  for v in x-ticks {
    let x = tx(v)
    line((x, oy), (x, oy - 0.1), stroke: 0.5pt + black70)
    content((x, oy - 0.3), text(fill: black70, size: 8pt)[#v])
  }

  // Y ticks
  let y-ticks = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.1, y), stroke: 0.5pt + black70)
    content((ox - 0.4, y), text(fill: black70, size: 8pt)[#v])
    // grid line
    if v > 0 {
      line((ox, y), (ox + pw, y), stroke: 0.3pt + black30)
    }
  }

  // === DATA ===

  // Naive LF (dashed, Black90)
  let naive = ((1, 0.58), (7, 2.43), (14, 8.88))
  for i in range(naive.len() - 1) {
    let (x1, y1) = naive.at(i)
    let (x2, y2) = naive.at(i + 1)
    line((tx(x1), ty(y1)), (tx(x2), ty(y2)), stroke: (paint: black90, thickness: 1.5pt, dash: "dashed"))
  }
  for (x, y) in naive {
    circle((tx(x), ty(y)), radius: 0.08, fill: black90, stroke: none)
  }

  // Fused stencil (Garnet, solid)
  let fused = ((1, 0.072), (7, 0.13), (14, 0.37))
  for i in range(fused.len() - 1) {
    let (x1, y1) = fused.at(i)
    let (x2, y2) = fused.at(i + 1)
    line((tx(x1), ty(y1)), (tx(x2), ty(y2)), stroke: 1.5pt + garnet)
  }
  for (x, y) in fused {
    circle((tx(x), ty(y)), radius: 0.08, fill: garnet, stroke: none)
  }

  // Rectangular kernel (Atlantic, flat)
  let rect-y = 0.045
  line((tx(0), ty(rect-y)), (tx(14), ty(rect-y)), stroke: 1.5pt + atlantic)
  circle((tx(0), ty(rect-y)), radius: 0.08, fill: atlantic, stroke: none)
  circle((tx(7), ty(rect-y)), radius: 0.08, fill: atlantic, stroke: none)
  circle((tx(14), ty(rect-y)), radius: 0.08, fill: atlantic, stroke: none)

  // Elliptical kernel (Congaree, flat)
  let ellip-y = 0.047
  line((tx(0), ty(ellip-y)), (tx(14), ty(ellip-y)), stroke: (paint: congaree, thickness: 1.5pt, dash: "dotted"))
  circle((tx(0), ty(ellip-y)), radius: 0.08, fill: congaree, stroke: none)
  circle((tx(7), ty(ellip-y)), radius: 0.08, fill: congaree, stroke: none)
  circle((tx(14), ty(ellip-y)), radius: 0.08, fill: congaree, stroke: none)

  // === ANNOTATION: ~3x gap ===
  // Between fused at m=14 and geometric at m=14
  let ann-x = tx(14) + 0.3
  let ann-top = ty(0.37)
  let ann-bot = ty(0.045)
  // bracket
  line((ann-x, ann-bot), (ann-x + 0.15, ann-bot), stroke: 0.6pt + black70)
  line((ann-x + 0.15, ann-bot), (ann-x + 0.15, ann-top), stroke: 0.6pt + black70)
  line((ann-x, ann-top), (ann-x + 0.15, ann-top), stroke: 0.6pt + black70)
  content((ann-x + 0.7, (ann-top + ann-bot)/2), text(fill: black70, size: 8pt, style: "italic")[~3$times$])

  // === LEGEND ===
  let lx = ox + 1.0
  let ly = oy + ph - 0.3
  let ls = 0.38  // line spacing

  // Legend box background
  rect((lx - 0.15, ly + 0.25), (lx + 3.2, ly - 3*ls - 0.2), stroke: 0.4pt + black30, fill: white)

  // Naive LF
  line((lx, ly), (lx + 0.6, ly), stroke: (paint: black90, thickness: 1.5pt, dash: "dashed"))
  circle((lx + 0.3, ly), radius: 0.06, fill: black90, stroke: none)
  content((lx + 0.75, ly), text(fill: black90, size: 8pt)[Naive LF], anchor: "west")

  // Fused stencil
  line((lx, ly - ls), (lx + 0.6, ly - ls), stroke: 1.5pt + garnet)
  circle((lx + 0.3, ly - ls), radius: 0.06, fill: garnet, stroke: none)
  content((lx + 0.75, ly - ls), text(fill: black90, size: 8pt)[Fused stencil], anchor: "west")

  // Rectangular kernel
  line((lx, ly - 2*ls), (lx + 0.6, ly - 2*ls), stroke: 1.5pt + atlantic)
  circle((lx + 0.3, ly - 2*ls), radius: 0.06, fill: atlantic, stroke: none)
  content((lx + 0.75, ly - 2*ls), text(fill: black90, size: 8pt)[Rectangular kernel], anchor: "west")

  // Elliptical kernel
  line((lx, ly - 3*ls), (lx + 0.6, ly - 3*ls), stroke: (paint: congaree, thickness: 1.5pt, dash: "dotted"))
  circle((lx + 0.3, ly - 3*ls), radius: 0.06, fill: congaree, stroke: none)
  content((lx + 0.75, ly - 3*ls), text(fill: black90, size: 8pt)[Elliptical kernel], anchor: "west")
})
