#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet = rgb("#73000A")
#let rose = rgb("#CC2E40")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let white = rgb("#FFFFFF")

#let clean-data = (
  ([Circle SG $d=1$], 2.32, 0.6958, black70, "west", 0.16, -0.18),
  ([Iso. Gaussian], 3.69, 0.6866, honeycomb, "west", 0.16, 0.18),
  ([Rectangular], 3.54, 0.7390, rose, "west", 0.16, -0.20),
  ([Elliptical], 3.56, 0.7409, horseshoe, "west", 0.16, 0.18),
  ([WVF $d=2$], 18.15, 0.8068, garnet, "east", -0.16, 0.18),
  ([WVF $d=4$], 21.98, 0.7980, black50, "west", 0.16, -0.18),
  ([LF Triton $d=2$], 18.23, 0.7086, atlantic, "west", 0.16, -0.18),
  ([LF Triton $d=4$], 37.90, 0.7410, congaree, "west", 0.16, 0.0),
)

#cetz.canvas({
  import cetz.draw: *

  let pw = 10.2
  let ph = 6.6
  let ox = 1.5
  let oy = 1.0

  let lx-min = 0.25
  let lx-max = 1.65
  let y-min = 0.67
  let y-max = 0.82

  let tx(v) = ox + (calc.log(v, base: 10) - lx-min) / (lx-max - lx-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

  content((ox + pw / 2, oy - 0.72), text(fill: black90, size: 10pt)[Median runtime per image (ms)])
  content((ox - 1.35, oy + ph / 2), angle: 90deg, text(fill: black90, size: 10pt)[ODS F-score])

  let x-ticks = (2, 3, 5, 10, 20, 40)
  for v in x-ticks {
    let x = tx(v)
    line((x, oy), (x, oy - 0.10), stroke: 0.5pt + black70)
    line((x, oy), (x, oy + ph), stroke: 0.25pt + black30)
    content((x, oy - 0.32), text(fill: black70, size: 8pt)[#str(v)])
  }

  let y-ticks = (0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.10, y), stroke: 0.5pt + black70)
    line((ox, y), (ox + pw, y), stroke: 0.25pt + black30)
    content((ox - 0.48, y), anchor: "east", text(fill: black70, size: 8pt)[#str(calc.round(v, digits: 2))])
  }

  let pareto = (
    (2.32, 0.6958),
    (3.56, 0.7409),
    (18.15, 0.8068),
  )
  for idx in range(pareto.len() - 1) {
    let (x1, y1) = pareto.at(idx)
    let (x2, y2) = pareto.at(idx + 1)
    line(
      (tx(x1), ty(y1)),
      (tx(x2), ty(y2)),
      stroke: (paint: rose, thickness: 1.0pt, dash: "dashed"),
    )
  }

  let point(label, xval, yval, col, anchor, dx, dy) = {
    let px = tx(xval)
    let py = ty(yval)
    circle((px, py), radius: 0.09, fill: col, stroke: 0.4pt + col.darken(20%))
    content((px + dx, py + dy), anchor: anchor, text(fill: black90, size: 7.2pt)[#label])
  }

  for entry in clean-data {
    let (label, xval, yval, col, anchor, dx, dy) = entry
    point(label, xval, yval, col, anchor, dx, dy)
  }

  let lx = ox + 0.25
  let ly = oy + ph - 0.35
  rect((lx - 0.15, ly + 0.24), (lx + 4.2, ly - 0.62), fill: white, stroke: 0.4pt + black30)
  line((lx, ly - 0.18), (lx + 0.6, ly - 0.18), stroke: (paint: rose, thickness: 1.0pt, dash: "dashed"))
  content((lx + 0.78, ly - 0.18), anchor: "west", text(fill: black90, size: 7.6pt)[Pareto frontier, averaged over BIPED v1 and v2])

  let note-x = ox + pw - 2.55
  let note-y = oy + 0.85
  rect((note-x - 1.95, note-y + 0.34), (note-x + 1.95, note-y - 0.34), fill: white, stroke: 0.4pt + black30)
  content((note-x, note-y), text(fill: black90, size: 7.3pt)[Clean pilot. 50 images per BIPED split.])
})
