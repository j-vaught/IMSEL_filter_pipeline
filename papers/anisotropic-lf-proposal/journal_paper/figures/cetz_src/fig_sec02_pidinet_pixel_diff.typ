#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet  = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let white   = rgb("#FFFFFF")

#cetz.canvas({
  import cetz.draw: *

  let cell = 1.45
  let N    = 3
  let gap  = 2.8

  // shared cell-drawing helper: filled bg + white text (like Sobel/Kirsch figure)
  let draw-cell(x, y, v, label-override) = {
    let fc = if label-override != none { black10 }
             else if v > 0 { garnet.lighten(40%) }
             else if v < 0 { atlantic.lighten(40%) }
             else { white }
    let tc = if label-override != none { black50 }
             else if v == 0 { black90 }
             else { white }
    let sw = if label-override != none { 1.5pt + black90 } else { 0.5pt + black30 }
    rect((x, y), (x + cell, y - cell), fill: fc, stroke: sw)
    let s = if label-override != none { label-override }
            else if v > 0 { "+" + str(v) }
            else { str(v) }
    let sz = if label-override != none { 8.5pt } else { 11pt }
    let st = if label-override != none { "italic" } else { "normal" }
    content((x + cell / 2, y - cell / 2),
      text(fill: tc, size: sz, weight: "bold", style: st)[#s])
  }

  // ── LEFT PANEL: Sobel Gx weight kernel ────────────────────────────────────
  let lx = 0.0
  let ly = 0.0

  let sobel = (
    (-1,  0,  1),
    (-2,  0,  2),
    (-1,  0,  1),
  )

  content((lx + N * cell / 2, ly + 0.62),
    text(fill: black90, size: 9pt)[Standard convolution])

  for r in range(N) {
    for c in range(N) {
      let x = lx + c * cell
      let y = ly - r * cell
      draw-cell(x, y, sobel.at(r).at(c), none)
    }
  }

  content((lx + N * cell / 2, ly - N * cell - 0.62),
    text(fill: black90, size: 9.5pt)[$op("out") = sum_i w_i dot p_i$])
  content((lx + N * cell / 2, ly - N * cell - 1.18),
    text(fill: black50, size: 8pt, style: "italic")[weights: fixed])

  // ── RIGHT PANEL: PIDINet difference grid ──────────────────────────────────
  let rx = lx + N * cell + gap
  let ry = 0.0

  let diffs = (
    (-18,  -5,   8),
    (-12,   0,  15),
    (-25,  -8,   5),
  )

  content((rx + N * cell / 2, ry + 0.62),
    text(fill: black90, size: 9pt)[PIDINet])

  for r in range(N) {
    for c in range(N) {
      let x = rx + c * cell
      let y = ry - r * cell
      let is-center = r == 1 and c == 1
      draw-cell(x, y, diffs.at(r).at(c), none)
    }
  }

  content((rx + N * cell / 2, ry - N * cell - 0.62),
    text(fill: black90, size: 9.5pt)[$op("out") = sum_i w_i dot Delta_i$])
  content((rx + N * cell / 2, ry - N * cell - 1.18),
    text(fill: black50, size: 8pt, style: "italic")[weights: learned])

})
