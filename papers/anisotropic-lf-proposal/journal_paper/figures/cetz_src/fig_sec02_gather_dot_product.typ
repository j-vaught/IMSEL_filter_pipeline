#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  let cell = 0.45
  let N = 7
  let panel-gap = 3.0

  // Draw a pixel grid
  let draw-grid(ox, oy) = {
    for r in range(N) {
      for c in range(N) {
        let x = ox + c * cell
        let y = oy - r * cell
        rect((x, y), (x + cell, y - cell), fill: white, stroke: 0.3pt + black30)
      }
    }
    rect((ox, oy), (ox + N * cell, oy - N * cell), stroke: 0.6pt + black90)
  }

  // Highlight specific cells
  let highlight-cells(ox, oy, cells, col) = {
    for (r, c) in cells {
      let x = ox + c * cell
      let y = oy - r * cell
      rect((x, y), (x + cell, y - cell), fill: col, stroke: 0.3pt + black90)
    }
  }

  // ========== LEFT PANEL: Regular Conv ==========
  let lx = 0.0
  let ly = 0.0

  // Panel label
  content((lx + N * cell / 2, ly + 0.8), text(fill: atlantic, size: 10pt, weight: "bold")[Regular conv #sym.dash.em cuDNN])

  draw-grid(lx, ly)

  // Highlight 3x3 region centered at (3,3)
  let reg-cells = ()
  for r in range(2, 5) {
    for c in range(2, 5) {
      reg-cells.push((r, c))
    }
  }
  highlight-cells(lx, ly, reg-cells, atlantic.lighten(60%))

  // Column vector position
  let vec-x = lx + N * cell + 0.8
  let vec-y = ly - 0.5
  let vec-h = 0.35

  // Arrows from highlighted cells to column vector
  for (i, (r, c)) in reg-cells.enumerate() {
    let sx = lx + c * cell + cell / 2
    let sy = ly - r * cell - cell / 2
    let ey = vec-y - i * vec-h - vec-h / 2
    line((sx, sy), (vec-x - 0.05, ey), stroke: 0.35pt + atlantic)
  }

  // Draw gathered values column
  for i in range(9) {
    let vy = vec-y - i * vec-h
    rect((vec-x, vy), (vec-x + 0.5, vy - vec-h), fill: atlantic.lighten(60%), stroke: 0.5pt + atlantic)
    content((vec-x + 0.25, vy - vec-h / 2), text(fill: black90, size: 5.5pt)[$g_#(i+1)$])
  }
  content((vec-x + 0.25, vec-y + 0.25), text(fill: black90, size: 7pt, weight: "bold")[g])

  // Element-wise multiply
  let mul-x = vec-x + 0.65
  let mid-y = vec-y - 4.5 * vec-h
  content((mul-x + 0.15, mid-y), text(fill: black90, size: 12pt)[$dot.o$])

  // Weight vector
  let wt-x = mul-x + 0.45
  for i in range(9) {
    let vy = vec-y - i * vec-h
    rect((wt-x, vy), (wt-x + 0.5, vy - vec-h), fill: garnet.lighten(70%), stroke: 0.5pt + garnet)
    content((wt-x + 0.25, vy - vec-h / 2), text(fill: black90, size: 5.5pt)[$w_#(i+1)$])
  }
  content((wt-x + 0.25, vec-y + 0.25), text(fill: black90, size: 7pt, weight: "bold")[w])

  // Sum arrow and output
  let sum-x = wt-x + 0.65
  line((sum-x, mid-y), (sum-x + 0.5, mid-y), stroke: 1pt + black90, mark: (end: ">", fill: black90))
  content((sum-x + 0.25, mid-y + 0.3), text(fill: black90, size: 7pt)[$Sigma$])

  let out-x = sum-x + 0.65
  rect((out-x, mid-y + 0.3), (out-x + 0.5, mid-y - 0.3), fill: black10, stroke: 1pt + black90)
  content((out-x + 0.25, mid-y), text(fill: black90, size: 9pt, weight: "bold")[$R$])

  // Annotation
  content((lx + N * cell / 2, ly - N * cell - 0.5), text(fill: black50, size: 7.5pt, style: "italic")[Regular grid = regular memory access])

  // ========== RIGHT PANEL: Stencil / ASF ==========
  let left-panel-end = out-x + 0.5
  let rx = left-panel-end + panel-gap
  let ry = 0.0

  content((rx + N * cell / 2, ry + 0.8), text(fill: garnet, size: 10pt, weight: "bold")[Stencil #sym.dash.em ASF])

  draw-grid(rx, ry)

  // Irregular stencil pattern
  let stencil-cells = (
    (0, 3),
    (1, 1), (1, 3), (1, 5),
    (2, 2), (2, 4),
    (3, 0), (3, 3), (3, 6),
    (4, 2), (4, 4),
    (5, 3),
  )
  highlight-cells(rx, ry, stencil-cells, garnet.lighten(55%))

  let n-stencil = stencil-cells.len()

  // Column vector
  let rvec-x = rx + N * cell + 0.8
  let rvec-y = ry - 0.15
  let rvec-h = 0.3

  // Arrows from stencil cells to column vector
  for (i, (r, c)) in stencil-cells.enumerate() {
    let sx = rx + c * cell + cell / 2
    let sy = ry - r * cell - cell / 2
    let ey = rvec-y - i * rvec-h - rvec-h / 2
    line((sx, sy), (rvec-x - 0.05, ey), stroke: 0.35pt + garnet)
  }

  // Draw gathered values
  for i in range(n-stencil) {
    let vy = rvec-y - i * rvec-h
    rect((rvec-x, vy), (rvec-x + 0.5, vy - rvec-h), fill: garnet.lighten(55%), stroke: 0.5pt + garnet)
    content((rvec-x + 0.25, vy - rvec-h / 2), text(fill: black90, size: 5pt)[$g_#(i+1)$])
  }
  content((rvec-x + 0.25, rvec-y + 0.25), text(fill: black90, size: 7pt, weight: "bold")[g])

  // Element-wise multiply
  let rmul-x = rvec-x + 0.65
  let rmid = rvec-y - n-stencil / 2 * rvec-h
  content((rmul-x + 0.15, rmid), text(fill: black90, size: 12pt)[$dot.o$])

  // Weight vector
  let rwt-x = rmul-x + 0.45
  for i in range(n-stencil) {
    let vy = rvec-y - i * rvec-h
    rect((rwt-x, vy), (rwt-x + 0.55, vy - rvec-h), fill: atlantic.lighten(60%), stroke: 0.5pt + atlantic)
    content((rwt-x + 0.275, vy - rvec-h / 2), text(fill: black90, size: 5pt)[$alpha_#(i+1)$])
  }
  content((rwt-x + 0.275, rvec-y + 0.25), text(fill: black90, size: 7pt, weight: "bold")[$bold(alpha)$])

  // Sum arrow and output
  let rsum-x = rwt-x + 0.7
  line((rsum-x, rmid), (rsum-x + 0.5, rmid), stroke: 1pt + black90, mark: (end: ">", fill: black90))
  content((rsum-x + 0.25, rmid + 0.3), text(fill: black90, size: 7pt)[$Sigma$])

  let rout-x = rsum-x + 0.65
  rect((rout-x, rmid + 0.3), (rout-x + 0.5, rmid - 0.3), fill: black10, stroke: 1pt + black90)
  content((rout-x + 0.3, rmid), text(fill: black90, size: 9pt, weight: "bold")[$R$])

  // Annotation
  content((rx + N * cell / 2, ry - N * cell - 0.5), text(fill: black50, size: 7.5pt, style: "italic")[Irregular pattern = same operation type])

  // ========== Bottom caption ==========
  let right-end = rout-x + 0.5
  let total-w = right-end
  let cap-y = calc.min(ly - N * cell, ry - N * cell) - 1.0
  content((total-w / 2, cap-y), text(fill: black90, size: 10pt)[
    Both reduce to: $R = bold(alpha) dot bold(g)$ #h(0.5em) (gather-dot-product)
  ])

  // Title
  content((total-w / 2, ly + 1.5), text(fill: black90, size: 11pt, weight: "bold")[Gather-Dot-Product Abstraction])
})
