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

  let gs = 0.5   // grid cell size
  let gn = 8     // grid dimension
  let panel-gap = 3.0

  // Helper: draw an 8x8 grid
  let draw-grid(ox, oy) = {
    for r in range(gn) {
      for c in range(gn) {
        let x = ox + c * gs
        let y = oy - r * gs
        rect((x, y), (x + gs, y - gs), fill: white, stroke: 0.3pt + black30)
      }
    }
  }

  // Helper: fill a cell
  let fill-cell(ox, oy, r, c, col) = {
    let x = ox + c * gs
    let y = oy - r * gs
    rect((x, y), (x + gs, y - gs), fill: col.lighten(60%), stroke: 0.3pt + col)
  }

  // Helper: cell center
  let cc(ox, oy, r, c) = {
    (ox + c * gs + gs/2, oy - r * gs - gs/2)
  }

  // ========== LEFT PANEL: Fused Stencil ==========
  let lx = 0
  let ly = 0

  // Title
  content((lx + gn * gs / 2, ly + 0.6), text(fill: garnet, size: 9pt, weight: "bold")[Fused Stencil -- Scattered])

  draw-grid(lx, ly)

  // Irregular stencil footprint (elongated, scattered)
  let scattered-cells = (
    (1, 2), (1, 5),
    (2, 1), (2, 3), (2, 6),
    (3, 0), (3, 4), (3, 7),
    (4, 2), (4, 5),
    (5, 1), (5, 4), (5, 6),
    (6, 3), (6, 7),
    (7, 0), (7, 5),
  )

  for (r, c) in scattered-cells {
    fill-cell(lx, ly, r, c, garnet)
  }

  // Central accumulator
  let acc-r = 4
  let acc-c = 4
  fill-cell(lx, ly, acc-r, acc-c, rose)
  let acc-pos = cc(lx, ly, acc-r, acc-c)
  // Small "A" label
  content(acc-pos, text(fill: white, size: 7pt, weight: "bold")[A])

  // Crossing arrows from scattered cells to accumulator
  for (r, c) in scattered-cells {
    let from = cc(lx, ly, r, c)
    line(from, acc-pos, stroke: 0.4pt + garnet.lighten(20%), mark: (end: "stealth", fill: garnet.lighten(20%), scale: 0.2))
  }

  // ========== RIGHT PANEL: Geometric Kernel ==========
  let rx = lx + gn * gs + panel-gap
  let ry = 0

  // Title
  content((rx + gn * gs / 2, ry + 0.6), text(fill: atlantic, size: 9pt, weight: "bold")[Geometric Kernel -- Coalesced])

  draw-grid(rx, ry)

  // Rectangular stencil footprint (rows 2-5, cols 1-6)
  for r in range(2, 6) {
    for c in range(1, 7) {
      fill-cell(rx, ry, r, c, atlantic)
    }
  }

  // Accumulator
  let acc2-pos = cc(rx, ry, 4, 4)
  fill-cell(rx, ry, 4, 4, congaree)
  content(acc2-pos, text(fill: white, size: 7pt, weight: "bold")[A])

  // Neat parallel arrows: row by row, left to right
  for r in range(2, 6) {
    for c in range(1, 7) {
      if not (r == 4 and c == 4) {
        let from = cc(rx, ry, r, c)
        // Arrow pointing right to suggest sequential read
        let to-x = from.at(0) + gs * 0.35
        let to-y = from.at(1)
        line(from, (to-x, to-y), stroke: 0.4pt + atlantic.lighten(20%), mark: (end: "stealth", fill: atlantic.lighten(20%), scale: 0.2))
      }
    }
  }

  // Labels below
  let bot-y = ly - gn * gs - 0.5
  content(
    (lx + gn * gs / 2, bot-y),
    text(fill: black70, size: 8pt, style: "italic")[Cache-unfriendly: random strides],
  )
  content(
    (rx + gn * gs / 2, bot-y),
    text(fill: black70, size: 8pt, style: "italic")[Cache-friendly: sequential rows],
  )
})
