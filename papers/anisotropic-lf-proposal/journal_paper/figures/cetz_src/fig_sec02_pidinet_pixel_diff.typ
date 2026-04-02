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

  let cell = 1.6
  let N = 3

  // Pixel values
  let values = (
    (82, 95, 108),
    (88, 100, 115),
    (75, 92, 105),
  )

  let center-val = 100

  // Differences from center
  let diffs = (
    (-18, -5, +8),
    (-12, 0, +15),
    (-25, -8, +5),
  )

  let ox = 0.0
  let oy = 0.0

  // Draw 3x3 grid
  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell
      let v = values.at(r).at(c)

      let fc = if r == 1 and c == 1 { black10 } else { white }
      let sc = if r == 1 and c == 1 { 1.5pt + black90 } else { 0.8pt + black30 }

      rect((x, y), (x + cell, y - cell), fill: fc, stroke: sc)
      content((x + cell / 2, y - cell / 2), text(fill: black90, size: 11pt, weight: "bold")[#v])
    }
  }

  // Center pixel coordinates
  let center-x = ox + 1 * cell + cell / 2
  let center-y = oy - 1 * cell - cell / 2

  // Arrow data: (row, col, delta)
  let arrow-data = (
    (0, 0, -18),
    (0, 1, -5),
    (0, 2, +8),
    (1, 0, -12),
    (1, 2, +15),
    (2, 0, -25),
    (2, 1, -8),
    (2, 2, +5),
  )

  // Draw arrows from center to each neighbor
  for (r, c, delta) in arrow-data {
    let tx = ox + c * cell + cell / 2
    let ty = oy - r * cell - cell / 2

    let dx = tx - center-x
    let dy = ty - center-y
    let dist = calc.sqrt(dx * dx + dy * dy)
    let nx = dx / dist
    let ny = dy / dist

    let start-x = center-x + nx * 0.4
    let start-y = center-y + ny * 0.4
    let end-x = tx - nx * 0.4
    let end-y = ty - ny * 0.4

    let col = if delta > 0 { garnet } else { atlantic }

    line((start-x, start-y), (end-x, end-y), stroke: 1.2pt + col, mark: (end: ">", fill: col))
  }

  // Delta labels outside the grid perimeter
  // Top row neighbors: place labels above cells
  let delta-label-data = (
    // (row, col, delta, label-x-offset, label-y-offset)  relative to cell center
    (0, 0, -18, -0.15, 0.65),    // top-left: above-left
    (0, 1, -5, 0.0, 0.65),       // top-center: above
    (0, 2, +8, 0.15, 0.65),      // top-right: above-right
    (1, 0, -12, -0.9, 0.0),      // mid-left: further to the left
    (1, 2, +15, 0.9, 0.0),       // mid-right: further to the right
    (2, 0, -25, -0.15, -0.65),   // bot-left: below-left
    (2, 1, -8, 0.0, -0.65),      // bot-center: below
    (2, 2, +5, 0.15, -0.65),     // bot-right: below-right
  )

  for (r, c, delta, lxo, lyo) in delta-label-data {
    let tx = ox + c * cell + cell / 2 + lxo
    let ty = oy - r * cell - cell / 2 + lyo
    let col = if delta > 0 { garnet } else { atlantic }
    let delta-str = if delta > 0 { "+" + str(delta) } else { str(delta) }
    content((tx, ty), text(fill: col, size: 7.5pt, weight: "bold")[$Delta = #delta-str$])
  }

  // === Right side: weight table ===
  let wt-x = ox + N * cell + 1.5
  let wt-y = oy - 0.1

  content((wt-x + 1.0, wt-y), text(fill: black90, size: 9.5pt, weight: "bold")[Learned weights])

  let weight-labels = ($w_1$, $w_2$, $w_3$, $w_4$, $w_5$, $w_6$, $w_7$, $w_8$)
  let deltas = (-18, -5, +8, -12, +15, -25, -8, +5)

  for i in range(8) {
    let wy = wt-y - 0.55 - i * 0.48
    let delta = deltas.at(i)
    let col = if delta > 0 { garnet } else { atlantic }
    let delta-str = if delta > 0 { "+" + str(delta) } else { str(delta) }
    content((wt-x + 1.0, wy),
      text(fill: black90, size: 8.5pt)[#weight-labels.at(i) $dot.c$ #text(fill: col)[#delta-str]])
  }

  // Formula below grid
  let grid-cx = ox + N * cell / 2
  let formula-y = oy - N * cell - 1.2
  content((grid-cx + 1.8, formula-y), text(fill: black90, size: 10.5pt)[
    $"Output" = sum_i w_i dot Delta_i$
  ])

  // Sobel comparison
  let inset-y = formula-y - 0.65
  content((grid-cx + 1.8, inset-y), text(fill: black50, size: 9pt, style: "italic")[
    cf. Sobel: fixed weights $\{-1, 0, +1\}$
  ])

  // Title
  content((grid-cx + 1.8, oy + 1.3), text(fill: black90, size: 11pt, weight: "bold")[PIDINet (Su et al., 2021)])
})
