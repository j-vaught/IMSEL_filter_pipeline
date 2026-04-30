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
#let honeycomb = rgb("#A49137")

#cetz.canvas({
  import cetz.draw: *

  let grid-size = 11
  let cell = 0.4
  // Offset the grid so there is room for labels outside
  let gx = 1.2
  let gy = -0.5
  let grid-w = grid-size * cell

  // Center of grid
  let cx = gx + grid-w / 2
  let cy = gy - grid-w / 2

  // === PIXEL GRID with diagonal edge ===
  for r in range(grid-size) {
    for c in range(grid-size) {
      let x = gx + c * cell
      let y = gy - r * cell

      let dist = (c + r) - (grid-size - 1)
      let fill-color = if dist < -2 {
        rgb("#4a4a4a")
      } else if dist < -1 {
        rgb("#6a6a6a")
      } else if dist < 0 {
        rgb("#8a8a8a")
      } else if dist < 1 {
        rgb("#aaaaaa")
      } else if dist < 2 {
        rgb("#cccccc")
      } else {
        rgb("#e8e8e8")
      }

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fill-color,
        stroke: 0.15pt + white.transparentize(50%),
      )
    }
  }

  // === 8 ARROWS radiating from center ===
  let arrow-len = 1.5

  // Response magnitudes for each direction
  // Edge normal for this diagonal is ~135 deg
  let responses = (0.2, 0.35, 0.25, 0.95, 0.15, 0.3, 0.2, 0.85)
  let winner = 3  // 135 degrees

  for k in range(8) {
    let angle-deg = k * 45
    let angle-rad = angle-deg * calc.pi / 180
    let resp = responses.at(k)

    let is-winner = k == winner

    let tip-x = cx + arrow-len * calc.cos(angle-rad)
    let tip-y = cy + arrow-len * calc.sin(angle-rad)

    let arrow-color = if is-winner { garnet } else { black50 }
    let arrow-width = if is-winner { 2pt } else { 0.8pt }

    // Arrow
    line(
      (cx, cy),
      (tip-x, tip-y),
      stroke: arrow-width + arrow-color,
      mark: (end: "stealth", fill: arrow-color, scale: 0.45),
    )

    // Response bar at the tip, perpendicular to the arrow
    let bar-height = resp * 0.6
    let bar-color = if is-winner { garnet } else { black30 }
    let bar-thickness = if is-winner { 3.5pt } else { 2pt }

    let perp-rad = angle-rad + calc.pi / 2
    let bar-cx = tip-x + 0.2 * calc.cos(angle-rad)
    let bar-cy = tip-y + 0.2 * calc.sin(angle-rad)

    let b1x = bar-cx - bar-height / 2 * calc.cos(perp-rad)
    let b1y = bar-cy - bar-height / 2 * calc.sin(perp-rad)
    let b2x = bar-cx + bar-height / 2 * calc.cos(perp-rad)
    let b2y = bar-cy + bar-height / 2 * calc.sin(perp-rad)

    line(
      (b1x, b1y),
      (b2x, b2y),
      stroke: bar-thickness + bar-color,
    )
  }

  // Center dot
  circle((cx, cy), radius: 0.06, fill: white, stroke: 0.5pt + black90)

  // === ANGLE LABELS fully outside the grid ===
  // Compute distance from center to grid edge for each direction, then place beyond
  let half-w = grid-w / 2
  for k in range(8) {
    let angle-deg = k * 45
    let angle-rad = angle-deg * calc.pi / 180
    let is-winner = k == winner

    // Distance from center to grid edge along this angle
    let cos-a = calc.cos(angle-rad)
    let sin-a = calc.sin(angle-rad)
    // Grid extends half-w in each direction from center
    let dx = if calc.abs(cos-a) > 0.01 { half-w / calc.abs(cos-a) } else { 999.0 }
    let dy = if calc.abs(sin-a) > 0.01 { half-w / calc.abs(sin-a) } else { 999.0 }
    let edge-dist = calc.min(dx, dy)
    let label-dist = edge-dist + 0.4

    let lx = cx + label-dist * cos-a
    let ly = cy + label-dist * sin-a
    let label-color = if is-winner { garnet } else { black50 }
    content((lx, ly), text(fill: label-color, size: 7.5pt, weight: if is-winner { "bold" } else { "regular" })[#str(angle-deg)#sym.degree])
  }

  // === Dark/light labels inside grid corners ===
  content((gx + 0.8, gy - 0.5), text(fill: white, size: 7pt, weight: "bold")[dark])
  content((gx + grid-w - 0.5, gy - grid-w + 0.45), text(fill: black90, size: 7pt, weight: "bold")[light])

  // === ANNOTATION below the grid ===
  let ann-x = gx + grid-w * 0.7
  let ann-y = gy - grid-w - 1.2
  content((ann-x, ann-y), text(fill: garnet, size: 10pt, weight: "bold")[
    $k^* = arg max |R_k|$
  ])

  // === TITLE ===
  content((cx + 1.0, gy + 0.8), text(fill: black90, size: 11pt, weight: "bold")[
    Discrete Orientation Sweep (This Work)
  ])
})
