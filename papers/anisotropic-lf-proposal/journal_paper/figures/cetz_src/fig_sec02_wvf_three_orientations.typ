#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let garnet-light = garnet.lighten(87%)

#let cell = 0.32
#let grid-n = 15
#let half = 7
#let edge-angle = 30.0
#let edge-rad = edge-angle * calc.pi / 180.0
#let radius = 2.8
#let grid-w = grid-n * cell

// All panels use same orientation
#let normal-deg = 45.0
#let normal-angle = normal-deg * calc.pi / 180.0
#let tangent-angle = normal-angle + calc.pi / 2

#let labels = ("(a) Wide View Filter", "(b) Virt. Pixels", "(c) Line Filter")
#let spacing = grid-w + 0.8

// Virtual point parameters for panels (b) and (c)
#let n-virt = 9  // 2m+1 = 9 virtual points (m=4)
#let virt-spacing = 1.0  // spacing along tangent in cell units (1 cell = 1 pixel)

#cetz.canvas({
  import cetz.draw: *

  for panel in range(3) {
    let ox = float(panel) * spacing

    // Draw grid with diagonal edge
    for r in range(grid-n) {
      for c in range(grid-n) {
        let x = ox + c * cell
        let y = (grid-n - 1 - r) * cell
        let dc = c - half
        let dr = r - half
        let dist-to-edge = dc * calc.cos(edge-rad) - dr * calc.sin(edge-rad)
        let intensity = 0.5 + 0.45 * calc.tanh(dist-to-edge * 0.8)
        let fill-color = black90.lighten(100% * intensity)
        rect(
          (x, y),
          (x + cell, y + cell),
          fill: fill-color,
          stroke: 0.2pt + black30,
        )
      }
    }

    let cx = ox + half * cell + cell / 2
    let cy = (grid-n - 1 - half) * cell + cell / 2

    // Tangent and normal unit vectors
    let tnx = calc.cos(tangent-angle)
    let tny = calc.sin(tangent-angle)
    let nnx = calc.cos(normal-angle)
    let nny = calc.sin(normal-angle)

    if panel == 0 {
      // ========================================
      // (a) Wide View Filter: single circular neighborhood
      // ========================================

      // Highlight circular neighborhood
      for r in range(grid-n) {
        for c in range(grid-n) {
          let dc = c - half
          let dr = r - half
          let dist = calc.sqrt(dc * dc + dr * dr)
          if dist <= radius {
            let x = ox + c * cell
            let y = (grid-n - 1 - r) * cell
            rect(
              (x, y),
              (x + cell, y + cell),
              fill: atlantic.lighten(75%),
              stroke: 0.2pt + black30,
            )
          }
        }
      }

      // Circle outline
      circle((cx, cy), radius: radius * cell, fill: none, stroke: 1.2pt + garnet)

    } else if panel == 1 {
      // ========================================
      // (b) Virtual Pixels: show evaluation points along tangent
      // ========================================

      // Draw virtual evaluation points along tangent, snapped to nearest pixel center
      // Walk along tangent direction, round to nearest grid cell at each step
      let m-half = calc.floor(n-virt / 2)  // = 4

      // Precompute snapped positions
      let virt-positions = ()
      for j in range(n-virt) {
        let offset = float(j - m-half)
        // Continuous position in grid-column/row space
        let col-cont = float(half) + offset * tnx
        let row-cont = float(half) - offset * tny  // row increases downward, tny is in canvas (upward)
        // Snap to nearest integer
        let col-snap = calc.round(col-cont)
        let row-snap = calc.round(row-cont)
        // Convert to canvas coords (pixel center)
        let vx = ox + float(col-snap) * cell + cell / 2
        let vy = float(grid-n - 1 - row-snap) * cell + cell / 2
        virt-positions.push((vx, vy))

        // Draw dot
        let dot-r = if j == m-half { 0.1 } else { 0.07 }
        let dot-clr = if j == m-half { garnet } else { rose }
        circle((vx, vy), radius: dot-r, fill: dot-clr, stroke: 0.5pt + black70)
      }

      // Dashed line through first and last virtual point
      let first = virt-positions.at(0)
      let last = virt-positions.at(n-virt - 1)
      line(
        first,
        last,
        stroke: (dash: "dashed", paint: rose, thickness: 0.8pt),
      )

    } else {
      // ========================================
      // (c) Line Filter: overlapping circular neighborhoods at each virtual point
      // ========================================

      // Draw overlapping neighborhoods: outermost first, center last (on top)
      let m-half = calc.floor(n-virt / 2)
      // Build draw order: distance from center, descending
      let draw-order = ()
      for dist in range(m-half, -1, step: -1) {
        // Draw both sides at this distance (except center which is dist=0, only one)
        if dist > 0 {
          draw-order.push(m-half - dist)
          draw-order.push(m-half + dist)
        } else {
          draw-order.push(m-half)
        }
      }
      for j in draw-order {
        let offset = float(j - m-half)
        let col-cont = float(half) + offset * tnx
        let row-cont = float(half) - offset * tny
        let col-snap = calc.round(col-cont)
        let row-snap = calc.round(row-cont)
        let vx = ox + float(col-snap) * cell + cell / 2
        let vy = float(grid-n - 1 - row-snap) * cell + cell / 2

        // Highlight pixels in this neighborhood — uniform atlantic blue for all
        let highlight-clr = atlantic.lighten(75%)

        for r in range(grid-n) {
          for c in range(grid-n) {
            let px = ox + c * cell + cell / 2
            let py = (grid-n - 1 - r) * cell + cell / 2
            let dx = px - vx
            let dy = py - vy
            let dist = calc.sqrt(dx * dx + dy * dy)
            if dist <= radius * cell {
              let x = ox + c * cell
              let y = (grid-n - 1 - r) * cell
              rect(
                (x, y),
                (x + cell, y + cell),
                fill: highlight-clr,
                stroke: 0.2pt + black30,
              )
            }
          }
        }

        // Circle outline at this virtual position
        let stroke-w = if j == m-half { 1.2pt } else { 0.6pt }
        let stroke-clr = if j == m-half { garnet } else { garnet.lighten(40%) }
        circle((vx, vy), radius: radius * cell, fill: none, stroke: stroke-w + stroke-clr)

        // Dot at center of each virtual neighborhood
        let dot-r = if j == m-half { 0.08 } else { 0.05 }
        circle((vx, vy), radius: dot-r, fill: garnet, stroke: none)
      }
    }

    // ========================================
    // Common: center dot, axes (only on panel a), derivative arrow
    // ========================================
    circle((cx, cy), radius: 0.06, fill: white, stroke: 0.5pt + black90)

    let axis-len = 1.1
    let nx = axis-len * nnx
    let ny = axis-len * nny
    let tax = axis-len * tnx
    let tay = axis-len * tny

    if panel == 0 {
      // x'-axis (normal) — horseshoe
      line(
        (cx - nx * 0.35, cy - ny * 0.35),
        (cx + nx, cy + ny),
        stroke: 1.3pt + horseshoe,
        mark: (end: "stealth", fill: horseshoe, size: 0.14),
      )

      // y'-axis (tangent) — rose
      line(
        (cx - tax * 0.35, cy - tay * 0.35),
        (cx + tax, cy + tay),
      stroke: 1.3pt + rose,
      mark: (end: "stealth", fill: rose, size: 0.14),
    )

      // Axis labels
      content(
        (cx + nx * 1.15 + 0.15, cy + ny * 1.15),
        text(fill: horseshoe, size: 7pt)[$x'$],
      )
      content(
        (cx + tax * 1.15 - 0.05, cy + tay * 1.15 - 0.15),
        text(fill: rose, size: 7pt)[$y'$],
      )
    }

    // df/dx' arrow — black (shown on all panels)
    let nd-len = 1.7
    let ndx = nd-len * nnx
    let ndy = nd-len * nny
    let off = 0.2
    let offx = off * tnx
    let offy = off * tny
    line(
      (cx + offx, cy + offy),
      (cx + ndx + offx, cy + ndy + offy),
      stroke: 2.4pt + black90,
      mark: (end: "stealth", fill: black90, size: 0.24),
    )
    content(
      (cx + ndx + offx - 0.05, cy + ndy + offy + 0.25),
      text(fill: black90, size: 7.5pt, weight: "bold")[$partial f \/ partial x'$],
    )

    // Panel label below
    content(
      (ox + grid-w / 2, -0.5),
      text(fill: black90, size: 10pt, weight: "bold")[#labels.at(panel)],
    )
  }
})
