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

  let cell = 0.38
  let grid-n = 15
  let half = 7

  let edge-angle = 30.0
  let edge-rad = edge-angle * calc.pi / 180.0

  // Draw 15x15 grid with diagonal edge
  for r in range(grid-n) {
    for c in range(grid-n) {
      let x = c * cell
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

  let cx = half * cell + cell / 2
  let cy = (grid-n - 1 - half) * cell + cell / 2

  // Edge tangent and normal directions
  let normal-angle = 120.0 * calc.pi / 180.0
  let tangent-angle = normal-angle + calc.pi / 2

  // 5 evaluation points along tangent, tight spacing to stay within grid
  let n-eval = 5
  let spacing = 0.75
  let radius = 1.6  // small enough so outermost circles don't go far outside grid

  // Garnet shades for neighborhoods
  let garnet-shades = (
    garnet.lighten(93%),
    garnet.lighten(88%),
    garnet.lighten(82%),
    garnet.lighten(88%),
    garnet.lighten(93%),
  )

  let stroke-shades = (
    garnet.lighten(60%),
    garnet.lighten(40%),
    garnet,
    garnet.lighten(40%),
    garnet.lighten(60%),
  )

  let stroke-widths = (0.6, 0.8, 1.4, 0.8, 0.6)

  // Compute evaluation point positions
  let eval-pts = ()
  for j in range(n-eval) {
    let offset = (j - 2) * spacing
    let px = cx + offset * calc.cos(tangent-angle)
    let py = cy + offset * calc.sin(tangent-angle)
    eval-pts.push((px, py))
  }

  // Draw overlapping neighborhoods: outermost first, center last
  let draw-order = (0, 4, 1, 3, 2)

  for idx in draw-order {
    let (px, py) = eval-pts.at(idx)

    // Highlight pixels in circular neighborhood
    for r in range(grid-n) {
      for c in range(grid-n) {
        let cell-cx = c * cell + cell / 2
        let cell-cy = (grid-n - 1 - r) * cell + cell / 2
        let dc = cell-cx - px
        let dr = cell-cy - py
        let dist = calc.sqrt(dc * dc + dr * dr)
        if dist <= radius * cell {
          let x = c * cell
          let y = (grid-n - 1 - r) * cell
          rect(
            (x, y),
            (x + cell, y + cell),
            fill: garnet-shades.at(idx).transparentize(10%),
            stroke: 0.2pt + black30,
          )
        }
      }
    }

    // Circle outline
    circle(
      (px, py),
      radius: radius * cell,
      fill: none,
      stroke: stroke-widths.at(idx) * 1pt + stroke-shades.at(idx),
    )

    // Evaluation point dot
    circle(
      (px, py),
      radius: if idx == 2 { 0.07 } else { 0.04 },
      fill: if idx == 2 { garnet } else { garnet.lighten(30%) },
      stroke: none,
    )
  }

  // Dashed tangent line connecting evaluation points
  let ext = 0.4
  let p-start-x = cx + (-2 * spacing - ext) * calc.cos(tangent-angle)
  let p-start-y = cy + (-2 * spacing - ext) * calc.sin(tangent-angle)
  let p-end-x = cx + (2 * spacing + ext) * calc.cos(tangent-angle)
  let p-end-y = cy + (2 * spacing + ext) * calc.sin(tangent-angle)

  line(
    (p-start-x, p-start-y),
    (p-end-x, p-end-y),
    stroke: (dash: "dashed", paint: atlantic, thickness: 1pt),
  )

  // Small Gaussian bell curve to the right of the grid
  // Oriented vertically (matching tangent line direction projected onto vertical)
  let grid-w = grid-n * cell
  let bell-cx = grid-w + 0.8  // baseline x position
  let bell-cy = cy             // centered at same y as grid center

  // The evaluation points span from y = cy + 2*spacing*sin(tangent) to cy - 2*spacing*sin(tangent)
  // tangent_angle ~ 210 deg, sin(210) ~ -0.5
  // So points go from upper-right to lower-left on the grid
  // For the bell curve sidebar, map j=-2..+2 to vertical positions
  let bell-spread = 2.0 * spacing  // half-height of bell in y
  let bell-h = 0.9                  // max horizontal extent

  // Draw baseline (vertical line)
  line(
    (bell-cx, bell-cy - bell-spread - 0.3),
    (bell-cx, bell-cy + bell-spread + 0.3),
    stroke: 0.4pt + black30,
  )

  // Gaussian bell curve opening to the right
  let n-bell = 50
  let bell-curve-pts = ()
  for i in range(n-bell + 1) {
    let t = -1.0 + 2.0 * i / n-bell  // -1 to +1
    let y-pos = bell-cy + t * bell-spread
    let gauss = bell-h * calc.exp(-4.0 * t * t)
    bell-curve-pts.push((bell-cx + gauss, y-pos))
  }

  // Fill
  let fill-closed = ((bell-cx, bell-cy - bell-spread),)
  fill-closed += bell-curve-pts
  fill-closed.push((bell-cx, bell-cy + bell-spread))
  line(..fill-closed, close: true, fill: congaree.lighten(90%), stroke: none)

  // Outline
  for i in range(bell-curve-pts.len() - 1) {
    line(bell-curve-pts.at(i), bell-curve-pts.at(i + 1), stroke: 1.2pt + congaree)
  }

  // Dots at evaluation positions on bell curve
  for j in range(n-eval) {
    let t = (j - 2) / 2.0
    let gauss = bell-h * calc.exp(-4.0 * t * t)
    let y-pos = bell-cy + t * bell-spread
    circle((bell-cx + gauss, y-pos), radius: 0.04, fill: congaree, stroke: none)
  }

  // w_j label
  content(
    (bell-cx + bell-h + 0.2, bell-cy),
    text(fill: congaree, size: 8pt)[$w_j$],
  )

  // Derivative arrow at center pointing along normal
  let nd-len = 1.5
  let ndx = nd-len * calc.cos(normal-angle)
  let ndy = nd-len * calc.sin(normal-angle)
  line(
    (cx, cy),
    (cx + ndx, cy + ndy),
    stroke: 2.5pt + rose,
    mark: (end: "stealth", fill: rose, size: 0.26),
  )
  content(
    (cx + ndx + 0.15, cy + ndy + 0.25),
    text(fill: rose, size: 8pt, weight: "bold")[$hat(f)'$],
  )

  // Annotation
  content(
    (grid-w / 2, -0.5),
    text(fill: black90, size: 8pt, style: "italic")[$(2m + 1)$ overlapping evaluations],
  )

  // Title
  content(
    (grid-w / 2, -1.15),
    text(fill: black90, size: 11pt, weight: "bold")[Line Filter (Bagan & Wang, 2023)],
  )
})
