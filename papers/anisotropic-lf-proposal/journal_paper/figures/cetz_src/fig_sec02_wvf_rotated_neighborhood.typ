#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let garnet-light = garnet.lighten(87%)

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

  // Highlight circular neighborhood
  let radius = 2.8
  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let dist = calc.sqrt(dc * dc + dr * dr)
      if dist <= radius {
        let x = c * cell
        let y = (grid-n - 1 - r) * cell
        rect(
          (x, y),
          (x + cell, y + cell),
          fill: garnet-light.transparentize(25%),
          stroke: 0.2pt + black30,
        )
      }
    }
  }

  let cx = half * cell + cell / 2
  let cy = (grid-n - 1 - half) * cell + cell / 2

  // Circular neighborhood outline
  circle(
    (cx, cy),
    radius: radius * cell,
    fill: none,
    stroke: 1.4pt + garnet,
  )

  // Center dot (white with dark border for visibility)
  circle((cx, cy), radius: 0.07, fill: white, stroke: 0.6pt + black90)

  // Normal direction: perpendicular to edge (edge at -50°, normal at -50° + 90° = 40°)
  let normal-angle = 45.0 * calc.pi / 180.0
  let tangent-angle = normal-angle + calc.pi / 2

  // Draw rotated coordinate axes (horseshoe green)
  let axis-len = 1.3

  // x'-axis (along edge normal)
  let nx = axis-len * calc.cos(normal-angle)
  let ny = axis-len * calc.sin(normal-angle)
  line(
    (cx - nx * 0.35, cy - ny * 0.35),
    (cx + nx, cy + ny),
    stroke: 1.5pt + horseshoe,
    mark: (end: "stealth", fill: horseshoe, size: 0.16),
  )

  // y'-axis (along edge tangent)
  let tax = axis-len * calc.cos(tangent-angle)
  let tay = axis-len * calc.sin(tangent-angle)
  line(
    (cx - tax * 0.35, cy - tay * 0.35),
    (cx + tax, cy + tay),
    stroke: 1.5pt + rose,
    mark: (end: "stealth", fill: rose, size: 0.16),
  )

  // Axis labels
  content(
    (cx + nx * 1.1 + 0.2, cy + ny * 1.1),
    text(fill: horseshoe, size: 7.5pt)[$x'$],
  )
  content(
    (cx + tax * 1.1 - 0.05, cy + tay * 1.1 - 0.2),
    text(fill: rose, size: 7.5pt)[$y'$],
  )

  // Bold normal derivative arrow in Rose -- SEPARATE from the axis
  // Drawn longer and offset to the other side so it doesn't overlap x' axis
  let nd-len = 2.0
  let ndx = nd-len * calc.cos(normal-angle)
  let ndy = nd-len * calc.sin(normal-angle)
  // Offset along tangent direction to separate from x' axis
  let off = 0.25
  let offx = off * calc.cos(tangent-angle)
  let offy = off * calc.sin(tangent-angle)
  line(
    (cx + offx, cy + offy),
    (cx + ndx + offx, cy + ndy + offy),
    stroke: 2.8pt + black90,
    mark: (end: "stealth", fill: black90, size: 0.28),
  )

  // df/dx' label
  content(
    (cx + ndx + offx - 0.1, cy + ndy + offy + 0.3),
    text(fill: black90, size: 8.5pt, weight: "bold")[$partial f \/ partial x'$],
  )

  // (title and annotation removed — goes in caption)
})
