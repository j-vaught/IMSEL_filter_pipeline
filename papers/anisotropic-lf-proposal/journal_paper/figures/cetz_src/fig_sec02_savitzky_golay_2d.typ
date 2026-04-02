#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  let cell = 0.5
  let grid-n = 9

  // Draw 9x9 grid
  for r in range(grid-n) {
    for c in range(grid-n) {
      let x = c * cell
      let y = (grid-n - 1 - r) * cell

      // 5x5 window: rows 2..6, cols 2..6
      let in-window = r >= 2 and r <= 6 and c >= 2 and c <= 6

      let fill-color = if in-window {
        // Gradient from light (top-left) to darker (bottom-right) to suggest fitted 2D surface
        let intensity = (r - 2 + c - 2) / 8.0
        garnet.lighten(92% - 40% * intensity)
      } else {
        black10
      }

      rect(
        (x, y),
        (x + cell, y + cell),
        fill: fill-color,
        stroke: 0.3pt + black50,
      )
    }
  }

  // Window outline
  rect(
    (2 * cell, (grid-n - 1 - 6) * cell),
    (7 * cell, (grid-n - 1 - 2) * cell + cell),
    fill: none,
    stroke: 1.4pt + garnet,
  )

  // Center pixel
  let cx = 4 * cell + cell / 2
  let cy = (grid-n - 1 - 4) * cell + cell / 2
  circle((cx, cy), radius: 0.07, fill: black90, stroke: none)

  // df/dx arrow (horizontal)
  let arr-len = 1.1
  line(
    (cx + 0.12, cy),
    (cx + arr-len, cy),
    stroke: 2pt + atlantic,
    mark: (end: "stealth", fill: atlantic, size: 0.22),
  )
  content(
    (cx + arr-len + 0.1, cy - 0.02),
    anchor: "west",
    text(fill: atlantic, size: 8.5pt)[$partial f \/ partial x$],
  )

  // df/dy arrow (vertical)
  line(
    (cx, cy + 0.12),
    (cx, cy + arr-len),
    stroke: 2pt + atlantic,
    mark: (end: "stealth", fill: atlantic, size: 0.22),
  )
  content(
    (cx + 0.12, cy + arr-len + 0.15),
    anchor: "west",
    text(fill: atlantic, size: 8.5pt)[$partial f \/ partial y$],
  )

  // Annotation
  let grid-w = grid-n * cell
  content(
    (grid-w / 2, -0.5),
    text(fill: black90, size: 8pt, style: "italic")[rectangular support, axis-aligned],
  )

  // Title
  content(
    (grid-w / 2, -1.15),
    text(fill: black90, size: 11pt, weight: "bold")[2D Savitzky--Golay (Meer et al., 1991)],
  )
})
