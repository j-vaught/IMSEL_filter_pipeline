#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")

#let cell = 0.55

// Draw a 3x3 kernel grid at position (ox, oy=top-left), with label below
#let draw-kernel(ox, oy, kernel, label) = {
  import cetz.draw: *
  for r in range(3) {
    for c in range(3) {
      let v = kernel.at(r).at(c)
      let x = ox + c * cell
      let y = oy - r * cell

      let is-center = r == 1 and c == 1
      let fill-color = if is-center { white } else if v > 0 { garnet.lighten(40%) } else if v < 0 { atlantic.lighten(40%) } else { white }
      let text-color = if is-center or v == 0 { black90 } else { white }
      let text-val = if v > 0 { "+" + str(v) } else { str(v) }

      rect((x, y), (x + cell, y - cell), fill: fill-color, stroke: 0.2pt + black30)
      content(
        (x + cell / 2, y - cell / 2),
        text(fill: text-color, size: 8pt, weight: "bold")[#text-val],
      )
    }
  }
  // Label below
  content(
    (ox + 1.5 * cell, oy - 3 * cell - 0.25),
    text(fill: black90, size: 8.5pt, style: "italic")[#label],
  )
}

#cetz.canvas({
  import cetz.draw: *

  let gap-x = 0.6   // horizontal gap between grids
  let gap-y = 0.5   // vertical gap between rows
  let col-gap = 1.2 // extra horizontal gap between Sobel and Kirsch columns
  let grid-w = 3 * cell
  let col-step = grid-w + gap-x

  // Kernels
  let sobel-gx = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
  let sobel-gy = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))
  let k0   = ((-3, -3, 5), (-3, 0, 5), (-3, -3, 5))
  let k45  = ((-3, 5, 5), (-3, 0, 5), (-3, -3, -3))
  let k90  = ((5, 5, 5), (-3, 0, -3), (-3, -3, -3))
  let k135 = ((5, 5, -3), (5, 0, -3), (-3, -3, -3))

  // Layout: 3 columns × 2 rows
  // Col 1: Sobel (Gx top, Gy bottom)
  // Col 2: Kirsch (0° top, 90° bottom)
  // Col 3: Kirsch (45° top, 135° bottom)

  let row1-y = 0.0
  let row2-y = row1-y - 3 * cell - gap-y

  // Column x-positions
  let col1-x = 0.2                              // Sobel
  let col2-x = 0.2 + grid-w + col-gap           // Kirsch col 1
  let col3-x = col2-x + col-step                // Kirsch col 2

  // Column header labels
  content(
    (col1-x + 1.5 * cell, row1-y + 0.5),
    text(fill: black90, size: 9pt, weight: "bold")[Sobel],
  )
  content(
    (col2-x + col-step / 2 + 1.5 * cell, row1-y + 0.5),
    text(fill: black90, size: 9pt, weight: "bold")[Kirsch],
  )

  // Row 1
  draw-kernel(col1-x, row1-y, sobel-gx, $G_x$)
  draw-kernel(col2-x, row1-y, k0, [0°])
  draw-kernel(col3-x, row1-y, k45, [45°])

  // Row 2
  draw-kernel(col1-x, row2-y, sobel-gy, $G_y$)
  draw-kernel(col2-x, row2-y, k90, [90°])
  draw-kernel(col3-x, row2-y, k135, [135°])
})
