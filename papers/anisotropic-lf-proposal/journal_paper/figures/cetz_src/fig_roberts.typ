#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let cell-size = 0.8
#let black90 = rgb("#363636")

#let draw-kernel(ctx, ox, oy, kernel, label) = {
  import cetz.draw: *

  let rows = kernel.len()
  let cols = kernel.at(0).len()

  for r in range(rows) {
    for c in range(cols) {
      let v = kernel.at(r).at(c)
      let x = ox + c * cell-size
      let y = oy - r * cell-size

      let fill-color = if v > 0 { garnet } else if v < 0 { atlantic } else { white }
      let text-color = if v == 0 { black90 } else { white }
      let text-val = if v > 0 { "+" + str(v) } else { str(v) }

      rect(
        (x, y),
        (x + cell-size, y - cell-size),
        fill: fill-color,
        stroke: 0.5pt + black90,
      )
      content(
        (x + cell-size / 2, y - cell-size / 2),
        text(fill: text-color, weight: "bold", size: 11pt)[#text-val],
      )
    }
  }
  // Label below
  let cx = ox + cols * cell-size / 2
  let by = oy - rows * cell-size - 0.3
  content((cx, by), text(fill: black90, size: 10pt, style: "italic")[$#label$])
}

#cetz.canvas({
  import cetz.draw: *

  let k1 = ((1, 0), (0, -1))
  let k2 = ((0, 1), (-1, 0))

  draw-kernel((), 0, 0, k1, $G_1$)
  draw-kernel((), 2.6, 0, k2, $G_2$)

  // Title below both
  let cx = (0 + 2.6 + 2 * cell-size) / 2
  content((cx, -2.6), text(fill: black90, size: 11pt, weight: "bold")[Roberts Cross (1963)])
})
