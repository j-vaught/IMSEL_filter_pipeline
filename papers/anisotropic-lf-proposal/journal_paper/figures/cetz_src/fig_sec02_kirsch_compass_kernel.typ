#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black30 = rgb("#C7C7C7")
#let cell-size = 0.7
#let black90 = rgb("#363636")

#let draw-kirsch-kernel(ctx, ox, oy, kernel, label) = {
  import cetz.draw: *

  let rows = kernel.len()
  let cols = kernel.at(0).len()

  for r in range(rows) {
    for c in range(cols) {
      let v = kernel.at(r).at(c)
      let x = ox + c * cell-size
      let y = oy - r * cell-size

      let is-center = r == 1 and c == 1
      let fill-color = if is-center { white } else if v > 0 { garnet.lighten(40%) } else if v < 0 { atlantic.lighten(40%) } else { garnet.lighten(40%) }
      let text-color = if is-center or v == 0 { black90 } else { white }
      let text-val = if v > 0 { "+" + str(v) } else { str(v) }

      rect(
        (x, y),
        (x + cell-size, y - cell-size),
        fill: fill-color,
        stroke: 0.2pt + black30,
      )
      content(
        (x + cell-size / 2, y - cell-size / 2),
        text(fill: text-color, weight: "bold", size: 9pt)[#text-val],
      )
    }
  }
  let cx = ox + cols * cell-size / 2
  let by = oy - rows * cell-size - 0.25
  content((cx, by), text(fill: black90, size: 9pt, style: "italic")[#label])
}

#cetz.canvas({
  import cetz.draw: *

  // 0 degrees - East
  let k0 = ((-3, -3, 5), (-3, 0, 5), (-3, -3, 5))
  // 45 degrees - NE
  let k45 = ((-3, 5, 5), (-3, 0, 5), (-3, -3, -3))
  // 90 degrees - North
  let k90 = ((5, 5, 5), (-3, 0, -3), (-3, -3, -3))
  // 135 degrees - NW
  let k135 = ((5, 5, -3), (5, 0, -3), (-3, -3, -3))

  let gap-x = 3.8
  let gap-y = 3.2

  // Top row
  draw-kirsch-kernel((), 0, 0, k0, $0 degree$)
  draw-kirsch-kernel((), gap-x, 0, k45, $45 degree$)

  // Bottom row
  draw-kirsch-kernel((), 0, -gap-y, k90, $90 degree$)
  draw-kirsch-kernel((), gap-x, -gap-y, k135, $135 degree$)

  // Center text between 4 grids
  let total-w = gap-x + 3 * cell-size
  let cx = total-w / 2
  let cy = -(3 * cell-size + (gap-y - 3 * cell-size) / 2)
})
