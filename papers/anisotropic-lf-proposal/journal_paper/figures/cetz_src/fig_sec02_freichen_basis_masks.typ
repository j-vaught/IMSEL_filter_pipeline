#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black10 = rgb("#ECECEC")
#let black30 = rgb("#C7C7C7")
#let cell-size = 0.8
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")

#let draw-fc-kernel(ctx, ox, oy, kernel, label) = {
  import cetz.draw: *

  let rows = kernel.len()
  let cols = kernel.at(0).len()

  for r in range(rows) {
    for c in range(cols) {
      let entry = kernel.at(r).at(c)
      let v = entry.at(0) // numeric value for color
      let display = entry.at(1) // display string
      let x = ox + c * cell-size
      let y = oy - r * cell-size

      let fill-color = if v > 0.001 { garnet.lighten(40%) } else if v < -0.001 { atlantic.lighten(40%) } else { black10 }
      let text-color = if calc.abs(v) < 0.001 { black90 } else { white }

      rect(
        (x, y),
        (x + cell-size, y - cell-size),
        fill: fill-color,
        stroke: 0.2pt + black30,
      )
      content(
        (x + cell-size / 2, y - cell-size / 2),
        text(fill: text-color, weight: "bold", size: 7.5pt)[#display],
      )
    }
  }
  let cx = ox + cols * cell-size / 2
  let by = oy - rows * cell-size - 0.25
  content((cx, by), text(fill: black90, size: 8.5pt, style: "italic")[#label])
}

#cetz.canvas({
  import cetz.draw: *

  // Frei-Chen basis masks (selected 4 of 9)
  // f1: edge detector (horizontal)  1/2√2 * [[-1,-√2,-1],[0,0,0],[1,√2,1]]
  let s2 = calc.sqrt(2)
  let f1 = (
    ((-1, $-1$), (-s2, $-sqrt(2)$), (-1, $-1$)),
    ((0, $0$), (0, $0$), (0, $0$)),
    ((1, $+1$), (s2, $+sqrt(2)$), (1, $+1$)),
  )

  // f2: edge detector (vertical) 1/2√2 * [[-1,0,1],[-√2,0,√2],[-1,0,1]]
  let f2 = (
    ((-1, $-1$), (0, $0$), (1, $+1$)),
    ((-s2, $-sqrt(2)$), (0, $0$), (s2, $+sqrt(2)$)),
    ((-1, $-1$), (0, $0$), (1, $+1$)),
  )

  // f5: line detector [[0,1,0],[-1,0,-1],[0,1,0]]
  let f5 = (
    ((0, $0$), (1, $+1$), (0, $0$)),
    ((-1, $-1$), (0, $0$), (-1, $-1$)),
    ((0, $0$), (1, $+1$), (0, $0$)),
  )

  // f3: line detector 1/2√2 * [[1,0,-1],[0,0,0],[-1,0,1]]
  let f3 = (
    ((1, $+1$), (0, $0$), (-1, $-1$)),
    ((0, $0$), (0, $0$), (0, $0$)),
    ((-1, $-1$), (0, $0$), (1, $+1$)),
  )

  let gap-x = 4.0

  // Edge subspace label — centered over f1 and f2 (x=0 to gap-x + 3*cell-size)
  let edge-cx = (0 + gap-x + 3 * cell-size) / 2
  content((edge-cx, 0.6), text(fill: black90, size: 9pt, weight: "bold")[Edge subspace])

  // Line subspace label — centered over f5 and f3 (x=2*gap-x to 3*gap-x + 3*cell-size)
  let line-cx = (2 * gap-x + 3 * gap-x + 3 * cell-size) / 2
  content((line-cx, 0.6), text(fill: black90, size: 9pt, weight: "bold")[Line subspace])

  draw-fc-kernel((), 0, 0, f1, $f_1$)
  draw-fc-kernel((), gap-x, 0, f2, $f_2$)
  draw-fc-kernel((), 2 * gap-x, 0, f5, $f_5$)
  draw-fc-kernel((), 3 * gap-x, 0, f3, $f_3$)
})
