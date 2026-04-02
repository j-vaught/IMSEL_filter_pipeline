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
#let cell-size = 0.55

// Draw a 5x5 kernel grid
#let draw-kernel-5x5(ctx, ox, oy, kernel, label) = {
  import cetz.draw: *

  let rows = kernel.len()
  let cols = kernel.at(0).len()

  for r in range(rows) {
    for c in range(cols) {
      let v = kernel.at(r).at(c)
      let x = ox + c * cell-size
      let y = oy - r * cell-size

      let abs-v = calc.abs(v)
      let max-v = 2.0
      let t = calc.min(abs-v / max-v, 1.0)

      let fill-color = if v > 0.01 {
        garnet.transparentize(100% - t * 100%)
      } else if v < -0.01 {
        atlantic.transparentize(100% - t * 100%)
      } else {
        white
      }

      rect(
        (x, y),
        (x + cell-size, y - cell-size),
        fill: fill-color,
        stroke: 0.3pt + black30,
      )
    }
  }
  let cx = ox + cols * cell-size / 2
  let by = oy - rows * cell-size - 0.3
  content((cx, by), text(fill: black90, size: 10pt, style: "italic")[$#label$])
}

#cetz.canvas({
  import cetz.draw: *

  // === TOP ROW: Two basis kernels ===

  // Even-symmetric (second derivative of Gaussian, cos-like)
  // Positive center horizontal band, negative above and below
  let f-even = (
    ( 0.0,  0.3,  0.5,  0.3,  0.0),
    ( 0.3,  0.8,  1.5,  0.8,  0.3),
    (-0.5, -1.5, -2.0, -1.5, -0.5),
    ( 0.3,  0.8,  1.5,  0.8,  0.3),
    ( 0.0,  0.3,  0.5,  0.3,  0.0),
  )

  // Odd-symmetric (first derivative of Gaussian, sin-like)
  // Negative top, positive bottom
  let f-odd = (
    ( 0.0, -0.3, -0.5, -0.3,  0.0),
    ( 0.0, -0.8, -2.0, -0.8,  0.0),
    ( 0.0,  0.0,  0.0,  0.0,  0.0),
    ( 0.0,  0.8,  2.0,  0.8,  0.0),
    ( 0.0,  0.3,  0.5,  0.3,  0.0),
  )

  let k-width = 5 * cell-size
  let gap = 1.8

  draw-kernel-5x5((), 0, 0, f-even, $F_"even"$)
  draw-kernel-5x5((), k-width + gap, 0, f-odd, $F_"odd"$)

  // Labels above
  content((k-width / 2, 0.5), text(fill: black50, size: 9pt)[Even (cos)])
  content((k-width + gap + k-width / 2, 0.5), text(fill: black50, size: 9pt)[Odd (sin)])

  // === BOTTOM: Polar plot of oriented energy ===
  let polar-cx = (2 * k-width + gap) / 2
  let polar-cy = -5.8
  let radius = 2.2

  // Light reference circle
  circle((polar-cx, polar-cy), radius: radius, stroke: 0.5pt + black10, fill: none)

  // Axes
  line((polar-cx - radius - 0.3, polar-cy), (polar-cx + radius + 0.3, polar-cy), stroke: 0.4pt + black30)
  line((polar-cx, polar-cy - radius - 0.3), (polar-cx, polar-cy + radius + 0.3), stroke: 0.4pt + black30)

  // Draw E(theta) as a figure-8 / peanut shape
  // E(theta) = cos^2(theta) gives lobes at 0 and 180 (horizontal edge normal)
  let n-pts = 200
  let pts = ()
  for i in range(n-pts + 1) {
    let theta = i / n-pts * 360
    let rad = theta * calc.pi / 180
    // Figure-8: r = |cos(theta)| -- two lobes pointing left and right
    let r-val = calc.pow(calc.cos(rad), 2) * 0.85 + 0.08
    let px = polar-cx + r-val * radius * calc.cos(rad)
    let py = polar-cy + r-val * radius * calc.sin(rad)
    pts.push((px, py))
  }

  for i in range(pts.len() - 1) {
    line(pts.at(i), pts.at(i + 1), stroke: 1.8pt + garnet)
  }

  // Peak direction arrow (pointing right, 0 degrees -- edge normal)
  let arrow-end = (0.85 + 0.08) * radius + 0.1
  line(
    (polar-cx, polar-cy),
    (polar-cx + arrow-end, polar-cy),
    stroke: 1.5pt + garnet,
    mark: (end: "stealth", fill: garnet, scale: 0.6),
  )
  content((polar-cx + arrow-end + 0.4, polar-cy), text(fill: garnet, size: 9pt, weight: "bold")[$hat(theta)$])

  // E(theta) label
  content((polar-cx + radius + 0.5, polar-cy + radius * 0.65), text(fill: garnet, size: 9pt, weight: "bold")[$E(theta)$])

  // Formula
  content((polar-cx, polar-cy - radius - 0.6), text(fill: black90, size: 9pt)[
    $E(theta) = F_"even"^2(theta) + F_"odd"^2(theta)$
  ])

  // === TITLE ===
  content((polar-cx, 1.2), text(fill: black90, size: 11pt, weight: "bold")[
    Oriented Energy (Morrone & Owens, 1987)
  ])
})
