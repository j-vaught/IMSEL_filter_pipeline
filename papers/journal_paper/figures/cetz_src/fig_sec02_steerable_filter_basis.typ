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

// Draw a 5x5 kernel grid with diverging color
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

  // === LEFT SIDE: Two basis kernels ===

  // G1: horizontal derivative of Gaussian (left-right dipole)
  let g1 = (
    ( 0.0,  0.0,  0.0,  0.0,  0.0),
    (-0.3, -0.8,  0.0,  0.8,  0.3),
    (-0.5, -2.0,  0.0,  2.0,  0.5),
    (-0.3, -0.8,  0.0,  0.8,  0.3),
    ( 0.0,  0.0,  0.0,  0.0,  0.0),
  )

  // G2: vertical derivative of Gaussian (top-bottom dipole)
  let g2 = (
    ( 0.0, -0.3, -0.5, -0.3,  0.0),
    ( 0.0, -0.8, -2.0, -0.8,  0.0),
    ( 0.0,  0.0,  0.0,  0.0,  0.0),
    ( 0.0,  0.8,  2.0,  0.8,  0.0),
    ( 0.0,  0.3,  0.5,  0.3,  0.0),
  )

  draw-kernel-5x5((), 0, 0, g1, $G_1$)
  draw-kernel-5x5((), 0, -3.6, g2, $G_2$)

  // Annotations for G1, G2
  content((5 * cell-size + 0.8, -2.5 * cell-size), text(fill: black50, size: 8pt)[
    $partial_x$ Gaussian
  ])
  content((5 * cell-size + 0.8, -3.6 - 2.5 * cell-size), text(fill: black50, size: 8pt)[
    $partial_y$ Gaussian
  ])

  // === RIGHT SIDE: Polar interpolation diagram ===
  let polar-cx = 8.0
  let polar-cy = -3.2
  let radius = 2.5

  // Reference circle (light)
  circle((polar-cx, polar-cy), radius: radius, stroke: 0.5pt + black30, fill: none)

  // Axes (light)
  line((polar-cx - radius - 0.4, polar-cy), (polar-cx + radius + 0.4, polar-cy), stroke: 0.4pt + black30)
  line((polar-cx, polar-cy - radius - 0.4), (polar-cx, polar-cy + radius + 0.4), stroke: 0.4pt + black30)

  // Axis labels
  content((polar-cx + radius + 0.6, polar-cy - 0.15), text(fill: black50, size: 8pt)[$0degree$])
  content((polar-cx + 0.3, polar-cy + radius + 0.35), text(fill: black50, size: 8pt)[$90degree$])

  // Draw R(theta) as a cos^2 figure-8, showing the signed response R(theta) = cos(theta - phi)
  // Plotted as |R(theta)| which creates two lobes (positive lobe in peak direction, negative opposite)
  // This clearly shows directional selectivity
  let phi = 20 * calc.pi / 180  // peak direction at ~20 degrees
  let n-pts = 300

  // Draw two half-lobes with different shading to indicate sign
  // Positive lobe (R > 0): solid atlantic
  // Negative lobe (R < 0): dashed/lighter atlantic
  let pos-pts = ()
  let neg-pts = ()

  let r-func(rad) = {
    calc.cos(rad - phi)  // ranges from -1 to 1
  }

  for i in range(n-pts + 1) {
    let theta = i / n-pts * 360
    let rad = theta * calc.pi / 180
    let r-val = r-func(rad)
    let r-abs = calc.abs(r-val) * 0.9  // scale to fit inside reference circle
    let px = polar-cx + r-abs * radius * calc.cos(rad)
    let py = polar-cy + r-abs * radius * calc.sin(rad)
    if r-val >= 0 {
      pos-pts.push((px, py))
    } else {
      neg-pts.push((px, py))
    }
  }

  // Draw positive lobe (solid, thick)
  for i in range(pos-pts.len() - 1) {
    line(pos-pts.at(i), pos-pts.at(i + 1), stroke: 2pt + atlantic)
  }
  // Draw negative lobe (thinner, dashed-style lighter)
  for i in range(neg-pts.len() - 1) {
    line(neg-pts.at(i), neg-pts.at(i + 1), stroke: 1.2pt + atlantic.transparentize(50%))
  }

  // Add "+" and "-" labels on the lobes
  let pos-label-r = 0.5 * 0.9 * radius
  content(
    (polar-cx + pos-label-r * calc.cos(phi), polar-cy + pos-label-r * calc.sin(phi)),
    text(fill: atlantic, size: 9pt, weight: "bold")[+],
  )
  content(
    (polar-cx - pos-label-r * calc.cos(phi), polar-cy - pos-label-r * calc.sin(phi)),
    text(fill: atlantic.transparentize(40%), size: 9pt, weight: "bold")[$minus$],
  )

  // Mark specific angles with dots on the curve
  for ang in (0, 45, 90) {
    let rad = ang * calc.pi / 180
    let r-val = r-func(rad)
    let r-abs = calc.abs(r-val) * 0.9
    let px = polar-cx + r-abs * radius * calc.cos(rad)
    let py = polar-cy + r-abs * radius * calc.sin(rad)
    circle((px, py), radius: 0.08, fill: garnet, stroke: none)
    // Label offset outward
    let d = calc.sqrt(calc.pow(px - polar-cx, 2) + calc.pow(py - polar-cy, 2))
    let nx = if d > 0.01 { (px - polar-cx) / d } else { calc.cos(rad) }
    let ny = if d > 0.01 { (py - polar-cy) / d } else { calc.sin(rad) }
    content((px + nx * 0.4, py + ny * 0.4), text(fill: garnet, size: 7.5pt)[#str(ang)#sym.degree])
  }

  // R(theta) label
  content((polar-cx + 1.2, polar-cy + radius + 0.35), text(fill: atlantic, size: 9pt, weight: "bold")[$R(theta)$])

  // Formula
  content((polar-cx, polar-cy - radius - 0.6), text(fill: black90, size: 9pt)[
    $R(theta) = cos theta dot R_1 + sin theta dot R_2$
  ])

  // Annotation
  content((polar-cx, polar-cy - radius - 1.15), text(fill: black50, size: 8pt)[
    Analytic interpolation --- no discrete evaluation needed
  ])

  // === TITLE ===
  let title-cx = (5 * cell-size / 2 + polar-cx) / 2 + 0.5
  content((title-cx, 1.0), text(fill: black90, size: 11pt, weight: "bold")[
    Steerable Filters (Freeman & Adelson, 1991)
  ])

  // Connecting arrow from kernels to polar plot
  line(
    (5 * cell-size + 1.4, -3.2),
    (polar-cx - radius - 0.5, polar-cy),
    stroke: 1pt + black30,
    mark: (end: "stealth", fill: black30, scale: 0.5),
  )
})
