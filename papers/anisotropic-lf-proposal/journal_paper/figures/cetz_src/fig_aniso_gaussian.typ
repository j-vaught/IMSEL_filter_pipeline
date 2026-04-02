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

#cetz.canvas({
  import cetz.draw: *

  let grid-size = 11
  let cell = 0.4
  let gx = 0.0
  let gy = 0.0

  // === PIXEL GRID ===
  for r in range(grid-size) {
    for c in range(grid-size) {
      let x = gx + c * cell
      let y = gy - r * cell
      rect(
        (x, y),
        (x + cell, y - cell),
        fill: white,
        stroke: 0.2pt + black30,
      )
    }
  }

  let grid-w = grid-size * cell
  let cx = gx + grid-w / 2
  let cy = gy - grid-w / 2

  // === ELLIPTICAL CONTOURS at ~30 degrees ===
  let angle-deg = 30
  let angle-rad = angle-deg * calc.pi / 180

  // Draw rotated ellipse as polygon segments
  let draw-ellipse(ecx, ecy, a, b, rot, clr, thickness) = {
    let n = 100
    let pts = ()
    for i in range(n + 1) {
      let t = i / n * 2 * calc.pi
      let ex = a * calc.cos(t)
      let ey = b * calc.sin(t)
      let rx = ex * calc.cos(rot) - ey * calc.sin(rot)
      let ry = ex * calc.sin(rot) + ey * calc.cos(rot)
      pts.push((ecx + rx, ecy + ry))
    }
    for i in range(pts.len() - 1) {
      line(pts.at(i), pts.at(i + 1), stroke: thickness + clr)
    }
  }

  // Three nested ellipses (elongated along 30 deg)
  draw-ellipse(cx, cy, 2.0, 0.65, angle-rad, congaree.transparentize(75%), 0.8pt)
  draw-ellipse(cx, cy, 1.35, 0.42, angle-rad, congaree.transparentize(50%), 1.0pt)
  draw-ellipse(cx, cy, 0.7, 0.22, angle-rad, congaree, 1.4pt)

  // Label for sigma
  content((cx + 2.3, cy + 1.2), text(fill: congaree, size: 8pt)[$sigma_"maj" >> sigma_"min"$])

  // === DERIVATIVE SHADING along perpendicular ===
  // Perpendicular to major axis (30 deg) is at 120 deg
  let perp-rad = angle-rad + calc.pi / 2

  // Draw gradient shading: garnet on one side, atlantic on the other
  // Use small filled circles along the perpendicular
  let n-fill = 20
  let max-dist = 1.6
  for i in range(n-fill) {
    let t = (i + 0.5) / n-fill
    let intensity = calc.max(0.0, 1.0 - t * 0.8)
    let dot-r = 0.12 - t * 0.04

    // Positive side (garnet)
    let px1 = cx + t * max-dist * calc.cos(perp-rad)
    let py1 = cy + t * max-dist * calc.sin(perp-rad)
    circle((px1, py1), radius: dot-r, fill: garnet.transparentize((1.0 - intensity) * 100%), stroke: none)

    // Negative side (atlantic)
    let px2 = cx - t * max-dist * calc.cos(perp-rad)
    let py2 = cy - t * max-dist * calc.sin(perp-rad)
    circle((px2, py2), radius: dot-r, fill: atlantic.transparentize((1.0 - intensity) * 100%), stroke: none)
  }

  // Perpendicular line (derivative cross-section)
  let p1x = cx + max-dist * calc.cos(perp-rad)
  let p1y = cy + max-dist * calc.sin(perp-rad)
  let p2x = cx - max-dist * calc.cos(perp-rad)
  let p2y = cy - max-dist * calc.sin(perp-rad)
  line((p1x, p1y), (p2x, p2y), stroke: 0.5pt + black50)

  // + and - labels at ends of perpendicular, offset outward
  content(
    (cx + (max-dist + 0.3) * calc.cos(perp-rad), cy + (max-dist + 0.3) * calc.sin(perp-rad)),
    text(fill: garnet, size: 11pt, weight: "bold")[+],
  )
  content(
    (cx - (max-dist + 0.35) * calc.cos(perp-rad), cy - (max-dist + 0.35) * calc.sin(perp-rad)),
    text(fill: atlantic, size: 11pt, weight: "bold")[$minus$],
  )

  // === INSET: Combined kernel weight grid ===
  let inset-ox = grid-w + 1.2
  let inset-oy = -0.2
  let ic = 0.5

  // 5x5 kernel: anisotropic derivative at ~30 deg
  // Positive upper-left to center, negative center to lower-right
  let kernel = (
    ( 0.0,  0.0,  0.0,  0.4,  0.6),
    ( 0.0,  0.0,  0.4,  1.0,  0.4),
    (-0.2, -0.4,  0.0,  0.4,  0.2),
    (-0.4, -1.0, -0.4,  0.0,  0.0),
    (-0.6, -0.4,  0.0,  0.0,  0.0),
  )

  for r in range(5) {
    for c in range(5) {
      let v = kernel.at(r).at(c)
      let x = inset-ox + c * ic
      let y = inset-oy - r * ic

      let abs-v = calc.abs(v)
      let t = calc.min(abs-v / 1.0, 1.0)

      let fill-color = if v > 0.01 {
        garnet.transparentize(100% - t * 100%)
      } else if v < -0.01 {
        atlantic.transparentize(100% - t * 100%)
      } else {
        white
      }

      rect(
        (x, y),
        (x + ic, y - ic),
        fill: fill-color,
        stroke: 0.3pt + black30,
      )
    }
  }

  content((inset-ox + 5 * ic / 2, inset-oy - 5 * ic - 0.35), text(fill: black90, size: 9pt, style: "italic")[
    Combined kernel
  ])

  // Arrow from grid to inset
  line(
    (grid-w + 0.3, cy + 0.2),
    (inset-ox - 0.2, inset-oy - 5 * ic / 2),
    stroke: 0.8pt + black30,
    mark: (end: "stealth", fill: black30, scale: 0.5),
  )

  // === ANNOTATIONS ===
  content((cx, gy - grid-w - 0.7), text(fill: black50, size: 9pt)[
    Elongated envelope #sym.times derivative profile
  ])

  // === TITLE ===
  let title-x = (grid-w + inset-ox + 5 * ic) / 2
  content((title-x, 1.0), text(fill: black90, size: 11pt, weight: "bold")[
    Anisotropic Gaussian Derivatives (Geusebroek et al., 2003)
  ])
})
