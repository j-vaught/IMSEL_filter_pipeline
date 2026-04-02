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

// Precompute Gaussian and derivatives
#let sigma = 1.2
#let n-pts = 61
#let x-range = 4.0

#let gauss(x) = calc.exp(-0.5 * x * x / (sigma * sigma))
#let gauss-d1(x) = -x / (sigma * sigma) * calc.exp(-0.5 * x * x / (sigma * sigma))
#let gauss-d2(x) = (x * x / (sigma * sigma) - 1.0) / (sigma * sigma) * calc.exp(-0.5 * x * x / (sigma * sigma))

// 2D kernel helpers
#let N = 7
#let cell = 0.5
#let half = (N - 1) / 2

#let gauss2d-dx(r, c) = {
  let dx = c - half
  let dy = r - half
  -dx / (sigma * sigma) * calc.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
}

#let log2d(r, c) = {
  let dx = c - half
  let dy = r - half
  let r2 = dx * dx + dy * dy
  let s2 = sigma * sigma
  (r2 / s2 - 2.0) / (s2) * calc.exp(-0.5 * r2 / s2)
}

// Color interpolation
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#cetz.canvas({
  import cetz.draw: *

  // === LEFT HALF: 1D curves ===
  let plot-w = 5.0
  let plot-h = 1.8
  let plot-gap = 0.6

  // Helper to draw a curve
  let draw-curve(ox, oy, func, y-scale, label-text, split-color: false) = {
    // Axes
    line((ox, oy), (ox + plot-w, oy), stroke: 0.5pt + black50)
    line((ox + plot-w / 2, oy - plot-h * 0.5), (ox + plot-w / 2, oy + plot-h * 0.6), stroke: 0.5pt + black50)

    // Curve segments
    let pts = ()
    for i in range(n-pts) {
      let x = -x-range + 2.0 * x-range * i / (n-pts - 1)
      let y = func(x) * y-scale
      let px = ox + plot-w / 2 + x * (plot-w / 2) / x-range
      let py = oy + y
      pts.push((px, py))
    }

    if split-color {
      for i in range(1, pts.len()) {
        let (x1, y1) = pts.at(i - 1)
        let (x2, y2) = pts.at(i)
        let mid-y = (y1 + y2) / 2
        let col = if mid-y >= oy { garnet } else { atlantic }
        line((x1, y1), (x2, y2), stroke: 1.5pt + col)
      }
    } else {
      for i in range(1, pts.len()) {
        let (x1, y1) = pts.at(i - 1)
        let (x2, y2) = pts.at(i)
        line((x1, y1), (x2, y2), stroke: 1.5pt + black90)
      }
    }

    // Label to the left
    content((ox - 0.6, oy + plot-h * 0.15), text(fill: black90, size: 9pt)[#label-text])
  }

  // Top: G(x)
  let y0 = 0
  draw-curve(0, y0, gauss, plot-h * 0.9, [$G(x)$])

  // sigma annotation: small bracket below the peak
  let center-px = plot-w / 2
  let sig-px = plot-w / 2 + sigma * (plot-w / 2) / x-range
  // Place sigma annotation below the bell curve, just above the x-axis
  let ann-y = y0 - 0.15
  line((center-px, ann-y), (center-px, ann-y - 0.15), stroke: 0.5pt + garnet)
  line((sig-px, ann-y), (sig-px, ann-y - 0.15), stroke: 0.5pt + garnet)
  line((center-px, ann-y - 0.15), (sig-px, ann-y - 0.15), stroke: 0.5pt + garnet)
  content(((center-px + sig-px) / 2, ann-y - 0.4), text(fill: garnet, size: 8pt)[$sigma$])

  // Middle: G'(x)
  let y1 = -(plot-h + plot-gap)
  draw-curve(0, y1, gauss-d1, plot-h * 1.4, [$G'(x)$], split-color: true)

  // Bottom: G''(x)
  let y2 = -2 * (plot-h + plot-gap)
  draw-curve(0, y2, gauss-d2, plot-h * 1.2, [$G''(x)$], split-color: true)

  // Color legend below curves
  let leg-y = y2 - plot-h * 0.6 - 0.3
  rect((0.5, leg-y), (0.8, leg-y - 0.25), fill: garnet, stroke: none)
  content((1.4, leg-y - 0.125), text(fill: black90, size: 7.5pt)[positive])
  rect((2.3, leg-y), (2.6, leg-y - 0.25), fill: atlantic, stroke: none)
  content((3.2, leg-y - 0.125), text(fill: black90, size: 7.5pt)[negative])

  // === RIGHT HALF: 2D kernels ===
  let rx = plot-w + 2.5
  let grid-w = N * cell

  // First derivative kernel (dipole)
  let max-dx = {
    let m = 0.0
    for r in range(N) {
      for c in range(N) {
        let v = calc.abs(gauss2d-dx(r, c))
        if v > m { m = v }
      }
    }
    m
  }

  // Vertically center the two grids against the three curve panels
  // Curves span from y0+plot-h*0.9 down to y2-plot-h*0.6
  // Total grid height = 2 * N*cell + 1.0 gap
  let total-grid-h = 2 * N * cell + 1.0
  let curves-top = plot-h * 0.9
  let curves-bot = y2 - plot-h * 0.6
  let curves-mid = (curves-top + curves-bot) / 2
  let ky1 = curves-mid + total-grid-h / 2

  for r in range(N) {
    for c in range(N) {
      let x = rx + c * cell
      let y = ky1 - r * cell
      let v = gauss2d-dx(r, c)
      let intensity = calc.abs(v) / max-dx
      let fc = if v > 0.01 {
        lerp-color(garnet, intensity)
      } else if v < -0.01 {
        lerp-color(atlantic, intensity)
      } else { white }
      rect((x, y), (x + cell, y - cell), fill: fc, stroke: 0.3pt + black30)
    }
  }
  rect((rx, ky1), (rx + grid-w, ky1 - N * cell), stroke: 0.8pt + black90)
  content((rx + grid-w / 2, ky1 + 0.4), text(fill: black90, size: 9pt, weight: "bold")[
    $partial G slash partial x$ (first deriv.)
  ])

  // LoG kernel
  let max-log = {
    let m = 0.0
    for r in range(N) {
      for c in range(N) {
        let v = calc.abs(log2d(r, c))
        if v > m { m = v }
      }
    }
    m
  }

  let ky2 = ky1 - N * cell - 1.0
  for r in range(N) {
    for c in range(N) {
      let x = rx + c * cell
      let y = ky2 - r * cell
      let v = log2d(r, c)
      let intensity = calc.abs(v) / max-log
      let fc = if v > 0.01 {
        lerp-color(garnet, intensity)
      } else if v < -0.01 {
        lerp-color(atlantic, intensity)
      } else { white }
      rect((x, y), (x + cell, y - cell), fill: fc, stroke: 0.3pt + black30)
    }
  }
  rect((rx, ky2), (rx + grid-w, ky2 - N * cell), stroke: 0.8pt + black90)
  content((rx + grid-w / 2, ky2 + 0.4), text(fill: black90, size: 9pt, weight: "bold")[
    $nabla^2 G$ (LoG)
  ])

  // Title - placed above everything
  let total-w = rx + grid-w
  let title-y = calc.max(ky1 + 0.4 + 0.5, plot-h * 0.9 + 0.5) + 0.5
  content((total-w / 2, title-y), text(fill: black90, size: 11pt, weight: "bold")[Gaussian Derivatives (Marr & Hildreth, 1980)])
})
