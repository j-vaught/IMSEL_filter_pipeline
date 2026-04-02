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

#let N = 13
#let cell = 0.42
#let theta = 30deg
#let sig-u = 2.0
#let sig-v = 1.2
#let half = (N - 1) / 2

// Precompute rotated coordinates and weights for rectangular kernel
#let cos-t = calc.cos(theta)
#let sin-t = calc.sin(theta)

// Check if pixel (r,c) is inside the rotated rectangle
#let rect-half-u = 3.0 * sig-u
#let rect-half-v = 3.0 * sig-v

#let in-rect(r, c) = {
  let dx = c - half
  let dy = -(r - half)
  let u = dx * cos-t + dy * sin-t
  let v = -dx * sin-t + dy * cos-t
  calc.abs(u) <= rect-half-u and calc.abs(v) <= rect-half-v
}

#let get-uv(r, c) = {
  let dx = c - half
  let dy = -(r - half)
  let u = dx * cos-t + dy * sin-t
  let v = -dx * sin-t + dy * cos-t
  (u, v)
}

#let gauss(u, v) = {
  calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

#let kernel-weight(u, v) = {
  -v * calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

// Find max values for normalization
#let max-gauss = 1.0
#let max-weight = {
  let m = 0.0
  for r in range(N) {
    for c in range(N) {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        let w = calc.abs(kernel-weight(u, v))
        if w > m { m = w }
      }
    }
  }
  m
}

#let max-deriv = {
  let m = 0.0
  for r in range(N) {
    for c in range(N) {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        if calc.abs(v) > m { m = calc.abs(v) }
      }
    }
  }
  m
}

// Color interpolation helper
#let lerp-color(base, t) = {
  // t in [0,1], blend base with white
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

// Draw a pixel grid panel
#let draw-panel(ox, oy, fill-fn, label-text, math-text) = {
  import cetz.draw: *

  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell

      let fc = fill-fn(r, c)

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.3pt + black30,
      )
    }
  }

  // Panel border
  rect(
    (ox, oy),
    (ox + N * cell, oy - N * cell),
    stroke: 0.8pt + black90,
  )

  // Label above
  let cx = ox + N * cell / 2
  content((cx, oy + 0.55), text(fill: black90, size: 10pt, weight: "bold")[#label-text])
  // Math below
  content((cx, oy - N * cell - 0.4), text(fill: black90, size: 9pt, style: "italic")[#math-text])
}

#cetz.canvas({
  import cetz.draw: *

  let gap = 0.9
  let grid-w = N * cell
  let row-gap = 1.6

  // Panel A: Mask (top-left)
  let ax = 0
  let ay = 0
  draw-panel(ax, ay,
    (r, c) => {
      if in-rect(r, c) { black10 } else { white }
    },
    [(A) Mask $M_R$],
    [$M_R (i,j) = bb(1)[|u| <= 3 sigma_u, |v| <= 3 sigma_v]$]
  )

  // Panel B: Envelope (top-right)
  let bx = grid-w + gap
  let by = 0
  draw-panel(bx, by,
    (r, c) => {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        let g = gauss(u, v)
        lerp-color(garnet, g)
      } else { white }
    },
    [(B) Envelope $G(u,v)$],
    [$G(u,v) = exp(-1/2 (u^2 / sigma_u^2 + v^2 / sigma_v^2))$]
  )

  // Panel C: Derivative (bottom-left)
  let cx-pos = 0
  let cy-pos = -grid-w - row-gap
  draw-panel(cx-pos, cy-pos,
    (r, c) => {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        let intensity = calc.abs(v) / max-deriv
        if v < 0 {
          lerp-color(garnet, intensity)
        } else if v > 0 {
          lerp-color(atlantic, intensity)
        } else { white }
      } else { white }
    },
    [(C) Derivative $-v$],
    [$-v "profile: positive (garnet) / negative (blue)"$]
  )

  // Panel D: Combined kernel (bottom-right)
  let dx = grid-w + gap
  let dy = -grid-w - row-gap
  draw-panel(dx, dy,
    (r, c) => {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        let w = kernel-weight(u, v)
        let intensity = calc.abs(w) / max-weight
        if w > 0.001 {
          lerp-color(garnet, intensity)
        } else if w < -0.001 {
          lerp-color(atlantic, intensity)
        } else { white }
      } else { white }
    },
    [(D) Combined $K_R$],
    [$K_R = -v dot G(u,v) dot M_R$]
  )

  // Title
  let total-w = 2 * grid-w + gap
  content((total-w / 2, 1.3), text(fill: black90, size: 11pt, weight: "bold")[Rectangular Kernel Construction])
})
