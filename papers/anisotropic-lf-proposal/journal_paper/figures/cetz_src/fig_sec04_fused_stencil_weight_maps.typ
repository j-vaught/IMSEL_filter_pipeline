#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#let N = 15
#let cell = 0.35
#let half = (N - 1) / 2

// Fused stencil parameters
#let m = 7
#let sig-along = 3.5   // spread along edge direction
#let sig-across = 1.4  // spread across edge direction

// Compute rotated derivative-of-Gaussian weight for fused stencil
// The fused stencil is a sum of virtual-pixel contributions, producing
// an irregular dipole with many near-zero cells from cancellation.
#let fused-weight(r, c, theta) = {
  let dx = c - half
  let dy = -(r - half)
  let cos-t = calc.cos(theta)
  let sin-t = calc.sin(theta)
  // u = along edge, v = across edge
  let u = dx * cos-t + dy * sin-t
  let v = -dx * sin-t + dy * cos-t

  // Base derivative-of-Gaussian shape
  let base = -v * calc.exp(-0.5 * (u * u / (sig-along * sig-along) + v * v / (sig-across * sig-across)))

  // Add irregular cancellation noise to simulate fused stencil artifacts
  // Use a deterministic pseudo-random perturbation based on position
  let hash = calc.sin(dx * 12.9898 + dy * 78.233) * 43758.5453
  let frac = hash - calc.floor(hash)
  let noise = (frac - 0.5) * 0.35

  // Many cells get pushed toward zero (cancellation effect)
  let raw = base + noise * calc.exp(-0.5 * (u * u / (sig-along * sig-along * 1.5) + v * v / (sig-across * sig-across * 1.5)))

  // Clip small values to zero (simulating near-zero cancellation)
  if calc.abs(raw) < 0.08 { 0.0 } else { raw }
}

// Find global max for normalization across all orientations
#let global-max = {
  let mx = 0.0
  for theta-deg in (0, 45, 90, 135) {
    let theta = theta-deg * 1deg
    for r in range(N) {
      for c in range(N) {
        let w = calc.abs(fused-weight(r, c, theta))
        if w > mx { mx = w }
      }
    }
  }
  mx
}

// Color interpolation helper
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

// Draw one weight map panel
#let draw-panel(ox, oy, theta-deg) = {
  import cetz.draw: *

  let theta = theta-deg * 1deg

  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell

      let w = fused-weight(r, c, theta)
      let intensity = calc.abs(w) / global-max

      let fc = if w > 0.001 {
        lerp-color(garnet, intensity)
      } else if w < -0.001 {
        lerp-color(atlantic, intensity)
      } else {
        white
      }

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.2pt + black30,
      )
    }
  }

  // Panel border
  rect(
    (ox, oy),
    (ox + N * cell, oy - N * cell),
    stroke: 0.8pt + black90,
  )

  // Angle label below
  let cx = ox + N * cell / 2
  let deg-str = str(theta-deg)
  content((cx, oy - N * cell - 0.45), text(fill: black90, size: 10pt, weight: "bold")[
    #math.equation(block: false, [#math.theta #math.eq #deg-str #math.degree])
  ])
}

#cetz.canvas({
  import cetz.draw: *

  let grid-w = N * cell
  let gap = 0.7

  // Draw four panels horizontally
  for (i, angle) in (0, 45, 90, 135).enumerate() {
    let ox = i * (grid-w + gap)
    draw-panel(ox, 0, angle)
  }

  // Title
  let total-w = 4 * grid-w + 3 * gap
  content((total-w / 2, 1.1), text(fill: black90, size: 11pt, weight: "bold")[
    Fused Stencil Weights at Four Orientations ($m = 7$, $N_p = 15$)
  ])

  // Color legend
  let ly = -N * cell - 1.2
  let lx = total-w / 2 - 2.5

  rect((lx, ly), (lx + 0.35, ly - 0.35), fill: garnet, stroke: 0.3pt + black90)
  content((lx + 0.6, ly - 0.175), anchor: "west", text(fill: black90, size: 8pt)[Positive])

  rect((lx + 1.8, ly), (lx + 2.15, ly - 0.35), fill: atlantic, stroke: 0.3pt + black90)
  content((lx + 2.4, ly - 0.175), anchor: "west", text(fill: black90, size: 8pt)[Negative])

  rect((lx + 3.6, ly), (lx + 3.95, ly - 0.35), fill: white, stroke: 0.3pt + black90)
  content((lx + 4.2, ly - 0.175), anchor: "west", text(fill: black90, size: 8pt)[Near-zero])
})
