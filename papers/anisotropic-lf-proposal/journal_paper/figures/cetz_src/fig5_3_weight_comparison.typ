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

#let N = 15
#let cell = 0.36
#let theta = 30deg
#let sig-u = 2.0
#let sig-v = 1.2
#let half = (N - 1) / 2
#let r-cut = 3.0

#let cos-t = calc.cos(theta)
#let sin-t = calc.sin(theta)

#let rect-half-u = 3.0 * sig-u
#let rect-half-v = 3.0 * sig-v

#let get-uv(r, c) = {
  let dx = c - half
  let dy = -(r - half)
  let u = dx * cos-t + dy * sin-t
  let v = -dx * sin-t + dy * cos-t
  (u, v)
}

#let in-rect(r, c) = {
  let (u, v) = get-uv(r, c)
  calc.abs(u) <= rect-half-u and calc.abs(v) <= rect-half-v
}

#let in-ellipse(r, c) = {
  let (u, v) = get-uv(r, c)
  (u * u) / (sig-u * sig-u) + (v * v) / (sig-v * sig-v) <= r-cut * r-cut
}

#let gauss(u, v) = {
  calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

#let kernel-weight(u, v) = {
  -v * calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

// Simple hash for pseudo-random perturbation
#let pseudo-rand(r, c) = {
  let x = r * 17 + c * 31 + 7
  let x2 = calc.rem(x * x + 13 * x + 37, 97)
  x2 / 97.0
}

// Fused stencil: irregular shape with noise added to weights
#let in-fused(r, c) = {
  let (u, v) = get-uv(r, c)
  // Irregular elongated shape - wider than ellipse in some directions, narrower in others
  let base = (u * u) / (sig-u * sig-u * 1.3) + (v * v) / (sig-v * sig-v * 0.9)
  let perturb = (pseudo-rand(r, c) - 0.5) * 2.5
  base + perturb <= r-cut * r-cut
}

#let fused-weight(r, c) = {
  let (u, v) = get-uv(r, c)
  let w = kernel-weight(u, v)
  // Add cancellation noise: many weights become near-zero
  let noise = (pseudo-rand(r, c) - 0.5) * 0.6
  let noisy = w + noise * calc.abs(w) * 2.0
  // Some weights get pushed toward zero (cancellation effect)
  let cancel = pseudo-rand(c, r)
  if cancel < 0.3 { noisy * 0.05 } else { noisy }
}

// Find max absolute weight across all three for consistent color scale
#let global-max = {
  let m = 0.0
  for r in range(N) {
    for c in range(N) {
      if in-rect(r, c) {
        let (u, v) = get-uv(r, c)
        let w = calc.abs(kernel-weight(u, v))
        if w > m { m = w }
      }
      if in-fused(r, c) {
        let w = calc.abs(fused-weight(r, c))
        if w > m { m = w }
      }
    }
  }
  m
}

#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let draw-weight-panel(ox, oy, inside-fn, weight-fn, label-text) = {
  import cetz.draw: *

  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell

      let fc = if inside-fn(r, c) {
        let w = weight-fn(r, c)
        let intensity = calc.abs(w) / global-max
        if w > 0.01 {
          lerp-color(garnet, intensity)
        } else if w < -0.01 {
          lerp-color(atlantic, intensity)
        } else { white }
      } else { white }

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.25pt + black30,
      )
    }
  }

  rect(
    (ox, oy),
    (ox + N * cell, oy - N * cell),
    stroke: 0.8pt + black90,
  )

  let cx = ox + N * cell / 2
  content((cx, oy + 0.5), text(fill: black90, size: 9.5pt, weight: "bold")[#label-text])
}

// Histogram drawing
#let draw-histogram(ox, oy, weights, label-col) = {
  import cetz.draw: *

  let hist-w = N * cell
  let hist-h = 1.4
  let n-bins = 10

  // Bin the absolute weights
  let max-abs = {
    let m = 0.0
    for w in weights {
      if calc.abs(w) > m { m = calc.abs(w) }
    }
    if m == 0.0 { 1.0 } else { m }
  }

  let bins = range(n-bins).map(_ => 0)
  for w in weights {
    let idx = calc.min(calc.floor(calc.abs(w) / max-abs * n-bins), n-bins - 1)
    bins.at(idx) = bins.at(idx) + 1
  }

  let max-count = {
    let m = 1
    for b in bins { if b > m { m = b } }
    m
  }

  let bin-w = hist-w / n-bins

  // Axes
  line((ox, oy), (ox + hist-w, oy), stroke: 0.5pt + black90)
  line((ox, oy), (ox, oy + hist-h), stroke: 0.5pt + black90)

  // Bars
  for i in range(n-bins) {
    let bh = bins.at(i) / max-count * hist-h * 0.9
    if bh > 0.01 {
      rect(
        (ox + i * bin-w, oy),
        (ox + (i + 1) * bin-w, oy + bh),
        fill: label-col,
        stroke: 0.3pt + black90,
      )
    }
  }

  // Labels
  let cx = ox + hist-w / 2
  content((cx, oy - 0.3), text(fill: black90, size: 7pt)[$|alpha|$])
}

#cetz.canvas({
  import cetz.draw: *

  let gap = 0.7
  let grid-w = N * cell

  // --- Panel 1: Fused Stencil ---
  let p1x = 0
  let p1y = 0
  draw-weight-panel(p1x, p1y,
    (r, c) => in-fused(r, c),
    (r, c) => fused-weight(r, c),
    [Fused Stencil]
  )

  // Collect weights for histogram
  let fused-weights = {
    let ws = ()
    for r in range(N) {
      for c in range(N) {
        if in-fused(r, c) {
          ws.push(fused-weight(r, c))
        }
      }
    }
    ws
  }

  let hist-y = p1y - grid-w - 0.4
  draw-histogram(p1x, hist-y, fused-weights, black50)

  // --- Panel 2: Rectangular ---
  let p2x = grid-w + gap
  let p2y = 0
  draw-weight-panel(p2x, p2y,
    (r, c) => in-rect(r, c),
    (r, c) => {
      let (u, v) = get-uv(r, c)
      kernel-weight(u, v)
    },
    [Rectangular $K_R$]
  )

  let rect-weights = {
    let ws = ()
    for r in range(N) {
      for c in range(N) {
        if in-rect(r, c) {
          let (u, v) = get-uv(r, c)
          ws.push(kernel-weight(u, v))
        }
      }
    }
    ws
  }

  draw-histogram(p2x, hist-y, rect-weights, atlantic)

  // --- Panel 3: Elliptical ---
  let p3x = 2 * (grid-w + gap)
  let p3y = 0
  draw-weight-panel(p3x, p3y,
    (r, c) => in-ellipse(r, c),
    (r, c) => {
      let (u, v) = get-uv(r, c)
      kernel-weight(u, v)
    },
    [Elliptical $K_E$]
  )

  let ellip-weights = {
    let ws = ()
    for r in range(N) {
      for c in range(N) {
        if in-ellipse(r, c) {
          let (u, v) = get-uv(r, c)
          ws.push(kernel-weight(u, v))
        }
      }
    }
    ws
  }

  draw-histogram(p3x, hist-y, ellip-weights, congaree)

  // Title
  let total-w = 3 * grid-w + 2 * gap
  content((total-w / 2, 1.2), text(fill: black90, size: 11pt, weight: "bold")[Weight Comparison])
})
