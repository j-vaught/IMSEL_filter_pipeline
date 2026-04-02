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

#let cell = 0.65
#let rows = 5
#let cols = 7

// Color interpolation helper
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

// Virtual pixel j: derivative contribution centered above midline
#let wj = (
  ( 0.00,  0.03,  0.12,  0.18,  0.14,  0.04,  0.00),
  ( 0.02,  0.10,  0.25,  0.38,  0.28,  0.11,  0.02),
  ( 0.01,  0.06,  0.15,  0.22,  0.16,  0.07,  0.01),
  ( 0.00, -0.02, -0.06, -0.10, -0.07, -0.03,  0.00),
  ( 0.00,  0.00, -0.02, -0.04, -0.02,  0.00,  0.00),
)

// Virtual pixel j+1: shifted down, overlaps with j in the middle rows
#let wj1 = (
  ( 0.00,  0.00,  0.02,  0.04,  0.02,  0.00,  0.00),
  ( 0.00,  0.02,  0.06,  0.10,  0.07,  0.03,  0.00),
  (-0.01, -0.06, -0.15, -0.22, -0.16, -0.07, -0.01),
  (-0.02, -0.10, -0.25, -0.38, -0.28, -0.11, -0.02),
  ( 0.00, -0.03, -0.12, -0.18, -0.14, -0.04,  0.00),
)

// Precompute summed weights
#let sum-weights = {
  let result = ()
  for r in range(rows) {
    let row = ()
    for c in range(cols) {
      let s = wj.at(r).at(c) + wj1.at(r).at(c)
      let rounded = calc.round(s * 100) / 100
      row.push(rounded)
    }
    result.push(row)
  }
  result
}

#let sum-max = {
  let mx = 0.0
  for r in range(rows) {
    for c in range(cols) {
      let v = calc.abs(sum-weights.at(r).at(c))
      if v > mx { mx = v }
    }
  }
  mx
}

#let nz-count = {
  let ct = 0
  for r in range(rows) {
    for c in range(cols) {
      if calc.abs(sum-weights.at(r).at(c)) < 0.035 { ct += 1 }
    }
  }
  ct
}

#let total-cells = rows * cols

// Draw a weight grid with colored cells and numeric labels
#let draw-grid(ox, oy, weights, pos-color, neg-color, max-val) = {
  import cetz.draw: *

  for r in range(rows) {
    for c in range(cols) {
      let x = ox + c * cell
      let y = oy - r * cell
      let w = weights.at(r).at(c)
      let intensity = calc.abs(w) / max-val

      let fc = if calc.abs(w) < 0.015 {
        white
      } else if w > 0 {
        lerp-color(pos-color, intensity)
      } else {
        lerp-color(neg-color, intensity)
      }

      let tc = if intensity > 0.45 and calc.abs(w) >= 0.015 { white } else { black90 }

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.3pt + black30,
      )

      let label = if calc.abs(w) < 0.005 { "0" } else {
        let sign = if w < 0 { "\u{2212}" } else { "" }
        let mag = calc.abs(w)
        let scaled = calc.round(mag * 100)
        let s = str(int(scaled))
        if s.len() == 1 { sign + "." + "0" + s } else { sign + "." + s }
      }

      content(
        (x + cell / 2, y - cell / 2),
        text(fill: tc, size: 5.5pt)[#label],
      )
    }
  }

  rect(
    (ox, oy),
    (ox + cols * cell, oy - rows * cell),
    stroke: 0.8pt + black90,
  )
}

// Draw the summed grid with near-zero highlighting
#let draw-sum-grid(ox, oy) = {
  import cetz.draw: *

  for r in range(rows) {
    for c in range(cols) {
      let x = ox + c * cell
      let y = oy - r * cell
      let w = sum-weights.at(r).at(c)
      let intensity = calc.abs(w) / sum-max

      let fc = if calc.abs(w) < 0.035 {
        black10
      } else if w > 0 {
        lerp-color(garnet, intensity)
      } else {
        lerp-color(atlantic, intensity)
      }

      let tc = if intensity > 0.45 and calc.abs(w) >= 0.035 { white } else { black90 }

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.3pt + black30,
      )

      let label = if calc.abs(w) < 0.005 { "0" } else {
        let sign = if w < 0 { "\u{2212}" } else { "" }
        let mag = calc.abs(w)
        let scaled = calc.round(mag * 100)
        let s = str(int(scaled))
        if s.len() == 1 { sign + "." + "0" + s } else { sign + "." + s }
      }

      content(
        (x + cell / 2, y - cell / 2),
        text(fill: tc, size: 5.5pt)[#label],
      )
    }
  }

  rect(
    (ox, oy),
    (ox + cols * cell, oy - rows * cell),
    stroke: 0.8pt + black90,
  )
}

#cetz.canvas({
  import cetz.draw: *

  let grid-w = cols * cell
  let grid-h = rows * cell
  let spacing = 1.4

  let max-w = 0.38

  // ===== TOP PART: Cancellation illustration =====

  // Grid A: Virtual pixel j
  let ax = 0
  let ay = 0
  draw-grid(ax, ay, wj, garnet, atlantic, max-w)
  content((ax + grid-w / 2, ay + 0.5), text(fill: black90, size: 9pt, weight: "bold")[Virtual pixel $j$])

  // Plus symbol
  let plus-x = ax + grid-w + spacing / 2
  let plus-y = -grid-h / 2
  content((plus-x, plus-y), text(fill: black90, size: 16pt, weight: "bold")[$+$])

  // Grid B: Virtual pixel j+1
  let bx = ax + grid-w + spacing
  let by = 0
  draw-grid(bx, by, wj1, congaree, rose, max-w)
  content((bx + grid-w / 2, by + 0.5), text(fill: black90, size: 9pt, weight: "bold")[Virtual pixel $j + 1$])

  // Equals symbol
  let eq-x = bx + grid-w + spacing / 2
  content((eq-x, plus-y), text(fill: black90, size: 16pt, weight: "bold")[$=$])

  // Grid C: Summed weights
  let cx-pos = bx + grid-w + spacing
  let cy-pos = 0

  draw-sum-grid(cx-pos, cy-pos)
  content((cx-pos + grid-w / 2, cy-pos + 0.5), text(fill: black90, size: 9pt, weight: "bold")[Summed (cancellation)])

  content((cx-pos + grid-w / 2, cy-pos - grid-h - 0.5), text(fill: black90, size: 8pt)[
    #nz-count of #total-cells cells near-zero from cancellation
  ])

  // ===== BOTTOM PART: Histogram of |alpha| values =====

  let hist-ox = 1.0
  let hist-oy = -grid-h - 1.8
  let hist-w = 10.0
  let hist-h = 4.0

  let bins = (
    (0.02, 0.95, true),
    (0.06, 0.70, true),
    (0.10, 0.40, true),
    (0.14, 0.24, false),
    (0.18, 0.16, false),
    (0.22, 0.12, false),
    (0.26, 0.09, false),
    (0.30, 0.07, false),
    (0.34, 0.055, false),
    (0.38, 0.045, false),
    (0.42, 0.038, false),
    (0.46, 0.032, false),
    (0.50, 0.028, false),
    (0.55, 0.024, false),
    (0.60, 0.020, false),
    (0.65, 0.017, false),
    (0.70, 0.014, false),
    (0.75, 0.011, false),
    (0.80, 0.009, false),
    (0.85, 0.007, false),
    (0.90, 0.005, false),
    (0.95, 0.003, false),
  )

  let bar-w-hist = 0.038 * hist-w

  // X-axis (horizontal baseline)
  line((hist-ox, hist-oy), (hist-ox + hist-w, hist-oy), stroke: 0.8pt + black90)
  // Y-axis (vertical, going up)
  line((hist-ox, hist-oy), (hist-ox, hist-oy - hist-h), stroke: 0.8pt + black90)

  // X-axis label
  content((hist-ox + hist-w / 2, hist-oy + 0.65), text(fill: black90, size: 9pt)[
    Weight magnitude $|alpha|$
  ])

  // Y-axis label
  content(
    (hist-ox - 0.7, hist-oy - hist-h / 2),
    angle: 90deg,
    text(fill: black90, size: 9pt)[Count of pixels]
  )

  // X-axis ticks
  for tick-val in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) {
    let tx = hist-ox + tick-val * hist-w
    line((tx, hist-oy), (tx, hist-oy + 0.1), stroke: 0.5pt + black90)
    let tick-label = str(tick-val)
    content((tx, hist-oy + 0.32), text(fill: black90, size: 7pt)[#tick-label])
  }

  // Draw histogram bars (growing upward)
  for bin in bins {
    let (xfrac, hfrac, is-nz) = bin
    let bx = hist-ox + xfrac * hist-w - bar-w-hist / 2
    let bh = hfrac * hist-h
    let fc = if is-nz { black30 } else { black50 }

    rect(
      (bx, hist-oy),
      (bx + bar-w-hist, hist-oy - bh),
      fill: fc,
      stroke: 0.3pt + black90,
    )
  }

  // Bracket and label for near-zero region (below x-axis)
  let nz-left = hist-ox
  let nz-right = hist-ox + 0.12 * hist-w + bar-w-hist / 2
  let bracket-y = hist-oy + 0.85

  line(
    (nz-left, bracket-y),
    (nz-right, bracket-y),
    stroke: 1.2pt + rose,
  )
  line((nz-left, bracket-y), (nz-left, bracket-y - 0.15), stroke: 1.2pt + rose)
  line((nz-right, bracket-y), (nz-right, bracket-y - 0.15), stroke: 1.2pt + rose)

  content(((nz-left + nz-right) / 2, bracket-y + 0.35), text(fill: rose, size: 8pt, weight: "bold")[
    \~28% near-zero
  ])

  // Caption below histogram
  content((hist-ox + hist-w / 2, hist-oy - hist-h - 0.45), text(fill: black90, size: 9pt, style: "italic")[
    Distribution of fused stencil weight magnitudes ($m = 7$, $N_p = 15$)
  ])

  // Overall title at top
  let total-w = cx-pos + grid-w
  content((total-w / 2, 1.0), text(fill: black90, size: 11pt, weight: "bold")[
    Weight Cancellation in Fused Stencil
  ])
})
