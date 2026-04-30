#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

// ============================================================
// CONFIGURABLE PARAMETERS
// ============================================================
#let all-heights = (1.0, 1.4, 0.9, 1.6, 2.0, 2.4, 2.1, 2.7, 2.5, 3.0, 3.2)
#let m = 3            // half-width → bandwidth covers 2m+1 = 7 bars
#let d = 2            // polynomial degree for LOESS (typically 1 or 2)
#let win-center = 5   // index of center bar (0-based)

// ============================================================
// DERIVED VALUES
// ============================================================
#let n = all-heights.len()
#let win-start = win-center - m
#let win-end = win-center + m
#let win-size = 2 * m + 1
#let bandwidth = float(m)

#let win-data = all-heights.slice(win-start, win-end + 1)

// ============================================================
// TRICUBE KERNEL WEIGHTS
// ============================================================
#let tricube(u) = {
  if calc.abs(u) >= 1.0 { 0.0 }
  else {
    let t = 1.0 - calc.pow(calc.abs(u), 3)
    calc.pow(t, 3)
  }
}

#let weights = {
  let w = ()
  for i in range(win-size) {
    let xi = float(i - m)
    let u = xi / bandwidth
    w.push(tricube(u))
  }
  w
}

// ============================================================
// WEIGHTED LEAST-SQUARES FIT
// ============================================================
#let wdot(a, b, w) = {
  let s = 0.0
  for i in range(a.len()) { s += a.at(i) * b.at(i) * w.at(i) }
  s
}

#let vander-cols = {
  let cols = ()
  for j in range(d + 1) {
    let col = ()
    for i in range(win-size) {
      let xi = float(i - m)
      let val = if j == 0 { 1.0 } else {
        let v = 1.0
        for _ in range(j) { v *= xi }
        v
      }
      col.push(val)
    }
    cols.push(col)
  }
  cols
}

#let gram = {
  let g = ()
  for j in range(d + 1) {
    let row = ()
    for k in range(d + 1) {
      row.push(wdot(vander-cols.at(j), vander-cols.at(k), weights))
    }
    g.push(row)
  }
  g
}

#let rhs = {
  let r = ()
  for j in range(d + 1) {
    r.push(wdot(vander-cols.at(j), win-data, weights))
  }
  r
}

#let solve-system(A, b) = {
  let n = b.len()
  let aug = ()
  for i in range(n) {
    let row = A.at(i) + (b.at(i),)
    aug.push(row)
  }
  for col in range(n) {
    let max-val = calc.abs(aug.at(col).at(col))
    let max-row = col
    for row in range(col + 1, n) {
      let v = calc.abs(aug.at(row).at(col))
      if v > max-val { max-val = v; max-row = row }
    }
    if max-row != col {
      let tmp = aug.at(col)
      aug.at(col) = aug.at(max-row)
      aug.at(max-row) = tmp
    }
    let pivot = aug.at(col).at(col)
    for row in range(col + 1, n) {
      let factor = aug.at(row).at(col) / pivot
      let new-row = ()
      for j in range(n + 1) {
        new-row.push(aug.at(row).at(j) - factor * aug.at(col).at(j))
      }
      aug.at(row) = new-row
    }
  }
  let x = (0.0,) * n
  for i-rev in range(n) {
    let i = n - 1 - i-rev
    let s = aug.at(i).at(n)
    for j in range(i + 1, n) {
      s -= aug.at(i).at(j) * x.at(j)
    }
    x.at(i) = s / aug.at(i).at(i)
  }
  x
}

#let coeffs = solve-system(gram, rhs)

#let poly-eval(xi) = {
  let val = 0.0
  let xp = 1.0
  for j in range(d + 1) {
    val += coeffs.at(j) * xp
    xp *= xi
  }
  val
}

#let deriv-at-center = coeffs.at(1)

#let fitted = {
  let vals = ()
  for i in range(win-size) {
    vals.push(poly-eval(float(i - m)))
  }
  vals
}

// ============================================================
// DRAWING
// ============================================================
#let bar-w = 0.55
#let gap = 0.12
#let step = bar-w + gap

#cetz.canvas({
  import cetz.draw: *

  // Window background rectangle
  let wx0 = win-start * step - gap / 2
  let wx1 = (win-end + 1) * step - gap / 2
  let max-h = calc.max(..all-heights)
  rect(
    (wx0, -0.1),
    (wx1, max-h + 0.7),
    fill: horseshoe.lighten(95%),
    stroke: 0.8pt + horseshoe,
  )

  // Draw all bars — color gradient: garnet (center, high weight) → gray (edge, low weight)
  for i in range(n) {
    let x = i * step
    let h = all-heights.at(i)

    let clr = if i >= win-start and i <= win-end {
      let wi = i - win-start
      let w = weights.at(wi)  // 1.0 at center, ~0 at edges
      // Blend: full weight → muted garnet, zero weight → light gray
      let garnet-part = garnet.lighten(55%)
      let gray-part = black50.lighten(60%)
      color.mix((garnet-part, w * 100%), (gray-part, (1.0 - w) * 100%))
    } else {
      black50.lighten(60%)
    }

    let stroke-clr = if i >= win-start and i <= win-end { black70 } else { black50 }
    rect(
      (x, 0),
      (x + bar-w, h),
      fill: clr,
      stroke: 0.4pt + stroke-clr,
    )
  }

  // (tricube kernel curve removed — weight is shown via bar color gradient)

  // Fitted polynomial curve
  for i in range(win-size - 1) {
    let idx0 = win-start + i
    let idx1 = win-start + i + 1
    let x0 = idx0 * step + bar-w / 2
    let x1 = idx1 * step + bar-w / 2
    let y0 = fitted.at(i)
    let y1 = fitted.at(i + 1)
    line((x0, y0), (x1, y1), stroke: 2.2pt + atlantic)
  }

  // Dots on fitted curve
  for i in range(win-size) {
    let idx = win-start + i
    let x = idx * step + bar-w / 2
    let y = fitted.at(i)
    circle((x, y), radius: 0.05, fill: atlantic, stroke: none)
  }

  // Label
  content(
    ((wx0 + wx1) / 2, max-h + 0.45),
    text(fill: atlantic, size: 8.5pt)[weighted fit],
  )

  // Tangent arrow at center
  let cx = win-center * step + bar-w / 2
  let cy = fitted.at(m)
  let canvas-slope = deriv-at-center

  let tang-half = 0.7
  let offset-y = 0.5
  line(
    (cx - tang-half, cy + offset-y - tang-half * canvas-slope),
    (cx + tang-half, cy + offset-y + tang-half * canvas-slope),
    stroke: 2pt + rose,
    mark: (end: "stealth", fill: rose, size: 0.22),
  )

  line(
    (cx, cy),
    (cx, cy + offset-y),
    stroke: (dash: "dotted", paint: rose, thickness: 0.6pt),
  )

  content(
    (cx + tang-half + 0.15, cy + offset-y + tang-half * canvas-slope + 0.2),
    anchor: "west",
    text(fill: rose, size: 9pt, weight: "bold")[$f'_"center"$],
  )

  // Bandwidth bracket below bars
  let brace-y = -0.5
  line(
    (wx0, brace-y),
    (wx1, brace-y),
    stroke: 0.8pt + black90,
    mark: (start: "|", end: "|"),
  )
  content(
    ((wx0 + wx1) / 2, brace-y - 0.35),
    text(fill: black90, size: 8.5pt)[bandwidth $h$],
  )
})
