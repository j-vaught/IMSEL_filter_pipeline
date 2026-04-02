#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

// ============================================================
// CONFIGURABLE PARAMETERS — change these and everything updates
// ============================================================
#let all-heights = (1.0, 1.4, 0.9, 1.6, 2.0, 2.4, 2.1, 2.7, 2.5, 3.0, 3.2)
#let m = 3            // half-width → window = 2m+1 = 7
#let d = 3            // polynomial degree
#let win-center = 5   // index of center bar in all-heights (0-based)

// ============================================================
// DERIVED VALUES
// ============================================================
#let n = all-heights.len()
#let win-start = win-center - m
#let win-end = win-center + m
#let win-size = 2 * m + 1

// Extract windowed data
#let win-data = all-heights.slice(win-start, win-end + 1)

// ============================================================
// LEAST-SQUARES POLYNOMIAL FIT (computed in Typst)
// ============================================================
// Local x coords: -m, -m+1, ..., 0, ..., m
// Build Vandermonde matrix V[i][j] = x_i^j, then solve (V^T V) c = V^T y

// Helper: dot product of two arrays
#let dot(a, b) = {
  let s = 0.0
  for i in range(a.len()) { s += a.at(i) * b.at(i) }
  s
}

// Build Vandermonde columns: col_j = (x_0^j, x_1^j, ..., x_{2m}^j)
#let vander-cols = {
  let cols = ()
  for j in range(d + 1) {
    let col = ()
    for i in range(win-size) {
      let xi = i - m  // local coord
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

// Gram matrix G = V^T V, where G[j][k] = dot(col_j, col_k)
#let gram = {
  let g = ()
  for j in range(d + 1) {
    let row = ()
    for k in range(d + 1) {
      row.push(dot(vander-cols.at(j), vander-cols.at(k)))
    }
    g.push(row)
  }
  g
}

// Right-hand side: rhs[j] = dot(col_j, win-data)
#let rhs = {
  let r = ()
  for j in range(d + 1) {
    r.push(dot(vander-cols.at(j), win-data))
  }
  r
}

// Solve Gc = rhs via Gaussian elimination with partial pivoting
#let solve-system(A, b) = {
  let n = b.len()
  // Build augmented matrix
  let aug = ()
  for i in range(n) {
    let row = A.at(i) + (b.at(i),)
    aug.push(row)
  }
  // Forward elimination
  for col in range(n) {
    // Find pivot
    let max-val = calc.abs(aug.at(col).at(col))
    let max-row = col
    for row in range(col + 1, n) {
      let v = calc.abs(aug.at(row).at(col))
      if v > max-val { max-val = v; max-row = row }
    }
    // Swap rows
    if max-row != col {
      let tmp = aug.at(col)
      aug.at(col) = aug.at(max-row)
      aug.at(max-row) = tmp
    }
    // Eliminate below
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
  // Back substitution
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

// Evaluate polynomial at local coord xi: p(xi) = sum_j coeffs[j] * xi^j
#let poly-eval(xi) = {
  let val = 0.0
  let xp = 1.0
  for j in range(d + 1) {
    val += coeffs.at(j) * xp
    xp *= xi
  }
  val
}

// Derivative at xi=0 is simply coeffs[1] (the linear coefficient)
#let deriv-at-center = coeffs.at(1)

// Compute fitted values at each window position
#let fitted = {
  let vals = ()
  for i in range(win-size) {
    vals.push(poly-eval(i - m))
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

  // Draw all bars
  for i in range(n) {
    let x = i * step
    let h = all-heights.at(i)
    let clr = if i >= win-start and i <= win-end { garnet } else { black50 }
    rect(
      (x, 0),
      (x + bar-w, h),
      fill: clr,
      stroke: 0.4pt + black90,
    )
  }

  // Draw fitted polynomial curve
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
    text(fill: atlantic, size: 8.5pt)[fitted polynomial],
  )

  // Tangent arrow at center — uses actual derivative
  let cx = win-center * step + bar-w / 2
  let cy = fitted.at(m)  // center of window = index m in fitted array
  let canvas-slope = deriv-at-center  // slope per integer step = per `step` canvas units

  let tang-half = 0.7
  let offset-y = 0.5
  line(
    (cx - tang-half, cy + offset-y - tang-half * canvas-slope),
    (cx + tang-half, cy + offset-y + tang-half * canvas-slope),
    stroke: 2pt + rose,
    mark: (end: "stealth", fill: rose, size: 0.22),
  )

  // Dashed connector from curve to tangent
  line(
    (cx, cy),
    (cx, cy + offset-y),
    stroke: (dash: "dotted", paint: rose, thickness: 0.6pt),
  )

  // Label derivative
  content(
    (cx + tang-half + 0.15, cy + offset-y + tang-half * canvas-slope + 0.2),
    anchor: "west",
    text(fill: rose, size: 9pt, weight: "bold")[$f'_"center"$],
  )

  // Window bracket below
  let brace-y = -0.5
  line(
    (wx0, brace-y),
    (wx1, brace-y),
    stroke: 0.8pt + black90,
    mark: (start: "|", end: "|"),
  )
  content(
    ((wx0 + wx1) / 2, brace-y - 0.35),
    text(fill: black90, size: 8.5pt)[window $= 2m + 1$],
  )
})
