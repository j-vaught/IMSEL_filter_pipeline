#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 8pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let grass = rgb("#CED318")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

// ============================================================
// CONFIGURABLE: 5x5 test grid — only 4 corners have data, rest = 0
// Each corner gets a unique color for debugging draw order
// ============================================================
// 9x9 grid: outer 2-cell ring is gray context, inner 5x5 is the S-G window
#let pixel-data = (
  (250, 248, 245, 240, 230, 200, 160, 120, 90),
  (248, 245, 240, 235, 220, 180, 140, 100, 78),
  (245, 240, 220, 210, 190, 140,  90,  75,  65),
  (240, 235, 215, 200, 160, 110,  80,  65,  55),
  (235, 225, 210, 180, 145, 100,  70,  55,  45),
  (220, 200, 170, 140, 105,  75,  60,  48,  40),
  (190, 160, 130, 110,  80,  65,  55,  42,  35),
  (150, 130, 105,  85,  70,  58,  48,  38,  30),
  (120, 105,  90,  75,  60,  50,  42,  32,  25),
)
#let grid-n = 9

// Colors: gray for outer 2-cell ring, garnet for inner 5x5, horseshoe for center
#let get-color(r, c) = {
  if r >= 2 and r <= 6 and c >= 2 and c <= 6 {
    if r == 4 and c == 4 { horseshoe } else { garnet }
  } else {
    black50
  }
}

// ============================================================
// 2D POLYNOMIAL FIT (degree 2) — computed in Typst
// ============================================================
// Local coords: x,y in {-2,-1,0,1,2}
// Monomials for d=2: 1, x, y, x², y², xy → M=6

#let dot(a, b) = {
  let s = 0.0
  for i in range(a.len()) { s += a.at(i) * b.at(i) }
  s
}

// Flatten only the inner 5x5 window (rows 2-6, cols 2-6) for polynomial fit
// Local coords centered at (4,4) → x,y in {-2,-1,0,1,2}
#let win-r0 = 2
#let win-r1 = 6
#let win-c0 = 2
#let win-c1 = 6
#let win-center-r = 4
#let win-center-c = 4

#let all-x = ()
#let all-y = ()
#let all-f = ()
#for r in range(win-r0, win-r1 + 1) {
  for c in range(win-c0, win-c1 + 1) {
    all-x.push(float(c - win-center-c))
    all-y.push(float(win-center-r - r))  // y increases upward
    all-f.push(float(pixel-data.at(r).at(c)))
  }
}
#let N = all-x.len()

// Build monomial columns: 1, x, y, x², y², xy
#let mono-cols = {
  let cols = ((), (), (), (), (), ())
  for i in range(N) {
    let x = all-x.at(i)
    let y = all-y.at(i)
    cols.at(0).push(1.0)
    cols.at(1).push(x)
    cols.at(2).push(y)
    cols.at(3).push(x * x)
    cols.at(4).push(y * y)
    cols.at(5).push(x * y)
  }
  cols
}
#let M = 6

// Gram matrix
#let gram = {
  let g = ()
  for j in range(M) {
    let row = ()
    for k in range(M) {
      row.push(dot(mono-cols.at(j), mono-cols.at(k)))
    }
    g.push(row)
  }
  g
}

// RHS
#let rhs = {
  let r = ()
  for j in range(M) {
    r.push(dot(mono-cols.at(j), all-f))
  }
  r
}

// Gaussian elimination solver
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

// Skip polynomial fit for grids smaller than 3x3 (not enough data for 6 unknowns)
#let coeffs = if N >= M { solve-system(gram, rhs) } else { (0.0,) * M }
// coeffs = (c00, c10, c01, c20, c02, c11)
// df/dx at center = c10 = coeffs.at(1)
// df/dy at center = c01 = coeffs.at(2)
#let dfdx = coeffs.at(1)
#let dfdy = coeffs.at(2)

// Evaluate fitted surface at (x, y)
#let poly2d(x, y) = {
  coeffs.at(0) + coeffs.at(1) * x + coeffs.at(2) * y + coeffs.at(3) * x * x + coeffs.at(4) * y * y + coeffs.at(5) * x * y
}

// ============================================================
// ISOMETRIC PROJECTION (manual, correct painter's algorithm)
// ============================================================
// View from front-left: x goes right, y goes into screen, z goes up
// Painter's order: draw far objects first (high y, low x) → near last (low y, high x)
#let scale-xy = 0.75
#let scale-z = 0.016
#let cell-size = 1.0
#let col-gap = 0.06

// Isometric: x-right tilts right+down, y-depth tilts left+down, z is straight up
#let project(gx, gy, gz) = {
  let cx = scale-xy * (gx * 0.866 - gy * 0.866)  // cos(30°)
  let cy = scale-xy * (gx * 0.5 + gy * 0.5) + scale-z * gz  // sin(30°) + height
  (cx, cy)
}

#cetz.canvas({
  import cetz.draw: *

  // Painter's algorithm: diagonal sweep by depth key (r - c).
  // Depth key ranges from -(N-1) (back-right, farthest) to +(N-1) (front-left, closest).
  // Within each diagonal, lower r draws first (further back).
  //
  // For a 5x5 grid the full order is:
  //   d=-4: (0,4)                          ← farthest (back-right)
  //   d=-3: (0,3), (1,4)
  //   d=-2: (0,2), (1,3), (2,4)
  //   ...
  //   d= 0: (0,0), (1,1), (2,2), (3,3), (4,4)
  //   ...
  //   d=+4: (4,0)                          ← closest (front-left)
  //
  // Draw columns (all flat at height 5 for wireframe testing)
  for d in range(-(grid-n - 1), grid-n) {
    for r in range(grid-n) {
      let c = r - d
      if c < 0 or c >= grid-n { continue }
      let gx = float(c) * cell-size
      let gy = float(grid-n - 1 - r) * cell-size
      let val = float(pixel-data.at(r).at(c))
      let h = val

      if h < 1 { continue }

      let base-clr = get-color(r, c)

      let x0 = gx + col-gap
      let y0 = gy + col-gap
      let x1 = gx + cell-size - col-gap
      let y1 = gy + cell-size - col-gap

      let p000 = project(x0, y0, 0)
      let p100 = project(x1, y0, 0)
      let p010 = project(x0, y1, 0)
      let p110 = project(x1, y1, 0)
      let p001 = project(x0, y0, h)
      let p101 = project(x1, y0, h)
      let p011 = project(x0, y1, h)
      let p111 = project(x1, y1, h)

      // 1. Left face
      line(p000, p010, p011, p001, close: true,
        fill: base-clr.lighten(72%),
        stroke: 0.4pt + black70)

      // 2. Front face
      line(p000, p100, p101, p001, close: true,
        fill: base-clr.lighten(65%),
        stroke: 0.4pt + black70)

      // 3. Top face
      line(p001, p101, p111, p011, close: true,
        fill: base-clr.lighten(60%),
        stroke: 0.4pt + black70)
    }
  }

  // ============================================================
  // WIREFRAME: fitted surface over the inner 5x5 window
  // ============================================================
  // Coordinate mapping:
  //   x-local ∈ [-2, +2] → gx = (win-center-c + x-local) * cell-size + cell-size/2
  //   y-local ∈ [-2, +2] → gy = (grid-n - 1 - win-center-r + y-local) * cell-size + cell-size/2
  //   z = poly2d(x-local, y-local)
  //
  // Helper to convert local coords to 3D projected point:
  let local-to-screen(xl, yl) = {
    let gx = (float(win-center-c) + xl) * cell-size + cell-size / 2
    let gy = (float(grid-n - 1 - win-center-r) + yl) * cell-size + cell-size / 2
    let z = poly2d(xl, yl)
    project(gx, gy, z)
  }

  let surf-n = 20  // subdivisions per line
  let half-w = 2   // half-width of window in local coords

  // Lines along x-direction (one per window row, at y-local = -2, -1, 0, 1, 2)
  for row in range(2 * half-w + 1) {
    let yl = float(row - half-w)
    for i in range(surf-n) {
      let xl0 = -float(half-w) + float(2 * half-w) * float(i) / float(surf-n)
      let xl1 = -float(half-w) + float(2 * half-w) * float(i + 1) / float(surf-n)
      let p0 = local-to-screen(xl0, yl)
      let p1 = local-to-screen(xl1, yl)
      line(p0, p1, stroke: 0.8pt + horseshoe.transparentize(30%))
    }
  }

  // Lines along y-direction (one per window column, at x-local = -2, -1, 0, 1, 2)
  for col in range(2 * half-w + 1) {
    let xl = float(col - half-w)
    for i in range(surf-n) {
      let yl0 = -float(half-w) + float(2 * half-w) * float(i) / float(surf-n)
      let yl1 = -float(half-w) + float(2 * half-w) * float(i + 1) / float(surf-n)
      let p0 = local-to-screen(xl, yl0)
      let p1 = local-to-screen(xl, yl1)
      line(p0, p1, stroke: 0.8pt + horseshoe.transparentize(30%))
    }
  }

  // Gradient arrows at center pixel on the fitted surface
  let center-z = poly2d(0, 0)
  let cp = local-to-screen(0, 0)

  let arrow-len = 2.5

  // df/dx arrow: move along x-local, z changes by dfdx
  let dx-end-gx = (float(win-center-c) + arrow-len) * cell-size + cell-size / 2
  let dx-end-gy = (float(grid-n - 1 - win-center-r)) * cell-size + cell-size / 2
  let dx-end = project(dx-end-gx, dx-end-gy, center-z + dfdx * arrow-len)
  line(cp, dx-end, stroke: 2.5pt + horseshoe, mark: (end: "stealth", fill: horseshoe, size: 0.25))
  content(
    (dx-end.at(0) + 0.1, dx-end.at(1) - 0.2),
    anchor: "west",
    text(fill: black90, size: 15pt, weight: "bold")[$partial f \/ partial x$])

  // df/dy arrow: move along y-local, z changes by dfdy
  let dy-end-gx = float(win-center-c) * cell-size + cell-size / 2
  let dy-end-gy = (float(grid-n - 1 - win-center-r) + arrow-len) * cell-size + cell-size / 2
  let dy-end = project(dy-end-gx, dy-end-gy, center-z + dfdy * arrow-len)
  line(cp, dy-end, stroke: 2.5pt + horseshoe, mark: (end: "stealth", fill: horseshoe, size: 0.25))
  content(
    (dy-end.at(0) - 0.1, dy-end.at(1) + 0.2),
    anchor: "east",
    text(fill: black90, size: 15pt, weight: "bold")[$partial f \/ partial y$])

  // Center dot
  circle(cp, radius: 0.1, fill: horseshoe, stroke: 1pt + white)
})
