#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90  = rgb("#363636")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let black10  = rgb("#ECECEC")

// ── Filter parameters ───────────────────────────────────────────────────
#let radius  = 3.0           // neighborhood radius (N_p ≈ 29)
#let m-line  = 4             // line half-length (9 line positions)
#let sigma-l = m-line / 2.0  // line Gaussian σ_ℓ
#let grid-N  = 15            // display grid size
#let cell-sz = 0.35          // cell size
#let half    = calc.div-euclid(grid-N - 1, 2)

// ── Circular neighborhood ───────────────────────────────────────────────
#let neighbors = {
  let pts = ()
  let r2 = radius * radius
  let ri = calc.ceil(radius)
  for dxi in range(2 * ri + 1) {
    let dx = dxi - ri
    for dyi in range(2 * ri + 1) {
      let dy = dyi - ri
      if dx * dx + dy * dy <= r2 {
        pts.push((dx, dy))
      }
    }
  }
  pts
}
#let Np = neighbors.len()

// ── Matrix utilities ────────────────────────────────────────────────────

#let mat-transpose(A) = {
  let nr = A.len()
  let nc = A.at(0).len()
  let result = ()
  for j in range(nc) {
    let row = ()
    for i in range(nr) {
      row.push(A.at(i).at(j))
    }
    result.push(row)
  }
  result
}

#let mat-mul(A, B) = {
  let ar = A.len()
  let ac = A.at(0).len()
  let bc = B.at(0).len()
  let result = ()
  for i in range(ar) {
    let row = ()
    for j in range(bc) {
      let s = 0.0
      for k in range(ac) {
        s = s + A.at(i).at(k) * B.at(k).at(j)
      }
      row.push(s)
    }
    result.push(row)
  }
  result
}

#let mat-invert(A) = {
  let n = A.len()
  // Augmented matrix [A | I]
  let aug = ()
  for i in range(n) {
    let row = ()
    for j in range(n) { row.push(A.at(i).at(j) * 1.0) }
    for j in range(n) { row.push(if i == j { 1.0 } else { 0.0 }) }
    aug.push(row)
  }
  // Gauss-Jordan with partial pivoting
  for k in range(n) {
    let max-val = calc.abs(aug.at(k).at(k))
    let max-row = k
    for i in range(k + 1, n) {
      let v = calc.abs(aug.at(i).at(k))
      if v > max-val { max-val = v; max-row = i }
    }
    if max-row != k {
      let tmp = aug.at(k)
      aug.at(k) = aug.at(max-row)
      aug.at(max-row) = tmp
    }
    let pivot = aug.at(k).at(k)
    let rk = aug.at(k)
    for j in range(2 * n) { rk.at(j) = rk.at(j) / pivot }
    aug.at(k) = rk
    for i in range(n) {
      if i != k {
        let fac = aug.at(i).at(k)
        let ri = aug.at(i)
        for j in range(2 * n) { ri.at(j) = ri.at(j) - fac * aug.at(k).at(j) }
        aug.at(i) = ri
      }
    }
  }
  // Extract right half
  let inv = ()
  for i in range(n) {
    let row = ()
    for j in range(n) { row.push(aug.at(i).at(n + j)) }
    inv.push(row)
  }
  inv
}

// ── Pseudoinverse gradient row ──────────────────────────────────────────

#let gradient-row(theta) = {
  let ct = calc.cos(theta)
  let st = calc.sin(theta)
  // Design matrix A (Np × 6) for d = 2
  // Row i = (1, x', y', x'²/2, y'²/2, x'y')
  let A = ()
  for ii in range(Np) {
    let dx = neighbors.at(ii).at(0)
    let dy = neighbors.at(ii).at(1)
    let x = dx * ct + dy * st
    let y = -dx * st + dy * ct
    A.push((1.0, x, y, x * x / 2.0, y * y / 2.0, x * y))
  }
  let At = mat-transpose(A)
  let AtA = mat-mul(At, A)
  let AtA-inv = mat-invert(AtA)
  let P = mat-mul(AtA-inv, At)
  P.at(1) // row 1 = f_x (normal derivative)
}

// ── Fused stencil ───────────────────────────────────────────────────────

#let fused-stencil(theta) = {
  let p = gradient-row(theta)
  // Tangent = θ + π/2
  let tx = -calc.sin(theta)
  let ty = calc.cos(theta)
  let stencil = (:)
  for jj in range(2 * m-line + 1) {
    let j = jj - m-line
    let wj = calc.exp(-(j * j) / (2.0 * sigma-l * sigma-l))
    let lx = j * tx
    let ly = j * ty
    for ii in range(Np) {
      let dx = neighbors.at(ii).at(0)
      let dy = neighbors.at(ii).at(1)
      let ox = calc.round(lx + dx)
      let oy = calc.round(ly + dy)
      let key = str(ox) + "," + str(oy)
      let contrib = wj * p.at(ii)
      if key in stencil {
        stencil.insert(key, stencil.at(key) + contrib)
      } else {
        stencil.insert(key, contrib)
      }
    }
  }
  stencil
}

// ── Precompute ──────────────────────────────────────────────────────────
#let orientations = (0, 30, 60, 90, 120, 150)
#let stencils = orientations.map(deg => fused-stencil(deg * 1deg))

#let global-max = {
  let mx = 0.0
  for st in stencils {
    for (_, val) in st {
      let v = calc.abs(val)
      if v > mx { mx = v }
    }
  }
  mx
}

// ── Drawing helpers ─────────────────────────────────────────────────────
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let draw-panel(ox, oy, stencil, theta-deg) = {
  import cetz.draw: *
  for r in range(grid-N) {
    for c in range(grid-N) {
      let x = ox + c * cell-sz
      let y = oy - r * cell-sz
      let dx = c - half
      let dy = -(r - half)
      let key = str(dx) + "," + str(dy)
      let fc = if key in stencil {
        let w = stencil.at(key)
        let intensity = calc.abs(w) / global-max
        if w > 0.001 { lerp-color(garnet, intensity) }
        else if w < -0.001 { lerp-color(atlantic, intensity) }
        else { white }
      } else { white }
      rect((x, y), (x + cell-sz, y - cell-sz), fill: fc, stroke: 0.2pt + black30)
    }
  }
  rect((ox, oy), (ox + grid-N * cell-sz, oy - grid-N * cell-sz), stroke: 0.8pt + black90)
  let cx = ox + grid-N * cell-sz / 2.0
  content((cx, oy - grid-N * cell-sz - 0.45),
    text(fill: black90, size: 10pt, weight: "bold")[
      #math.equation(block: false, [#math.theta #math.eq #str(theta-deg) #math.degree])
    ])
}

// ── Render ──────────────────────────────────────────────────────────────
#cetz.canvas({
  import cetz.draw: *
  let gw = grid-N * cell-sz
  let gx = 0.7
  let gy = 1.4
  for i in range(6) {
    let col = calc.rem(i, 3)
    let row = calc.div-euclid(i, 3)
    let ox = col * (gw + gx)
    let oy = -row * (gw + gy)
    draw-panel(ox, oy, stencils.at(i), orientations.at(i))
  }
  // Diverging gradient colorbar
  let tw = 3.0 * gw + 2.0 * gx
  let ly = -(gw + gy) - gw - 1.0
  let bar-w = 6.0
  let bar-h = 0.3
  let n-steps = 40
  let step-w = bar-w / n-steps
  let bar-x = tw / 2.0 - bar-w / 2.0

  for s in range(n-steps) {
    let t = s / (n-steps - 1)  // 0 to 1
    let fc = if t < 0.5 {
      let intensity = 1.0 - t * 2.0  // 1 → 0 as t goes 0 → 0.5
      lerp-color(atlantic, intensity)
    } else {
      let intensity = (t - 0.5) * 2.0  // 0 → 1 as t goes 0.5 → 1
      lerp-color(garnet, intensity)
    }
    rect((bar-x + s * step-w, ly), (bar-x + (s + 1) * step-w, ly - bar-h),
      fill: fc, stroke: none)
  }
  rect((bar-x, ly), (bar-x + bar-w, ly - bar-h), stroke: 0.5pt + black90)

  // Labels
  content((bar-x, ly - bar-h - 0.3),
    text(fill: black90, size: 7.5pt)[$-alpha_max$])
  content((bar-x + bar-w / 2.0, ly - bar-h - 0.3),
    text(fill: black90, size: 7.5pt)[$0$])
  content((bar-x + bar-w, ly - bar-h - 0.3),
    text(fill: black90, size: 7.5pt)[$+alpha_max$])
})
