#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90  = rgb("#363636")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let black10  = rgb("#ECECEC")

// ── Parameters (paper's stated values) ──────────────────────────────────
#let poly-d    = 4              // polynomial degree
#let target-Np = 100            // neighbor count
#let m-line    = 7              // line half-length (15 positions)
#let sigma-l   = m-line / 2.0   // line Gaussian σ_ℓ
#let grid-N    = 29             // display grid size
#let cell-sz   = 0.2            // cell size
#let half      = calc.div-euclid(grid-N - 1, 2)

// ── Factorial ───────────────────────────────────────────────────────────
#let fact(n) = {
  if n <= 0 { 1 } else if n == 1 { 1 } else if n == 2 { 2 }
  else if n == 3 { 6 } else { 24 }
}

// ── Select N_p closest integer pixels ───────────────────────────────────
#let nbr-data = {
  let max-r = calc.ceil(calc.sqrt(target-Np / calc.pi)) + 3
  let cands = ()
  for dxi in range(2 * max-r + 1) {
    let dx = dxi - max-r
    for dyi in range(2 * max-r + 1) {
      let dy = dyi - max-r
      let d2 = dx * dx + dy * dy
      cands.push((dx, dy, d2))
    }
  }
  cands = cands.sorted(key: c => c.at(2))
  let pts = ()
  let max-d2 = 0
  for i in range(calc.min(target-Np, cands.len())) {
    pts.push((cands.at(i).at(0), cands.at(i).at(1)))
    if cands.at(i).at(2) > max-d2 { max-d2 = cands.at(i).at(2) }
  }
  (pts: pts, radius: calc.sqrt(max-d2))
}
#let neighbors = nbr-data.pts
#let nbr-radius = nbr-data.radius
#let Np = neighbors.len()

// ── Monomial basis for d=4 ──────────────────────────────────────────────
#let monomial-basis(x, y) = {
  let b = ()
  for deg in range(poly-d + 1) {
    for p in range(deg + 1) {
      let q = deg - p
      let xp = if p == 0 { 1.0 } else { calc.pow(x, p) }
      let yq = if q == 0 { 1.0 } else { calc.pow(y, q) }
      b.push(xp * yq / (fact(p) * fact(q)))
    }
  }
  b
}
#let M = monomial-basis(1.0, 1.0).len()

#let line-weight(j) = if m-line == 0 {
  if j == 0 { 1.0 } else { 0.0 }
} else {
  calc.exp(-(j * j) / (2.0 * sigma-l * sigma-l))
}

// ── Matrix utilities ────────────────────────────────────────────────────
#let mat-transpose(A) = {
  let nr = A.len()
  let nc = A.at(0).len()
  let result = ()
  for j in range(nc) {
    let row = ()
    for i in range(nr) { row.push(A.at(i).at(j)) }
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
      for k in range(ac) { s = s + A.at(i).at(k) * B.at(k).at(j) }
      row.push(s)
    }
    result.push(row)
  }
  result
}

#let mat-invert(A) = {
  let n = A.len()
  let aug = ()
  for i in range(n) {
    let row = ()
    for j in range(n) { row.push(A.at(i).at(j) * 1.0) }
    for j in range(n) { row.push(if i == j { 1.0 } else { 0.0 }) }
    aug.push(row)
  }
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
  let inv = ()
  for i in range(n) {
    let row = ()
    for j in range(n) { row.push(aug.at(i).at(n + j)) }
    inv.push(row)
  }
  inv
}

// ── Gradient row ────────────────────────────────────────────────────────
#let gradient-row(theta) = {
  let ct = calc.cos(theta)
  let st = calc.sin(theta)
  let A = ()
  for ii in range(Np) {
    let dx = neighbors.at(ii).at(0)
    let dy = neighbors.at(ii).at(1)
    let x = dx * ct + dy * st
    let y = -dx * st + dy * ct
    A.push(monomial-basis(x, y))
  }
  let At = mat-transpose(A)
  let AtA = mat-mul(At, A)
  let AtA-inv = mat-invert(AtA)
  let P = mat-mul(AtA-inv, At)
  // Basis order is (1, y, x, ...), so the normal derivative d/dx' is row 2.
  P.at(2)
}

// ── Fused stencil (takes precomputed gradient row) ──────────────────────
// theta = normal direction; line extends perpendicular (along tangent)
#let fused-stencil-from(p, theta) = {
  let tx = -calc.sin(theta)
  let ty = calc.cos(theta)
  let stencil = (:)
  for jj in range(2 * m-line + 1) {
    let j = jj - m-line
    let wj = line-weight(j)
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

// ── Compute η from precomputed gradient row and stencil ─────────────────
#let compute-eta-from(p, stencil) = {
  let numer = 0.0
  for (_, val) in stencil { numer = numer + calc.abs(val) }
  let sum-wj = 0.0
  for jj in range(2 * m-line + 1) {
    let j = jj - m-line
    sum-wj = sum-wj + line-weight(j)
  }
  let sum-abs-p = 0.0
  for ii in range(Np) { sum-abs-p = sum-abs-p + calc.abs(p.at(ii)) }
  numer / (sum-wj * sum-abs-p)
}

// ── Precompute everything (gradient row computed once per orientation) ───
#let orientations = (0, 30, 60, 90)

// theta = normal direction; gradient-row rotates by theta, line extends along theta+90°
#let all-data = orientations.map(deg => {
  let theta = deg * 1deg
  let p = gradient-row(theta)
  let st = fused-stencil-from(p, theta)
  let eta = compute-eta-from(p, st)
  (stencil: st, eta: eta)
})

#let stencils = all-data.map(d => d.stencil)
#let etas = all-data.map(d => d.eta)

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

#let draw-panel(ox, oy, stencil, theta-deg, eta-val) = {
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
        if w > 0.0001 { lerp-color(garnet, intensity) }
        else if w < -0.0001 { lerp-color(atlantic, intensity) }
        else { white }
      } else { white }
      rect((x, y), (x + cell-sz, y - cell-sz), fill: fc, stroke: 0.24pt + black30)
    }
  }
  for r in range(grid-N) {
    for c in range(grid-N) {
      let x = ox + c * cell-sz
      let y = oy - r * cell-sz
      let dx = c - half
      let dy = -(r - half)
      let key = str(dx) + "," + str(dy)
      if key in stencil {
        let left-key = str(dx - 1) + "," + str(dy)
        let right-key = str(dx + 1) + "," + str(dy)
        let top-key = str(dx) + "," + str(dy + 1)
        let bottom-key = str(dx) + "," + str(dy - 1)
        if left-key not in stencil {
          line((x, y), (x, y - cell-sz), stroke: 0.72pt + black90)
        }
        if right-key not in stencil {
          line((x + cell-sz, y), (x + cell-sz, y - cell-sz), stroke: 0.72pt + black90)
        }
        if top-key not in stencil {
          line((x, y), (x + cell-sz, y), stroke: 0.72pt + black90)
        }
        if bottom-key not in stencil {
          line((x, y - cell-sz), (x + cell-sz, y - cell-sz), stroke: 0.72pt + black90)
        }
      }
    }
  }
  rect((ox, oy), (ox + grid-N * cell-sz, oy - grid-N * cell-sz), stroke: 0.9pt + black90)
  let cx = ox + grid-N * cell-sz / 2.0
  content((cx, oy - grid-N * cell-sz - 0.35),
    text(fill: black90, size: 10.5pt, weight: "bold")[
      #math.equation(block: false, [#math.theta #math.eq #str(theta-deg) #math.degree])
    ])
  let eta-pct = str(calc.round(eta-val * 100, digits: 1))
  content((cx, oy - grid-N * cell-sz - 0.7),
    text(fill: black50, size: 8.5pt, style: "italic")[
      $eta$ = #eta-pct%
    ])
}

// ── Render ──────────────────────────────────────────────────────────────
#cetz.canvas({
  import cetz.draw: *
  let gw = grid-N * cell-sz
  let gx = 0.6
  let n-cols = orientations.len()
  for i in range(orientations.len()) {
    let col = i
    let ox = col * (gw + gx)
    let oy = 0
    draw-panel(ox, oy, stencils.at(i), orientations.at(i), etas.at(i))
  }
  // Diverging gradient colorbar
  let tw = n-cols * gw + (n-cols - 1) * gx
  let ly = -gw - 1.1
  let bar-w = 5.0
  let bar-h = 0.25
  let n-steps = 40
  let step-w = bar-w / n-steps
  let bar-x = tw / 2.0 - bar-w / 2.0
  for s in range(n-steps) {
    let t = s / (n-steps - 1)
    let fc = if t < 0.5 {
      lerp-color(atlantic, 1.0 - t * 2.0)
    } else {
      lerp-color(garnet, (t - 0.5) * 2.0)
    }
    rect((bar-x + s * step-w, ly), (bar-x + (s + 1) * step-w, ly - bar-h),
      fill: fc, stroke: none)
  }
  rect((bar-x, ly), (bar-x + bar-w, ly - bar-h), stroke: 0.8pt + black90)
  content((bar-x, ly - bar-h - 0.25),
    text(fill: black90, size: 8.5pt)[$-alpha_max$])
  content((bar-x + bar-w / 2.0, ly - bar-h - 0.25),
    text(fill: black90, size: 8.5pt)[$0$])
  content((bar-x + bar-w, ly - bar-h - 0.25),
    text(fill: black90, size: 8.5pt)[$+alpha_max$])
})
