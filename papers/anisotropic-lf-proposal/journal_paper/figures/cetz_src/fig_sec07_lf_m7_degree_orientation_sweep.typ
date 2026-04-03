#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90  = rgb("#363636")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let white    = rgb("#FFFFFF")

// ── Parameters ───────────────────────────────────────────────────────────
#let degrees = (2, 3, 4, 5, 6, 7)
#let orientations = (0, 30, 60, 90)
#let target-Np = 500
#let m-line = 13
#let sigma-l = m-line / 2.0
#let grid-N = 49
#let cell-sz = 0.065
#let half = calc.div-euclid(grid-N - 1, 2)

// ── Factorial ────────────────────────────────────────────────────────────
#let fact(n) = {
  if n <= 0 { 1 }
  else if n == 1 { 1 }
  else if n == 2 { 2 }
  else if n == 3 { 6 }
  else if n == 4 { 24 }
  else if n == 5 { 120 }
  else if n == 6 { 720 }
  else if n == 7 { 5040 }
  else if n == 8 { 40320 }
  else { 362880 }
}

// ── Matrix utilities ─────────────────────────────────────────────────────
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

// ── Basis, support, and fused LF construction ───────────────────────────
#let monomial-basis(d, x, y) = {
  let b = ()
  for deg in range(d + 1) {
    for p in range(deg + 1) {
      let q = deg - p
      let xp = if p == 0 { 1.0 } else { calc.pow(x, p) }
      let yq = if q == 0 { 1.0 } else { calc.pow(y, q) }
      b.push(xp * yq / (fact(p) * fact(q)))
    }
  }
  b
}

#let neighbor-data(target-Np) = {
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
  for i in range(calc.min(target-Np, cands.len())) {
    pts.push((cands.at(i).at(0), cands.at(i).at(1)))
  }
  pts
}

#let neighbors = neighbor-data(target-Np)
#let Np = neighbors.len()

#let line-weight(j) = if m-line == 0 {
  if j == 0 { 1.0 } else { 0.0 }
} else {
  calc.exp(-(j * j) / (2.0 * sigma-l * sigma-l))
}

#let gradient-row(d, theta) = {
  let A = ()
  let ct = calc.cos(theta)
  let st = calc.sin(theta)
  for ii in range(Np) {
    let dx = neighbors.at(ii).at(0)
    let dy = neighbors.at(ii).at(1)
    let x = dx * ct + dy * st
    let y = -dx * st + dy * ct
    A.push(monomial-basis(d, x, y))
  }
  let At = mat-transpose(A)
  let AtA = mat-mul(At, A)
  let AtA-inv = mat-invert(AtA)
  let P = mat-mul(AtA-inv, At)
  P.at(2)
}

#let fused-stencil(d, theta) = {
  let p = gradient-row(d, theta)
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
  let max-abs = {
    let mx = 0.0
    for (_, val) in stencil {
      let v = calc.abs(val)
      if v > mx { mx = v }
    }
    mx
  }
  (stencil: stencil, max-abs: max-abs)
}

#let all-data = degrees.map(d =>
  orientations.map(deg => {
    let theta = deg * 1deg
    (d: d, theta-deg: deg) + fused-stencil(d, theta)
  })
)

// ── Drawing helpers ──────────────────────────────────────────────────────
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let draw-panel(ox, oy, data) = {
  import cetz.draw: *
  let local-max = data.max-abs
  let stencil = data.stencil
  for r in range(grid-N) {
    for c in range(grid-N) {
      let x = ox + c * cell-sz
      let y = oy - r * cell-sz
      let dx = c - half
      let dy = -(r - half)
      let key = str(dx) + "," + str(dy)
      let fc = if key in stencil {
        let w = stencil.at(key)
        let intensity = calc.abs(w) / local-max
        if w > 0.0001 { lerp-color(garnet, intensity) }
        else if w < -0.0001 { lerp-color(atlantic, intensity) }
        else { white }
      } else { white }
      rect((x, y), (x + cell-sz, y - cell-sz), fill: fc, stroke: none)
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
          line((x, y), (x, y - cell-sz), stroke: 0.4pt + black90)
        }
        if right-key not in stencil {
          line((x + cell-sz, y), (x + cell-sz, y - cell-sz), stroke: 0.4pt + black90)
        }
        if top-key not in stencil {
          line((x, y), (x + cell-sz, y), stroke: 0.4pt + black90)
        }
        if bottom-key not in stencil {
          line((x, y - cell-sz), (x + cell-sz, y - cell-sz), stroke: 0.4pt + black90)
        }
      }
    }
  }
  rect((ox, oy), (ox + grid-N * cell-sz, oy - grid-N * cell-sz), stroke: 0.5pt + black90)
}

// ── Render ───────────────────────────────────────────────────────────────
#cetz.canvas({
  import cetz.draw: *
  let gw = grid-N * cell-sz
  let gx = 0.18
  let gy = 0.32
  let left-pad = 0.75
  let top-pad = 0.55
  let row-w = orientations.len() * gw + (orientations.len() - 1) * gx

  for col in range(orientations.len()) {
    let cx = left-pad + col * (gw + gx) + gw / 2.0
    content((cx, top-pad + 0.25),
      text(fill: black90, size: 9pt, weight: "bold")[
        $theta$ = #str(orientations.at(col))#sym.degree
      ])
  }

  for row in range(degrees.len()) {
    let cy = top-pad - row * (gw + gy) - gw / 2.0
    content((0.08, cy),
      text(fill: black90, size: 9pt, weight: "bold")[
        $d$ = #str(degrees.at(row))
      ],
      anchor: "west")
    for col in range(orientations.len()) {
      let ox = left-pad + col * (gw + gx)
      let oy = top-pad - row * (gw + gy)
      draw-panel(ox, oy, all-data.at(row).at(col))
    }
  }

  let ly = top-pad - degrees.len() * (gw + gy) + gy - 0.35
  let bar-w = 3.8
  let bar-h = 0.18
  let n-steps = 40
  let step-w = bar-w / n-steps
  let bar-x = left-pad + row-w / 2.0 - bar-w / 2.0
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
  rect((bar-x, ly), (bar-x + bar-w, ly - bar-h), stroke: 0.5pt + black90)
  content((bar-x, ly - bar-h - 0.2), text(fill: black90, size: 7.5pt)[$-alpha_max$])
  content((bar-x + bar-w / 2.0, ly - bar-h - 0.2), text(fill: black90, size: 7.5pt)[$0$])
  content((bar-x + bar-w, ly - bar-h - 0.2), text(fill: black90, size: 7.5pt)[$+alpha_max$])
})
