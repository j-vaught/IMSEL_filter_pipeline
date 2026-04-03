#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let rose     = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90  = rgb("#363636")
#let black70  = rgb("#5C5C5C")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let black10  = rgb("#ECECEC")
#let white    = rgb("#FFFFFF")

// Parameters for the visual explanation.
#let theta = 30deg
#let d = 2
#let target-Np = 100
#let m-line = 7
#let sigma-l = m-line / 2.0

#let schem-n = 13
#let schem-cell = 0.28
#let schem-half = calc.div-euclid(schem-n - 1, 2)

#let map-n = 29
#let map-cell = 0.16
#let map-half = calc.div-euclid(map-n - 1, 2)

// ── Utilities ────────────────────────────────────────────────────────────
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

#let line-weight(j) = {
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

#let wvf-stencil(d, theta) = {
  let p = gradient-row(d, theta)
  let stencil = (:)
  for ii in range(Np) {
    let dx = neighbors.at(ii).at(0)
    let dy = neighbors.at(ii).at(1)
    stencil.insert(str(dx) + "," + str(dy), p.at(ii))
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

#let lf-stencil(d, theta) = {
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

// ── Drawing helpers ──────────────────────────────────────────────────────
#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let draw-cell-grid(ox, oy, n, cell, stroke-col) = {
  import cetz.draw: *
  for r in range(n) {
    for c in range(n) {
      let x = ox + c * cell
      let y = oy - r * cell
      rect((x, y), (x + cell, y - cell), fill: white, stroke: 0.18pt + stroke-col)
    }
  }
}

#let draw-weight-map(ox, oy, n, cell, stencil, max-abs) = {
  import cetz.draw: *
  for r in range(n) {
    for c in range(n) {
      let x = ox + c * cell
      let y = oy - r * cell
      let dx = c - if n == schem-n { schem-half } else { map-half }
      let dy = -(r - if n == schem-n { schem-half } else { map-half })
      let key = str(dx) + "," + str(dy)
      let fc = if key in stencil {
        let w = stencil.at(key)
        let intensity = calc.abs(w) / max-abs
        if w > 0.0001 { lerp-color(garnet, intensity) }
        else if w < -0.0001 { lerp-color(atlantic, intensity) }
        else { white }
      } else { white }
      rect((x, y), (x + cell, y - cell), fill: fc, stroke: 0.08pt + black30)
    }
  }
  rect((ox, oy), (ox + n * cell, oy - n * cell), stroke: 0.45pt + black90)
}

#let box-label(x, y, w, h, lab) = {
  import cetz.draw: *
  rect((x, y), (x + w, y - h), fill: black10, stroke: 0.7pt + black90)
  content((x + w / 2, y - h / 2), text(fill: black90, size: 8.3pt, weight: "bold")[#lab])
}

#let arrow(x1, y1, x2, y2, col) = {
  import cetz.draw: *
  line((x1, y1), (x2, y2), stroke: 0.9pt + col, mark: (end: "stealth", fill: col, scale: 0.45))
}

#cetz.canvas({
  import cetz.draw: *

  let schem-w = schem-n * schem-cell
  let map-w = map-n * map-cell
  let gap = 0.75
  let col-gap = 1.0
  let row-gap = 1.45

  let wvf = wvf-stencil(d, theta)
  let lf = lf-stencil(d, theta)

  // Top-left schematic panel.
  let ax = 0.0
  let ay = 0.0
  draw-cell-grid(ax, ay, schem-n, schem-cell, black30)

  // Simple edge cue in the neighborhood.
  for r in range(schem-n) {
    for c in range(schem-n) {
      let dc = c - schem-half
      let dr = r - schem-half
      let dist = dc * calc.cos(theta) - dr * calc.sin(theta)
      if calc.abs(dc * dc + dr * dr) <= 16 {
        let x = ax + c * schem-cell
        let y = ay - r * schem-cell
        let inten = 0.5 + 0.4 * calc.tanh(dist * 0.8)
        rect((x, y), (x + schem-cell, y - schem-cell), fill: black90.lighten(100% * inten), stroke: 0.18pt + black30)
      }
    }
  }

  let cx = ax + schem-w / 2
  let cy = ay - schem-w / 2
  let nrad = 2.5 * schem-cell
  circle((cx, cy), radius: nrad, fill: none, stroke: 1.2pt + garnet)

  let nx = 1.0 * calc.cos(theta)
  let ny = 1.0 * calc.sin(theta)
  let tx = -calc.sin(theta)
  let ty = calc.cos(theta)
  line((cx - 0.4 * nx, cy - 0.4 * ny), (cx + 1.2 * nx, cy + 1.2 * ny), stroke: 1.2pt + horseshoe, mark: (end: "stealth", fill: horseshoe, scale: 0.35))
  line((cx - 0.4 * tx, cy - 0.4 * ty), (cx + 1.2 * tx, cy + 1.2 * ty), stroke: 1.2pt + rose, mark: (end: "stealth", fill: rose, scale: 0.35))
  content((cx + 1.15 * nx, cy + 1.15 * ny + 0.05), text(fill: horseshoe, size: 7.2pt, weight: "bold")[$x'$])
  content((cx + 1.15 * tx, cy + 1.15 * ty - 0.08), text(fill: rose, size: 7.2pt, weight: "bold")[$y'$])

  let eqx = ax + schem-w + 0.55
  let eqy = ay - 0.15
  box-label(eqx, eqy, 2.25, 0.55, [$A_(#math.theta) z approx b$])
  arrow(ax + schem-w - 0.02, cy + 0.55, eqx - 0.12, eqy - 0.2, black70)
  box-label(eqx, eqy - 0.85, 2.25, 0.55, [$hat(z) = P_(#math.theta) b$])
  arrow(eqx + 1.12, eqy - 0.55, eqx + 1.12, eqy - 0.32, black70)
  box-label(eqx, eqy - 1.7, 2.25, 0.55, [$p_(f_x) = P_(#math.theta)[1, :]$])

  content((ax + schem-w / 2, ay + 0.55), text(fill: black90, size: 9pt, weight: "bold")[WVF fit becomes one weight row])
  content((ax + schem-w / 2, ay - schem-w - 0.35), text(fill: black90, size: 8pt)[one dot product per pixel])

  // Top-right WVF map panel.
  let bx = ax + schem-w + col-gap + 2.8
  let by = ay
  draw-weight-map(bx, by, map-n, map-cell, wvf.stencil, wvf.max-abs)
  content((bx + map-w / 2, by + 0.55), text(fill: black90, size: 9pt, weight: "bold")[WVF weights])
  content((bx + map-w / 2, by - map-w - 0.28), text(fill: black90, size: 8pt)[fixed `p_(f_x)` for one orientation])

  // Bottom-left line-extension schematic.
  let cy2 = -schem-w - row-gap
  draw-cell-grid(ax, cy2, schem-n, schem-cell, black30)

  let line-pts = ()
  let n-line = 5
  for j in range(n-line) {
    let off = (j - 2) * 0.8 * schem-cell * 3.0
    let px = ax + schem-w / 2 + off * (-calc.sin(theta))
    let py = cy2 - schem-w / 2 + off * calc.cos(theta)
    line-pts.push((px, py))
  }
  for (i, p) in line-pts.enumerate() {
    let (px, py) = p
    circle((px, py), radius: 2.3 * schem-cell, fill: garnet.lighten(88%), stroke: 0.9pt + garnet.lighten(25%))
    circle((px, py), radius: 0.06, fill: garnet, stroke: none)
    if i < line-pts.len() - 1 {
      let (qx, qy) = line-pts.at(i + 1)
      line((px, py), (qx, qy), stroke: 0.7pt + black50, mark: (end: "stealth", fill: black50, scale: 0.25))
    }
  }
  line((line-pts.at(0).at(0) - 0.25, line-pts.at(0).at(1) - 0.15), (line-pts.at(4).at(0) + 0.25, line-pts.at(4).at(1) + 0.15), stroke: (dash: "dashed", paint: atlantic, thickness: 0.9pt))

  let eq2x = ax + schem-w + 0.55
  let eq2y = cy2 - 0.15
  box-label(eq2x, eq2y, 2.55, 0.55, [$R_(#math.theta) = sum_(j=-m)^m w_j hat(f)_x^((j))$])
  box-label(eq2x, eq2y - 0.85, 2.55, 0.55, [$w_j = exp(-j^2 / (2 sigma_ell^2))$])
  box-label(eq2x, eq2y - 1.7, 2.55, 0.55, [shift + sum + dedup])
  content((ax + schem-w / 2, cy2 + 0.55), text(fill: black90, size: 9pt, weight: "bold")[LF repeats the same WVF along the tangent])
  content((ax + schem-w / 2, cy2 - schem-w - 0.35), text(fill: black90, size: 8pt)[many shifted fits collapse into one fused stencil])

  // Bottom-right LF map panel.
  let dx = bx
  let dy = cy2
  draw-weight-map(dx, dy, map-n, map-cell, lf.stencil, lf.max-abs)
  content((dx + map-w / 2, dy + 0.55), text(fill: black90, size: 9pt, weight: "bold")[LF fused stencil])
  content((dx + map-w / 2, dy - map-w - 0.28), text(fill: black90, size: 8pt)[single accumulated weight map])

  // Connector arrows between schematic and map panels.
  arrow(ax + schem-w + 2.95, ay - schem-w / 2, bx - 0.2, by - schem-w / 2, rose)
  arrow(ax + schem-w + 3.05, cy2 - schem-w / 2, dx - 0.2, dy - schem-w / 2, rose)
})
