#set page(width: auto, height: auto, margin: 10pt)
#set text(font: "New Computer Modern", size: 10pt)

// Compute weight efficiency η for the paper's stated parameters:
// d = 4 (M = 15 monomials), N_p = 100, m = 7

// ── Parameters ──────────────────────────────────────────────────────────
#let poly-d   = 4
#let target-Np = 100
#let m-line   = 7
#let sigma-l  = m-line / 2.0

// ── Factorial (up to 4) ─────────────────────────────────────────────────
#let fact(n) = {
  if n <= 0 { 1 } else if n == 1 { 1 } else if n == 2 { 2 }
  else if n == 3 { 6 } else { 24 }
}

// ── Select N_p closest integer pixels ───────────────────────────────────
#let neighbors = {
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
  let result = ()
  for i in range(calc.min(target-Np, cands.len())) {
    result.push((cands.at(i).at(0), cands.at(i).at(1)))
  }
  result
}
#let Np = neighbors.len()

// ── Monomial basis for degree d at (x, y) ───────────────────────────────
// Order: group by total degree, within each degree by descending x-power
// d=4 → 15 terms: 1, x, y, x²/2, xy, y²/2, x³/6, x²y/2, xy²/2, y³/6,
//                  x⁴/24, x³y/6, x²y²/4, xy³/6, y⁴/24
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

#let M = {
  let b = monomial-basis(1.0, 1.0)
  b.len()
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
  P.at(1) // f_x row
}

// ── Fused stencil ───────────────────────────────────────────────────────
#let fused-stencil(theta) = {
  let p = gradient-row(theta)
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

// ── Compute η ───────────────────────────────────────────────────────────
#let compute-eta(theta) = {
  let p = gradient-row(theta)
  let st = fused-stencil(theta)
  let numer = 0.0
  for (_, val) in st { numer = numer + calc.abs(val) }
  let sum-wj = 0.0
  for jj in range(2 * m-line + 1) {
    let j = jj - m-line
    sum-wj = sum-wj + calc.exp(-(j * j) / (2.0 * sigma-l * sigma-l))
  }
  let sum-abs-p = 0.0
  for ii in range(Np) { sum-abs-p = sum-abs-p + calc.abs(p.at(ii)) }
  (numer: numer, denom: sum-wj * sum-abs-p, eta: numer / (sum-wj * sum-abs-p))
}

// ── Compute and display ─────────────────────────────────────────────────
#let orientations = (0, 30, 45, 60, 90, 120, 150)

#text(weight: "bold", size: 12pt)[Weight Efficiency $eta$ at $d = #poly-d$, $N_p = #Np$, $m = #m-line$]
#v(6pt)
#text(size: 9pt)[($M = #M$ monomials, $sigma_ell = #sigma-l$)]
#v(10pt)

#for deg in orientations {
  let result = compute-eta(deg * 1deg)
  let pct = str(calc.round(result.eta * 100, digits: 2))
  [
    $theta = #deg degree$: #h(1em)
    $eta = #pct$% #h(1em)
    (numerator = #str(calc.round(result.numer, digits: 4)),
     denominator = #str(calc.round(result.denom, digits: 4)))
    #linebreak()
  ]
}
