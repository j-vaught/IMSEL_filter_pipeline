#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let white = rgb("#FFFFFF")

#let N = 15
#let h = calc.div-euclid(N - 1, 2)
#let cell = 0.24
#let gap-x = 1.5
#let gap-y = 0.5
#let degrees = (1, 3, 5)

#let fact(n) = {
  if n <= 0 { 1 }
  else if n == 1 { 1 }
  else if n == 2 { 2 }
  else if n == 3 { 6 }
  else if n == 4 { 24 }
  else if n == 5 { 120 }
  else { 720 }
}

#let square-coords() = {
  let pts = ()
  for r in range(N) {
    let y = r - h
    for c in range(N) {
      let x = c - h
      pts.push((x * 1.0, y * 1.0))
    }
  }
  pts
}

#let coords = square-coords()

#let basis(order, x, y) = {
  let cols = (1.0,)
  if order >= 1 {
    cols.push(x)
    cols.push(y)
  }
  if order >= 2 {
    cols.push(x * x / 2.0)
    cols.push(y * y / 2.0)
    cols.push(x * y)
  }
  if order >= 3 {
    cols.push(x * x * x / 6.0)
    cols.push(y * y * y / 6.0)
    cols.push(x * x * y / 2.0)
    cols.push(x * y * y / 2.0)
  }
  if order >= 4 {
    cols.push(x * x * x * x / 24.0)
    cols.push(y * y * y * y / 24.0)
    cols.push(x * x * x * y / 6.0)
    cols.push(x * x * y * y / 4.0)
    cols.push(x * y * y * y / 6.0)
  }
  if order >= 5 {
    cols.push(x * x * x * x * x / 120.0)
    cols.push(y * y * y * y * y / 120.0)
    cols.push(x * x * x * x * y / 24.0)
    cols.push(x * x * x * y * y / 12.0)
    cols.push(x * x * y * y * y / 12.0)
    cols.push(x * y * y * y * y / 24.0)
  }
  cols
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

#let pseudoinverse(order) = {
  let A = ()
  for i in range(coords.len()) {
    let x = coords.at(i).at(0)
    let y = coords.at(i).at(1)
    A.push(basis(order, x, y))
  }
  let At = mat-transpose(A)
  let AtA = mat-mul(At, A)
  let AtA-inv = mat-invert(AtA)
  mat-mul(AtA-inv, At)
}

#let row-to-kernel(row) = {
  let K = ()
  for r in range(N) {
    let kr = ()
    for c in range(N) {
      kr.push(row.at(r * N + c))
    }
    K.push(kr)
  }
  K
}

#let kernel-data = degrees.map(d => {
  let P = pseudoinverse(d)
  (
    degree: d,
    kx: row-to-kernel(P.at(1)),
    ky: row-to-kernel(P.at(2)),
  )
})

#let global-max = {
  let mx = 0.0
  for item in kernel-data {
    for K in (item.kx, item.ky) {
      for r in range(N) {
        for c in range(N) {
          let v = calc.abs(K.at(r).at(c))
          if v > mx { mx = v }
        }
      }
    }
  }
  mx
}

#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let weight-fill(w) = {
  let a = calc.abs(w)
  if a <= global-max * 0.0005 { white }
  else if w > 0 { lerp-color(garnet, a / global-max) }
  else { lerp-color(atlantic, a / global-max) }
}

#let signed-color(t) = {
  if calc.abs(t) < 0.0001 { white }
  else if t > 0 { lerp-color(garnet, t) }
  else { lerp-color(atlantic, calc.abs(t)) }
}

#let draw-panel(ox, oy, K) = {
  import cetz.draw: *
  let panel-w = N * cell
  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell
      rect(
        (x, y),
        (x + cell, y - cell),
        fill: weight-fill(K.at(r).at(c)),
        stroke: 0.14pt + black30,
      )
    }
  }
  let mid-x = ox + (h + 0.5) * cell
  let mid-y = oy - (h + 0.5) * cell
  line((mid-x, oy), (mid-x, oy - panel-w), stroke: 0.45pt + black50)
  line((ox, mid-y), (ox + panel-w, mid-y), stroke: 0.45pt + black50)
  rect((ox, oy), (ox + panel-w, oy - panel-w), stroke: 0.8pt + black90)
}

#cetz.canvas({
  import cetz.draw: *

  let panel-w = N * cell
  let total-w = 2 * panel-w + gap-x

  content((panel-w / 2.0, 0.45), text(fill: black90, size: 10pt, weight: "bold")[$bold(K)_x^("square")$])
  content((panel-w + gap-x + panel-w / 2.0, 0.45), text(fill: black90, size: 10pt, weight: "bold")[$bold(K)_y^("square")$])

  for idx in range(degrees.len()) {
    let item = kernel-data.at(idx)
    let oy = -(idx * (panel-w + gap-y))
    draw-panel(0.0, oy, item.kx)
    draw-panel(panel-w + gap-x, oy, item.ky)
    content(
      (-0.55, oy - panel-w / 2.0),
      text(fill: black70, size: 10pt, weight: "bold")[$d = #(item.degree)$],
      anchor: "east",
    )
  }

  let legend-y = -(3 * panel-w + 2 * gap-y) - 0.55
  let legend-x = total-w / 2.0 - 2.0
  let legend-w = 4.0
  let legend-h = 0.24
  let steps = 48

  for i in range(steps) {
    let x0 = legend-x + i * legend-w / steps
    let x1 = legend-x + (i + 1) * legend-w / steps
    let t = -1.0 + 2.0 * (i + 0.5) / steps
    rect(
      (x0, legend-y),
      (x1, legend-y - legend-h),
      fill: signed-color(t),
      stroke: none,
    )
  }
  rect((legend-x, legend-y), (legend-x + legend-w, legend-y - legend-h), stroke: 0.2pt + black30)
  content((legend-x, legend-y - legend-h - 0.16), text(fill: black70, size: 8pt)[$-w_"max"$], anchor: "north-west")
  content((legend-x + legend-w / 2.0, legend-y - legend-h - 0.16), text(fill: black70, size: 8pt)[$0$], anchor: "north")
  content((legend-x + legend-w, legend-y - legend-h - 0.16), text(fill: black70, size: 8pt)[$+w_"max"$], anchor: "north-east")
  content((legend-x + legend-w / 2.0, legend-y + 0.16), text(fill: black70, size: 8pt)[weight], anchor: "south")
})
