#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black30 = rgb("#C7C7C7")
#let white = rgb("#FFFFFF")

#let factorial(n) = {
  if n <= 1 {
    1.0
  } else {
    let acc = 1.0
    for k in range(2, n + 1) {
      acc = acc * k
    }
    acc
  }
}

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
  let aug = ()
  for i in range(n) {
    let row = ()
    for j in range(n) {
      row.push(A.at(i).at(j) * 1.0)
    }
    for j in range(n) {
      row.push(if i == j { 1.0 } else { 0.0 })
    }
    aug.push(row)
  }
  for k in range(n) {
    let max-row = k
    let max-val = calc.abs(aug.at(k).at(k))
    for i in range(k + 1, n) {
      let v = calc.abs(aug.at(i).at(k))
      if v > max-val {
        max-val = v
        max-row = i
      }
    }
    if max-row != k {
      let tmp = aug.at(k)
      aug.at(k) = aug.at(max-row)
      aug.at(max-row) = tmp
    }
    let pivot = aug.at(k).at(k)
    let rk = aug.at(k)
    for j in range(2 * n) {
      rk.at(j) = rk.at(j) / pivot
    }
    aug.at(k) = rk
    for i in range(n) {
      if i != k {
        let fac = aug.at(i).at(k)
        let ri = aug.at(i)
        for j in range(2 * n) {
          ri.at(j) = ri.at(j) - fac * aug.at(k).at(j)
        }
        aug.at(i) = ri
      }
    }
  }
  let inv = ()
  for i in range(n) {
    let row = ()
    for j in range(n) {
      row.push(aug.at(i).at(n + j))
    }
    inv.push(row)
  }
  inv
}

#let monomial-basis(x, y, d) = {
  let b = ()
  for deg in range(d + 1) {
    for offset in range(deg + 1) {
      let p = deg - offset
      let q = offset
      let xp = if p == 0 { 1.0 } else { calc.pow(x, p) }
      let yq = if q == 0 { 1.0 } else { calc.pow(y, q) }
      b.push(xp * yq / (factorial(p) * factorial(q)))
    }
  }
  b
}

#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let oriented-uv(dx, dy, theta) = {
  let ct = calc.cos(theta)
  let st = calc.sin(theta)
  let u = dx * ct + dy * st
  let v = -dx * st + dy * ct
  (u, v)
}

#let build-support(shape, theta, grid-n, support-u, support-v) = {
  let half = calc.div-euclid(grid-n - 1, 2)
  let pts = ()
  for r in range(grid-n) {
    for c in range(grid-n) {
      let dx = c - half
      let dy = -(r - half)
      let (u, v) = oriented-uv(dx, dy, theta)
      let keep = if shape == "rect" {
        calc.abs(u) <= support-u and calc.abs(v) <= support-v
      } else if shape == "ellipse" {
        (u * u) / (support-u * support-u) + (v * v) / (support-v * support-v) <= 1.0
      } else {
        false
      }
      if keep {
        pts.push((dx, dy))
      }
    }
  }
  pts
}

#let poly-stencil(shape, theta, d, grid-n, support-u, support-v) = {
  let support = build-support(shape, theta, grid-n, support-u, support-v)
  let A = ()
  for ii in range(support.len()) {
    let dx = support.at(ii).at(0)
    let dy = support.at(ii).at(1)
    let (u, v) = oriented-uv(dx, dy, theta)
    A.push(monomial-basis(u, v, d))
  }
  let At = mat-transpose(A)
  let AtA = mat-mul(At, A)
  let AtA-inv = mat-invert(AtA)
  let P = mat-mul(AtA-inv, At)
  let dv-row = P.at(2)
  let stencil = (:)
  for ii in range(support.len()) {
    let dx = support.at(ii).at(0)
    let dy = support.at(ii).at(1)
    stencil.insert(str(dx) + "," + str(dy), dv-row.at(ii))
  }
  let max-abs = 0.0
  for (_, val) in stencil {
    let a = calc.abs(val)
    if a > max-abs {
      max-abs = a
    }
  }
  (stencil: stencil, max-abs: max-abs)
}

#let gaussian-stencil(theta, grid-n, sigma-u, sigma-v, support-u, support-v) = {
  let support = build-support("ellipse", theta, grid-n, support-u, support-v)
  let sigma-u-eff = 2.0 * sigma-u
  let sigma-v-eff = 2.0 * sigma-v
  let raw = ()
  let mean = 0.0
  for ii in range(support.len()) {
    let dx = support.at(ii).at(0)
    let dy = support.at(ii).at(1)
    let (u, v) = oriented-uv(dx, dy, theta)
    let g = calc.exp(-0.5 * (u * u / (sigma-u-eff * sigma-u-eff) + v * v / (sigma-v-eff * sigma-v-eff)))
    let w = -(v / (sigma-v-eff * sigma-v-eff)) * g
    raw.push((dx, dy, w))
    mean = mean + w
  }
  mean = mean / support.len()
  let stencil = (:)
  let max-abs = 0.0
  for ii in range(raw.len()) {
    let dx = raw.at(ii).at(0)
    let dy = raw.at(ii).at(1)
    let w = raw.at(ii).at(2) - mean
    stencil.insert(str(dx) + "," + str(dy), w)
    let a = calc.abs(w)
    if a > max-abs {
      max-abs = a
    }
  }
  (stencil: stencil, max-abs: max-abs)
}

#let build-family-data(kind, orientations, d, grid-n, support-u, support-v, sigma-u, sigma-v) = {
  orientations.map(theta-deg => {
    let theta = theta-deg * 1deg
    if kind == "rect_poly" {
      (theta-deg: theta-deg) + poly-stencil("rect", theta, d, grid-n, support-u, support-v)
    } else if kind == "ellipse_poly" {
      (theta-deg: theta-deg) + poly-stencil("ellipse", theta, d, grid-n, support-u, support-v)
    } else {
      (theta-deg: theta-deg) + gaussian-stencil(theta, grid-n, sigma-u, sigma-v, support-u, support-v)
    }
  })
}

#let draw-panel(ox, oy, data, grid-n, cell, global-max) = {
  import cetz.draw: *
  let half = calc.div-euclid(grid-n - 1, 2)
  let stencil = data.stencil
  for r in range(grid-n) {
    for c in range(grid-n) {
      let x = ox + c * cell
      let y = oy - r * cell
      let dx = c - half
      let dy = -(r - half)
      let key = str(dx) + "," + str(dy)
      let fc = if key in stencil {
        let w = stencil.at(key)
        let intensity = calc.abs(w) / global-max
        if w > 0.000001 {
          lerp-color(garnet, intensity)
        } else if w < -0.000001 {
          lerp-color(atlantic, intensity)
        } else {
          white
        }
      } else {
        white
      }
      rect((x, y), (x + cell, y - cell), fill: fc, stroke: none)
    }
  }
  for r in range(grid-n) {
    for c in range(grid-n) {
      let x = ox + c * cell
      let y = oy - r * cell
      let dx = c - half
      let dy = -(r - half)
      let key = str(dx) + "," + str(dy)
      if key in stencil {
        let left-key = str(dx - 1) + "," + str(dy)
        let right-key = str(dx + 1) + "," + str(dy)
        let top-key = str(dx) + "," + str(dy + 1)
        let bottom-key = str(dx) + "," + str(dy - 1)
        if left-key not in stencil {
          line((x, y), (x, y - cell), stroke: 0.35pt + black90)
        }
        if right-key not in stencil {
          line((x + cell, y), (x + cell, y - cell), stroke: 0.35pt + black90)
        }
        if top-key not in stencil {
          line((x, y), (x + cell, y), stroke: 0.35pt + black90)
        }
        if bottom-key not in stencil {
          line((x, y - cell), (x + cell, y - cell), stroke: 0.35pt + black90)
        }
      }
    }
  }
  rect((ox, oy), (ox + grid-n * cell, oy - grid-n * cell), stroke: 0.35pt + black30)
  content(
    (ox + grid-n * cell / 2, oy + 0.32),
    text(fill: black90, size: 9pt, weight: "bold")[$theta = #(data.theta-deg)#sym.degree$]
  )
}

#let draw-heatbar(ox, oy, width, height) = {
  import cetz.draw: *
  let steps = 80
  for i in range(steps) {
    let t0 = i / steps
    let t1 = (i + 1) / steps
    let x0 = ox + width * t0
    let x1 = ox + width * t1
    let fc = if t0 < 0.5 {
      let q = 1.0 - (t0 / 0.5)
      lerp-color(atlantic, q)
    } else {
      let q = (t0 - 0.5) / 0.5
      lerp-color(garnet, q)
    }
    rect((x0, oy), (x1, oy - height), fill: fc, stroke: none)
  }
  rect((ox, oy), (ox + width, oy - height), stroke: 0.35pt + black90)
  content((ox, oy - height - 0.18), text(fill: black90, size: 8pt)[$-w_"max"$], anchor: "north-west")
  content((ox + width / 2, oy - height - 0.18), text(fill: black90, size: 8pt)[$0$], anchor: "north")
  content((ox + width, oy - height - 0.18), text(fill: black90, size: 8pt)[$+w_"max"$], anchor: "north-east")
}

#let render-orientation-sweep(
  kind,
  d: 1,
  orientations: (0, 30, 60, 90),
  grid-n: 15,
  cell: 0.24,
  sigma-u: 2.0,
  sigma-v: 1.2,
) = {
  let support-u = 3.0 * sigma-u
  let support-v = 3.0 * sigma-v
  let data = build-family-data(kind, orientations, d, grid-n, support-u, support-v, sigma-u, sigma-v)
  let global-max = {
    let m = 0.0
    for item in data {
      if item.max-abs > m {
        m = item.max-abs
      }
    }
    m
  }
  cetz.canvas({
    import cetz.draw: *

    let panel-w = grid-n * cell
    let gap-x = 0.7
    let ox = 0.0
    let oy = 0.0

    for idx in range(data.len()) {
      let px = ox + idx * (panel-w + gap-x)
      draw-panel(px, oy, data.at(idx), grid-n, cell, global-max)
    }

    let total-w = data.len() * panel-w + (data.len() - 1) * gap-x
    draw-heatbar(ox + total-w * 0.22, oy - panel-w - 0.48, total-w * 0.56, 0.12)
  })
}
