#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let sandstorm = rgb("#FFF2E3")
#let white = rgb("#FFFFFF")

#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let panel-title(x, y, text-body) = {
  import cetz.draw: *
  content((x, y), text(fill: black90, size: 9.5pt, weight: "bold")[#text-body], anchor: "west")
}

#let small-label(x, y, text-body) = {
  import cetz.draw: *
  content((x, y), text(fill: black70, size: 7.4pt)[#text-body], anchor: "west")
}

#let draw-neighborhood-panel(ox, oy) = {
  import cetz.draw: *
  let cell = 0.22
  let grid-n = 13
  let half = 6
  let edge-angle = 30.0
  let edge-rad = edge-angle * calc.pi / 180.0
  let radius = 2.5
  panel-title(ox, oy + 0.45, [1. Rotated Local Fit])

  for r in range(grid-n) {
    for c in range(grid-n) {
      let x = ox + c * cell
      let y = oy - r * cell
      let dc = c - half
      let dr = r - half
      let dist = dc * calc.cos(edge-rad) - dr * calc.sin(edge-rad)
      let intensity = 0.5 + 0.45 * calc.tanh(dist * 0.9)
      let fc = black90.lighten(100% * intensity)
      rect((x, y), (x + cell, y - cell), fill: fc, stroke: 0.12pt + black30)
    }
  }

  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let dist = calc.sqrt(dc * dc + dr * dr)
      if dist <= radius {
        let x = ox + c * cell
        let y = oy - r * cell
        rect((x, y), (x + cell, y - cell), fill: garnet.lighten(88%).transparentize(25%), stroke: 0.12pt + black30)
      }
    }
  }

  let cx = ox + half * cell + cell / 2
  let cy = oy - half * cell - cell / 2
  circle((cx, cy), radius: radius * cell, fill: none, stroke: 1pt + garnet)
  circle((cx, cy), radius: 0.04, fill: white, stroke: 0.4pt + black90)

  let normal-angle = 45.0 * calc.pi / 180.0
  let tangent-angle = normal-angle + calc.pi / 2
  let axis-len = 0.95
  let nx = axis-len * calc.cos(normal-angle)
  let ny = axis-len * calc.sin(normal-angle)
  let tx = axis-len * calc.cos(tangent-angle)
  let ty = axis-len * calc.sin(tangent-angle)

  line((cx - nx * 0.25, cy - ny * 0.25), (cx + nx, cy + ny), stroke: 1pt + horseshoe, mark: (end: "stealth", fill: horseshoe, size: 0.11))
  line((cx - tx * 0.25, cy - ty * 0.25), (cx + tx, cy + ty), stroke: 1pt + rose, mark: (end: "stealth", fill: rose, size: 0.11))
  content((cx + nx * 1.05 + 0.08, cy + ny * 1.05), text(fill: horseshoe, size: 7pt)[$x'$])
  content((cx + tx * 1.05 - 0.02, cy + ty * 1.05 - 0.12), text(fill: rose, size: 7pt)[$y'$])

  line((cx + tx * 0.14, cy + ty * 0.14), (cx + tx * 0.14 + nx * 1.2, cy + ty * 0.14 + ny * 1.2), stroke: 1.5pt + black90, mark: (end: "stealth", fill: black90, size: 0.18))
  content((cx + nx * 1.35, cy + ny * 1.35 + 0.14), text(fill: black90, size: 7pt, weight: "bold")[$partial f \/ partial x'$])

  small-label(ox, oy - grid-n * cell - 0.22, [$A_theta z approx b$ over circular neighbors])
}

#let draw-pseudoinverse-panel(ox, oy) = {
  import cetz.draw: *
  panel-title(ox, oy + 0.45, [2. WVF Row Extraction])

  // b vector
  let vx = ox + 0.1
  let vy = oy - 0.2
  let bw = 0.45
  let bh = 0.22
  for i in range(5) {
    rect((vx, vy - i * bh), (vx + bw, vy - (i + 1) * bh), fill: black10, stroke: 0.12pt + black30)
  }
  content((vx + bw / 2, vy + 0.18), text(fill: black90, size: 7.5pt, weight: "bold")[$b$])

  // P_theta block
  let mx = ox + 1.05
  let my = oy - 0.02
  let cw = 0.34
  let ch = 0.2
  for r in range(4) {
    for c in range(6) {
      let fc = if r == 1 { sandstorm } else { white }
      rect((mx + c * cw, my - r * ch), (mx + (c + 1) * cw, my - (r + 1) * ch), fill: fc, stroke: 0.12pt + black30)
    }
  }
  rect((mx, my), (mx + 6 * cw, my - 4 * ch), stroke: 0.5pt + black90)
  content((mx + 3 * cw, my + 0.2), text(fill: black90, size: 7.8pt, weight: "bold")[$P_theta = (A_theta^top A_theta)^(-1) A_theta^top$])
  content((mx + 6 * cw + 0.15, my - 1.5 * ch), text(fill: garnet, size: 7.5pt, weight: "bold")[$p_(f_x)^top$], anchor: "west")
  line((mx + 6 * cw + 0.02, my - 1.5 * ch), (mx + 6 * cw + 0.11, my - 1.5 * ch), stroke: 1pt + garnet)

  // multiplication arrow and dot product
  line((vx + bw + 0.12, vy - 2.5 * bh), (mx - 0.08, my - 1.5 * ch), stroke: 0.8pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  content((ox + 2.1, oy - 1.25), text(fill: black90, size: 8pt, weight: "bold")[$hat(f)_(x') = p_(f_x)^top b$])
  small-label(ox, oy - 1.62, [One row of the pseudoinverse is already a fixed WVF weight vector.])
}

#let draw-weight-map(ox, oy, elongated, title) = {
  import cetz.draw: *
  let n = 13
  let cell = 0.18
  let half = 6
  let angle = 30deg
  let ct = calc.cos(angle)
  let st = calc.sin(angle)
  panel-title(ox, oy + 0.45, [#title])

  for r in range(n) {
    for c in range(n) {
      let dx = c - half
      let dy = -(r - half)
      let u = dx * ct + dy * st
      let v = -dx * st + dy * ct
      let env-u = if elongated { 4.2 } else { 2.5 }
      let env-v = 1.5
      let base = calc.exp(-0.5 * ((u * u) / (env-u * env-u) + (v * v) / (env-v * env-v)))
      let w = -v * base
      let fc = if elongated {
        if calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8 {
          if w > 0.02 { lerp-color(garnet, calc.min(calc.abs(w) / 1.0, 1.0)) }
          else if w < -0.02 { lerp-color(atlantic, calc.min(calc.abs(w) / 1.0, 1.0)) }
          else { white }
        } else { white }
      } else {
        if calc.sqrt(dx * dx + dy * dy) <= 3.0 {
          if w > 0.02 { lerp-color(garnet, calc.min(calc.abs(w) / 1.0, 1.0)) }
          else if w < -0.02 { lerp-color(atlantic, calc.min(calc.abs(w) / 1.0, 1.0)) }
          else { white }
        } else { white }
      }
      rect((ox + c * cell, oy - r * cell), (ox + (c + 1) * cell, oy - (r + 1) * cell), fill: fc, stroke: none)
    }
  }

  // outer boundary
  for r in range(n) {
    for c in range(n) {
      let dx = c - half
      let dy = -(r - half)
      let inside = if elongated {
        let u = dx * ct + dy * st
        let v = -dx * st + dy * ct
        calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8
      } else {
        calc.sqrt(dx * dx + dy * dy) <= 3.0
      }
      if inside {
        let x = ox + c * cell
        let y = oy - r * cell
        let left-inside = if elongated {
          let u = (dx - 1) * ct + dy * st
          let v = -(dx - 1) * st + dy * ct
          calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8
        } else { calc.sqrt((dx - 1) * (dx - 1) + dy * dy) <= 3.0 }
        let right-inside = if elongated {
          let u = (dx + 1) * ct + dy * st
          let v = -(dx + 1) * st + dy * ct
          calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8
        } else { calc.sqrt((dx + 1) * (dx + 1) + dy * dy) <= 3.0 }
        let top-inside = if elongated {
          let u = dx * ct + (dy + 1) * st
          let v = -dx * st + (dy + 1) * ct
          calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8
        } else { calc.sqrt(dx * dx + (dy + 1) * (dy + 1)) <= 3.0 }
        let bottom-inside = if elongated {
          let u = dx * ct + (dy - 1) * st
          let v = -dx * st + (dy - 1) * ct
          calc.abs(u) <= 4.8 and calc.abs(v) <= 2.8
        } else { calc.sqrt(dx * dx + (dy - 1) * (dy - 1)) <= 3.0 }
        if not left-inside { line((x, y), (x, y - cell), stroke: 0.35pt + black90) }
        if not right-inside { line((x + cell, y), (x + cell, y - cell), stroke: 0.35pt + black90) }
        if not top-inside { line((x, y), (x + cell, y), stroke: 0.35pt + black90) }
        if not bottom-inside { line((x, y - cell), (x + cell, y - cell), stroke: 0.35pt + black90) }
      }
    }
  }
  rect((ox, oy), (ox + n * cell, oy - n * cell), stroke: 0.45pt + black90)
}

#let draw-lf-fusion-panel(ox, oy) = {
  import cetz.draw: *
  panel-title(ox, oy + 0.45, [4. LF = Shifted WVFs + Gaussian Weights])

  let cell = 0.14
  let centers = ((0.0, 0.0), (0.55, -0.32), (1.1, -0.64))
  for (idx, c) in centers.enumerate() {
    let cx = ox + 0.55 + c.at(0)
    let cy = oy - 0.55 + c.at(1)
    circle((cx, cy), radius: 0.48, fill: if idx == 1 { sandstorm.transparentize(15%) } else { black10.transparentize(40%) }, stroke: 0.6pt + black50)
    content((cx + 0.6, cy + 0.24 - idx * 0.02), text(fill: black70, size: 7pt)[$w_#(str(idx - 1)) dot WVF_j$], anchor: "west")
  }
  line((ox + 0.34, oy - 0.1), (ox + 1.8, oy - 1.1), stroke: 1pt + rose, mark: (end: "stealth", fill: rose, size: 0.12))
  content((ox + 1.02, oy - 0.18), text(fill: rose, size: 7pt, weight: "bold")[tangent line])

  line((ox + 2.15, oy - 0.84), (ox + 3.05, oy - 0.84), stroke: 0.9pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  content((ox + 2.16, oy - 0.52), text(fill: black70, size: 7pt)[reindex and sum], anchor: "west")

  draw-weight-map(ox + 3.25, oy - 0.2, true, [Fused LF Stencil])
  small-label(ox + 3.25, oy - 2.85, [A weighted sum of shifted WVFs is one fixed elongated stencil.])
}

#cetz.canvas({
  import cetz.draw: *
  let y0 = 0
  draw-neighborhood-panel(0.0, y0)
  line((3.2, y0 - 1.2), (4.0, y0 - 1.2), stroke: 0.9pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))

  draw-pseudoinverse-panel(4.25, y0)
  line((8.0, y0 - 1.2), (8.8, y0 - 1.2), stroke: 0.9pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))

  draw-weight-map(9.05, y0 - 0.2, false, [3. WVF Stencil])
  small-label(9.05, y0 - 2.85, [For fixed orientation, WVF is already a standard discrete filter.])

  line((11.55, y0 - 1.2), (12.35, y0 - 1.2), stroke: 0.9pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  draw-lf-fusion-panel(12.55, y0)
})
