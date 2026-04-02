#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#let N = 13
#let cell = 0.42
#let theta = 30deg
#let sig-u = 2.0
#let sig-v = 1.2
#let half = (N - 1) / 2
#let r-cut = 3.0

#let cos-t = calc.cos(theta)
#let sin-t = calc.sin(theta)

#let rect-half-u = 3.0 * sig-u
#let rect-half-v = 3.0 * sig-v

#let get-uv(r, c) = {
  let dx = c - half
  let dy = -(r - half)
  let u = dx * cos-t + dy * sin-t
  let v = -dx * sin-t + dy * cos-t
  (u, v)
}

#let in-rect(r, c) = {
  let (u, v) = get-uv(r, c)
  calc.abs(u) <= rect-half-u and calc.abs(v) <= rect-half-v
}

#let in-ellipse(r, c) = {
  let (u, v) = get-uv(r, c)
  (u * u) / (sig-u * sig-u) + (v * v) / (sig-v * sig-v) <= r-cut * r-cut
}

// Corner pixels: in rect but not in ellipse
#let is-corner(r, c) = {
  in-rect(r, c) and not in-ellipse(r, c)
}

#let gauss(u, v) = {
  calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

#let kernel-weight(u, v) = {
  -v * calc.exp(-0.5 * (u * u / (sig-u * sig-u) + v * v / (sig-v * sig-v)))
}

#let max-weight = {
  let m = 0.0
  for r in range(N) {
    for c in range(N) {
      if in-ellipse(r, c) {
        let (u, v) = get-uv(r, c)
        let w = calc.abs(kernel-weight(u, v))
        if w > m { m = w }
      }
    }
  }
  m
}

#let max-deriv = {
  let m = 0.0
  for r in range(N) {
    for c in range(N) {
      if in-ellipse(r, c) {
        let (u, v) = get-uv(r, c)
        if calc.abs(v) > m { m = calc.abs(v) }
      }
    }
  }
  m
}

#let lerp-color(base, t) = {
  let t2 = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((base, t2 * 100%), (white, (1.0 - t2) * 100%))
}

#let draw-panel(ox, oy, fill-fn, mark-corners, label-text, math-text) = {
  import cetz.draw: *

  for r in range(N) {
    for c in range(N) {
      let x = ox + c * cell
      let y = oy - r * cell
      let fc = fill-fn(r, c)

      rect(
        (x, y),
        (x + cell, y - cell),
        fill: fc,
        stroke: 0.3pt + black30,
      )

      // Draw X marks on corner pixels
      if mark-corners and is-corner(r, c) {
        let cx = x + cell / 2
        let cy = y - cell / 2
        let d = cell * 0.3
        line((cx - d, cy - d), (cx + d, cy + d), stroke: 1.2pt + rose)
        line((cx - d, cy + d), (cx + d, cy - d), stroke: 1.2pt + rose)
      }
    }
  }

  // Panel border
  rect(
    (ox, oy),
    (ox + N * cell, oy - N * cell),
    stroke: 0.8pt + black90,
  )

  let cx = ox + N * cell / 2
  content((cx, oy + 0.55), text(fill: black90, size: 10pt, weight: "bold")[#label-text])
  content((cx, oy - N * cell - 0.4), text(fill: black90, size: 9pt, style: "italic")[#math-text])
}

#cetz.canvas({
  import cetz.draw: *

  let gap = 0.9
  let grid-w = N * cell
  let row-gap = 1.6

  // Panel A: Mask with corner exclusions marked
  let ax = 0
  let ay = 0
  draw-panel(ax, ay,
    (r, c) => {
      if in-ellipse(r, c) { black10 }
      else if is-corner(r, c) { rgb("#FFF2E3") }  // sandstorm for excluded corners
      else { white }
    },
    true,
    [(A) Mask $M_E$],
    [$M_E (i,j) = bb(1)[u^2 / sigma_u^2 + v^2 / sigma_v^2 <= r_0^2]$]
  )

  // Panel B: Envelope
  let bx = grid-w + gap
  let by = 0
  draw-panel(bx, by,
    (r, c) => {
      if in-ellipse(r, c) {
        let (u, v) = get-uv(r, c)
        let g = gauss(u, v)
        lerp-color(garnet, g)
      } else { white }
    },
    false,
    [(B) Envelope $G(u,v)$],
    [$G(u,v) = exp(-1/2 (u^2 / sigma_u^2 + v^2 / sigma_v^2))$]
  )

  // Panel C: Derivative
  let cx-pos = 0
  let cy-pos = -grid-w - row-gap
  draw-panel(cx-pos, cy-pos,
    (r, c) => {
      if in-ellipse(r, c) {
        let (u, v) = get-uv(r, c)
        let intensity = calc.abs(v) / max-deriv
        if v < 0 {
          lerp-color(garnet, intensity)
        } else if v > 0 {
          lerp-color(atlantic, intensity)
        } else { white }
      } else { white }
    },
    false,
    [(C) Derivative $-v$],
    [$-v "profile: positive (garnet) / negative (blue)"$]
  )

  // Panel D: Combined kernel
  let dx = grid-w + gap
  let dy = -grid-w - row-gap
  draw-panel(dx, dy,
    (r, c) => {
      if in-ellipse(r, c) {
        let (u, v) = get-uv(r, c)
        let w = kernel-weight(u, v)
        let intensity = calc.abs(w) / max-weight
        if w > 0.001 {
          lerp-color(garnet, intensity)
        } else if w < -0.001 {
          lerp-color(atlantic, intensity)
        } else { white }
      } else { white }
    },
    false,
    [(D) Combined $K_E$],
    [$K_E = -v dot G(u,v) dot M_E$]
  )

  // Title
  let total-w = 2 * grid-w + gap
  content((total-w / 2, 1.3), text(fill: black90, size: 11pt, weight: "bold")[Elliptical Kernel Construction])
})
