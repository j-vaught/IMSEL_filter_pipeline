#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 10pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let rose = rgb("#CC2E40")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  // Layout: top panel = data + fit, bottom panel = weight bars at each point
  // Shared x-axis so the connection is immediate

  let data = (
    (0.8, 1.5),
    (1.4, 1.9),
    (1.9, 1.6),
    (2.5, 2.4),
    (3.0, 2.7),
    (3.5, 2.5),
    (4.0, 3.0),
    (4.5, 3.4),  // target (index 7)
    (5.0, 3.1),
    (5.5, 3.6),
    (6.0, 3.3),
    (6.5, 3.9),
    (7.0, 3.7),
    (7.5, 4.1),
  )

  let target-idx = 7
  let (tx, ty) = data.at(target-idx)
  let bw = 2.0

  // ============================================================
  // TOP PANEL: Data points + fitted line (y range ~1.0 to 4.5)
  // ============================================================
  let top-y0 = 2.5  // bottom of top panel in canvas coords
  let top-scale = 1.3  // y scaling

  // Top panel axes
  line((0.2, top-y0), (8.3, top-y0), stroke: 0.5pt + black50)
  line((0.2, top-y0), (0.2, top-y0 + 4.5), stroke: 0.5pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  content((0.0, top-y0 + 4.7), text(fill: black70, size: 8pt)[$f(x)$])

  // Map data y to canvas y in top panel
  let top-y(val) = { top-y0 + (val - 1.0) * top-scale }

  // Bandwidth shading in top panel
  rect(
    (calc.max(tx - bw, 0.3), top-y0),
    (calc.min(tx + bw, 8.0), top-y0 + 4.3),
    fill: congaree.lighten(94%),
    stroke: none,
  )
  // Bandwidth boundary
  line((tx - bw, top-y0), (tx - bw, top-y0 + 4.3), stroke: (dash: "dashed", paint: congaree.lighten(40%), thickness: 0.6pt))
  line((tx + bw, top-y0), (tx + bw, top-y0 + 4.3), stroke: (dash: "dashed", paint: congaree.lighten(40%), thickness: 0.6pt))

  // Fitted line (weighted least squares result)
  line(
    (tx - bw + 0.2, top-y(2.15)),
    (tx + bw - 0.2, top-y(3.55)),
    stroke: 2.2pt + atlantic,
  )

  // Data points
  for i in range(data.len()) {
    let (px, py) = data.at(i)
    let cy = top-y(py)
    let dist = calc.abs(px - tx)
    let u = dist / bw

    if i == target-idx {
      circle((px, cy), radius: 0.13, fill: garnet, stroke: 1.2pt + garnet)
    } else if u < 1.0 {
      let w = calc.pow(1.0 - calc.pow(u, 3), 3)
      let r = 0.04 + 0.07 * w
      circle((px, cy), radius: r, fill: black90.lighten(60% - 50% * w), stroke: 0.4pt + black70)
    } else {
      circle((px, cy), radius: 0.035, fill: black30, stroke: 0.3pt + black50)
    }
  }

  // Label target
  content((tx, top-y(ty) + 0.35), text(fill: garnet, size: 8pt, weight: "bold")[target $x_0$])
  // Label fit
  content((tx + bw - 0.5, top-y(3.55) + 0.3), text(fill: atlantic, size: 8pt)[$hat(f)(x_0)$])

  // ============================================================
  // BOTTOM PANEL: Weight bars at each data point's x position
  // ============================================================
  let bot-y0 = 0.0   // baseline
  let bot-h = 1.8    // max bar height
  let bar-w = 0.25   // bar width

  // Bottom panel axis
  line((0.2, bot-y0), (8.3, bot-y0), stroke: 0.5pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  content((8.5, bot-y0), text(fill: black70, size: 8pt)[$x$])
  line((0.2, bot-y0), (0.2, bot-y0 + bot-h + 0.3), stroke: 0.5pt + black50, mark: (end: "stealth", fill: black50, size: 0.12))
  content((-0.25, bot-y0 + bot-h * 0.5), text(fill: congaree, size: 7.5pt)[weight\ $K(x)$])

  // "w = 0" and "w = 1" tick labels
  content((0.0, bot-y0 - 0.15), text(fill: black70, size: 6.5pt)[0])
  content((0.0, bot-y0 + bot-h), text(fill: black70, size: 6.5pt)[1])
  line((0.15, bot-y0 + bot-h), (0.25, bot-y0 + bot-h), stroke: 0.4pt + black50)

  // Draw weight bars for each data point
  for i in range(data.len()) {
    let (px, py) = data.at(i)
    let dist = calc.abs(px - tx)
    let u = dist / bw

    if u < 1.0 {
      let w = calc.pow(1.0 - calc.pow(u, 3), 3)
      let h = w * bot-h
      let clr = if i == target-idx { garnet } else { congaree.lighten(40% - 35% * w) }

      // Bar
      rect(
        (px - bar-w / 2, bot-y0),
        (px + bar-w / 2, bot-y0 + h),
        fill: clr,
        stroke: 0.4pt + clr.darken(20%),
      )

      // Vertical connector from bar top to data point in top panel
      let cy = top-y(py)
      line(
        (px, bot-y0 + h),
        (px, cy),
        stroke: (dash: "dotted", paint: black30, thickness: 0.6pt),
      )
    } else {
      // Zero-weight marker
      line(
        (px - 0.08, bot-y0 + 0.03),
        (px + 0.08, bot-y0 + 0.03),
        stroke: 0.8pt + black30,
      )
    }
  }

  // Smooth kernel curve overlay on bottom panel
  let n-kernel = 80
  for i in range(n-kernel) {
    let x1 = tx - bw + 2 * bw * i / n-kernel
    let x2 = tx - bw + 2 * bw * (i + 1) / n-kernel
    let u1 = calc.abs(x1 - tx) / bw
    let u2 = calc.abs(x2 - tx) / bw
    let w1 = if u1 < 1.0 { calc.pow(1.0 - calc.pow(u1, 3), 3) } else { 0.0 }
    let w2 = if u2 < 1.0 { calc.pow(1.0 - calc.pow(u2, 3), 3) } else { 0.0 }
    line(
      (x1, bot-y0 + w1 * bot-h),
      (x2, bot-y0 + w2 * bot-h),
      stroke: 1.2pt + congaree,
    )
  }

  // Label kernel curve
  content((tx + 1.5, bot-y0 + bot-h * 0.85 + 0.2), text(fill: congaree, size: 7.5pt, style: "italic")[tricube kernel])

  // Bandwidth bracket between panels
  let bracket-y = top-y0 - 0.3
  line((tx - bw, bracket-y), (tx + bw, bracket-y), stroke: 0.7pt + congaree, mark: (start: "|", end: "|"))
  content((tx, bracket-y - 0.25), text(fill: congaree, size: 7.5pt)[bandwidth $h$])

  // Title
  content(
    (4.2, -0.7),
    text(fill: black90, size: 10pt, weight: "bold")[LOESS (Cleveland, 1979)],
  )
})
