#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let rose = rgb("#CC2E40")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  // Data points: (x, y) with scatter around a curve
  let data = (
    (0.3, 1.8),
    (0.9, 2.3),
    (1.4, 2.1),
    (1.9, 2.7),
    (2.3, 3.1),
    (2.8, 2.9),
    (3.2, 3.4),
    (3.6, 3.7),  // target point (index 7)
    (4.0, 3.9),
    (4.4, 3.6),
    (4.9, 4.1),
    (5.3, 4.3),
    (5.8, 3.9),
    (6.2, 4.4),
    (6.7, 4.6),
  )

  let target-idx = 7
  let (tx, ty) = data.at(target-idx)
  let bw = 2.0  // kernel bandwidth

  // X-axis baseline
  line(
    (-0.1, 0.8),
    (7.1, 0.8),
    stroke: 0.4pt + black30,
  )

  // Draw tricube kernel bell curve centered on target x
  // Kernel drawn below data but sharing the same x-axis
  let kernel-base-y = 0.8
  let kernel-h = 1.3

  // Fill under kernel
  let fill-pts = ()
  let n-kernel = 50
  for i in range(n-kernel + 1) {
    let x = tx - bw + 2 * bw * i / n-kernel
    let u = calc.abs(x - tx) / bw
    let w = if u < 1.0 { calc.pow(1.0 - calc.pow(u, 3), 3) } else { 0.0 }
    fill-pts.push((x, kernel-base-y + w * kernel-h))
  }

  let closed-pts = ((tx - bw, kernel-base-y),)
  closed-pts += fill-pts
  closed-pts.push((tx + bw, kernel-base-y))

  line(
    ..closed-pts,
    close: true,
    fill: congaree.lighten(88%),
    stroke: none,
  )

  // Kernel outline
  for i in range(fill-pts.len() - 1) {
    line(fill-pts.at(i), fill-pts.at(i + 1), stroke: 1.2pt + congaree)
  }

  // Label kernel peak
  content(
    (tx, kernel-base-y + kernel-h + 0.2),
    text(fill: congaree, size: 7.5pt)[tricube kernel],
  )

  // Draw vertical weight lines from baseline to kernel for data points within bandwidth
  for i in range(data.len()) {
    let (px, py) = data.at(i)
    let dist = calc.abs(px - tx)
    let u = dist / bw
    if u < 1.0 {
      let w = calc.pow(1.0 - calc.pow(u, 3), 3)
      let ky = kernel-base-y + w * kernel-h
      line(
        (px, kernel-base-y),
        (px, ky),
        stroke: (dash: "dotted", paint: congaree.lighten(40%), thickness: 0.5pt),
      )
    }
  }

  // Draw data points with size/shade based on distance to target
  for i in range(data.len()) {
    let (px, py) = data.at(i)
    let dist = calc.abs(px - tx)
    let u = dist / bw

    let (rad, clr) = if i == target-idx {
      (0.12, garnet)
    } else if u < 1.0 {
      let w = calc.pow(1.0 - calc.pow(u, 3), 3)
      (0.04 + 0.06 * w, black90.lighten(55% - 45% * w))
    } else {
      (0.035, black30)
    }

    circle(
      (px, py),
      radius: rad,
      fill: clr,
      stroke: if i == target-idx { 0.8pt + garnet } else { 0.3pt + black50 },
    )
  }

  // Local fitted line through the weighted region
  line(
    (2.0, 2.85),
    (5.2, 4.15),
    stroke: 2pt + atlantic,
  )

  // Annotation
  content(
    (3.5, 0.2),
    text(fill: black90, size: 8pt, style: "italic")[distance-weighted polynomial fit],
  )

  // Title
  content(
    (3.5, -0.4),
    text(fill: black90, size: 11pt, weight: "bold")[LOESS (Cleveland, 1979)],
  )
})
