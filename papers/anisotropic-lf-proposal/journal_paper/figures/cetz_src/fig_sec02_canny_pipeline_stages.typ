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

#cetz.canvas({
  import cetz.draw: *

  let box-w = 2.4
  let box-h = 2.6
  let gap = 1.0
  let icon-cy-off = -0.95  // icon center offset from box top

  let stage-labels = (
    "Gaussian\nSmoothing",
    [Gradient\ $(G_x, G_y)$],
    [Angle\ $theta = "atan2"$],
    "Non-Max\nSuppression",
    "Hysteresis\nThresholding",
  )

  let stage-nums = ("1", "2", "3", "4", "5")

  for idx in range(5) {
    let x = idx * (box-w + gap)
    let y = 0

    // Box
    let fill-col = if idx == 4 { garnet.lighten(85%) } else { black10 }
    let stroke-col = if idx == 4 { garnet } else { black90 }
    rect((x, y), (x + box-w, y - box-h), fill: fill-col, stroke: 1pt + stroke-col)

    // Stage number
    content((x + 0.25, y - 0.25), text(fill: atlantic, size: 9pt, weight: "bold")[#stage-nums.at(idx)])

    let cx = x + box-w / 2
    let cy = y + icon-cy-off

    // Icons
    if idx == 0 {
      rect((cx - 0.55, cy - 0.55), (cx + 0.55, cy + 0.55), fill: black30, stroke: none)
      rect((cx - 0.4, cy - 0.4), (cx + 0.4, cy + 0.4), fill: black50, stroke: none)
      rect((cx - 0.22, cy - 0.22), (cx + 0.22, cy + 0.22), fill: black90, stroke: none)
    } else if idx == 1 {
      line((cx - 0.5, cy), (cx + 0.5, cy), stroke: 1.8pt + garnet, mark: (end: ">", fill: garnet))
      line((cx, cy - 0.5), (cx, cy + 0.5), stroke: 1.8pt + atlantic, mark: (end: ">", fill: atlantic))
      content((cx + 0.6, cy - 0.15), text(fill: garnet, size: 7pt)[$G_x$])
      content((cx - 0.35, cy + 0.6), text(fill: atlantic, size: 7pt)[$G_y$])
    } else if idx == 2 {
      // Protractor arc
      arc((cx - 0.15, cy - 0.1), start: 0deg, stop: 130deg, radius: 0.5, stroke: 1.5pt + atlantic)
      line((cx - 0.15, cy - 0.1), (cx + 0.35, cy - 0.1), stroke: 0.8pt + black50)
      let angle = 130deg
      let ax = (cx - 0.15) + 0.5 * calc.cos(angle)
      let ay = (cy - 0.1) + 0.5 * calc.sin(angle)
      line((cx - 0.15, cy - 0.1), (ax, ay), stroke: 1pt + garnet)
      content((cx + 0.1, cy + 0.55), text(fill: garnet, size: 8pt)[$theta$])
    } else if idx == 3 {
      line((cx - 0.45, cy + 0.35), (cx + 0.2, cy - 0.3), stroke: 1.2pt + black90)
      line((cx - 0.15, cy + 0.45), (cx + 0.35, cy - 0.1), stroke: 1.2pt + black90)
      line((cx - 0.35, cy - 0.05), (cx + 0.05, cy - 0.5), stroke: 1.2pt + black90)
    } else if idx == 4 {
      line((cx - 0.45, cy + 0.3), (cx - 0.1, cy + 0.05), stroke: 2pt + garnet)
      line((cx - 0.1, cy + 0.05), (cx + 0.25, cy + 0.2), stroke: 2pt + garnet)
      line((cx + 0.25, cy + 0.2), (cx + 0.4, cy - 0.15), stroke: 2pt + garnet)
      line((cx - 0.35, cy - 0.1), (cx, cy - 0.35), stroke: 2pt + garnet)
      line((cx, cy - 0.35), (cx + 0.25, cy - 0.5), stroke: 2pt + garnet)
    }

    // Label at bottom of box
    content((cx, y - box-h + 0.4), text(fill: black90, size: 8pt)[#stage-labels.at(idx)])

    // Arrows between stages
    if idx < 4 {
      let ax1 = x + box-w + 0.08
      let ax2 = x + box-w + gap - 0.08
      let ay = y - box-h / 2
      line((ax1, ay), (ax2, ay), stroke: 1.2pt + black90, mark: (end: ">", fill: black90))
    }
  }

  // Annotations below boxes (single row, no sub-labels)
  let ann-y = -box-h - 0.5

  let x1-center = 0 * (box-w + gap) + box-w / 2
  content((x1-center, ann-y), text(fill: garnet, size: 7.5pt, style: "italic")[Smoothing blurs weak edges])

  let x3-center = 2 * (box-w + gap) + box-w / 2
  content((x3-center, ann-y), text(fill: garnet, size: 7.5pt, style: "italic")[Angular error at non-cardinal directions])

  // Title
  let total-w = 5 * box-w + 4 * gap
  content((total-w / 2, 1.0), text(fill: black90, size: 11pt, weight: "bold")[Canny Edge Detector (1986)])
})
