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

  // Chart area with Y=0 at bottom, Y=chart-h at top
  // In CeTZ, positive Y is up on the page.
  let chart-ox = 1.5
  let chart-oy = 0       // bottom of chart (eta=0)
  let chart-h = 5.5      // top of chart is at y = chart-h (eta=1.0)
  let chart-w = 7.5

  let bar-w = 1.5
  let bar-gap = 1.0

  let bars = (
    ("Fused Stencil", 0.72, ">0.99", garnet),
    ("Rectangular", 0.995, ">0.99", atlantic),
    ("Elliptical", 0.992, ">0.99", congaree),
  )

  // Title at top
  content((chart-ox + chart-w / 2, chart-oy + chart-h + 0.7), text(fill: black90, size: 11pt, weight: "bold")[
    Weight Efficiency Comparison
  ])

  // Y-axis (vertical, bottom to top)
  line((chart-ox, chart-oy), (chart-ox, chart-oy + chart-h), stroke: 0.8pt + black90)
  // X-axis (horizontal baseline at bottom)
  line((chart-ox, chart-oy), (chart-ox + chart-w, chart-oy), stroke: 0.8pt + black90)

  // Y-axis label
  content(
    (chart-ox - 1.2, chart-oy + chart-h / 2),
    angle: 90deg,
    text(fill: black90, size: 10pt)[Weight Efficiency $eta$]
  )

  // Y-axis ticks and grid lines
  for tick-idx in range(6) {
    let tick-val = tick-idx * 0.2
    let ty = chart-oy + tick-val * chart-h
    line((chart-ox - 0.1, ty), (chart-ox, ty), stroke: 0.5pt + black90)
    let tick-label = str(calc.round(tick-val, digits: 1))
    content((chart-ox - 0.25, ty), anchor: "east", text(fill: black90, size: 8pt)[#tick-label])
    // Light grid line
    if tick-val > 0.0 and tick-val < 1.0 {
      line((chart-ox, ty), (chart-ox + chart-w, ty), stroke: 0.3pt + black30)
    }
  }

  // Dashed line at eta = 1.0 (top of chart)
  let ideal-y = chart-oy + 1.0 * chart-h
  line(
    (chart-ox, ideal-y),
    (chart-ox + chart-w, ideal-y),
    stroke: (paint: black90, thickness: 1pt, dash: "dashed"),
  )
  content((chart-ox + chart-w + 0.15, ideal-y), anchor: "west", text(fill: black90, size: 8pt, style: "italic")[
    Ideal (no cancellation)
  ])

  // Draw bars (growing upward from baseline)
  let total-bars-w = 3 * bar-w + 2 * bar-gap
  let start-x = chart-ox + (chart-w - total-bars-w) / 2

  for (i, bar) in bars.enumerate() {
    let (label, value, disp-label, col) = bar
    let bx = start-x + i * (bar-w + bar-gap)
    let bar-top = chart-oy + value * chart-h

    // Bar rectangle from baseline (bottom) upward
    rect(
      (bx, chart-oy),
      (bx + bar-w, bar-top),
      fill: col,
      stroke: 0.5pt + black90,
    )

    // Value label above bar: use custom display label for first bar, >0.99 for others
    let val-label = if i == 0 { str(calc.round(value, digits: 2)) } else { disp-label }
    content((bx + bar-w / 2, bar-top + 0.25), text(fill: black90, size: 9pt, weight: "bold")[#val-label])

    // Category label below x-axis
    content((bx + bar-w / 2, chart-oy - 0.45), text(fill: black90, size: 9pt)[#label])
  }

  // "~28% wasted" annotation: place between fused bar and rectangular bar
  let fused-bx = start-x
  let fused-bar-top = chart-oy + 0.72 * chart-h

  // Position the arrow between the fused and rectangular bars
  let rect-bx = start-x + 1 * (bar-w + bar-gap)
  let ann-x = (fused-bx + bar-w + rect-bx) / 2

  // Vertical line spanning the gap between fused bar top and ideal line
  line(
    (ann-x, fused-bar-top),
    (ann-x, ideal-y),
    stroke: 1.2pt + rose,
  )
  // Arrow tips
  line((ann-x - 0.08, ideal-y - 0.15), (ann-x, ideal-y), stroke: 1.2pt + rose)
  line((ann-x + 0.08, ideal-y - 0.15), (ann-x, ideal-y), stroke: 1.2pt + rose)
  line((ann-x - 0.08, fused-bar-top + 0.15), (ann-x, fused-bar-top), stroke: 1.2pt + rose)
  line((ann-x + 0.08, fused-bar-top + 0.15), (ann-x, fused-bar-top), stroke: 1.2pt + rose)

  // Horizontal ticks at ends
  line((ann-x - 0.12, fused-bar-top), (ann-x + 0.12, fused-bar-top), stroke: 1pt + rose)
  line((ann-x - 0.12, ideal-y), (ann-x + 0.12, ideal-y), stroke: 1pt + rose)

  let mid-y = (fused-bar-top + ideal-y) / 2
  content((ann-x, mid-y + 0.15), anchor: "south", text(fill: rose, size: 8.5pt, weight: "bold")[
    \~28%
  ])
  content((ann-x, mid-y - 0.15), anchor: "north", text(fill: rose, size: 8.5pt, weight: "bold")[
    wasted
  ])
})
