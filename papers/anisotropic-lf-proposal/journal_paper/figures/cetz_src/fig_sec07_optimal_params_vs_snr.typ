#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90  = rgb("#363636")
#let black70  = rgb("#5C5C5C")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let black10  = rgb("#ECECEC")
#let white    = rgb("#FFFFFF")

// Extracted data: (SNR_dB, optimal_Np, optimal_m)
// Aggregated across all noise types (gaussian, salt_pepper, poisson, speckle, uniform)
// and all datasets (BSDS500, BIPED v1, BIPED v2, UDED)
#let data = (
  (-5.23, 190.3, 15.7),
  (-3.01, 186.2, 14.2),
  (0.00, 149.7, 9.5),
  (1.76, 124.5, 6.9),
  (3.01, 109.8, 5.6),
  (4.77, 103.8, 3.9),
  (6.02, 93.0, 3.2),
)

// Clean baseline (infinite SNR, no noise)
#let clean-Np = 53.2
#let clean-m = 2.1

#cetz.canvas({
  import cetz.draw: *

  // Two-panel layout (side by side)
  let pw = 5.0   // panel width
  let ph = 4.0   // panel height
  let gap = 2.0  // gap between panels
  let ox1 = 1.5  // left panel origin x
  let ox2 = ox1 + pw + gap  // right panel origin x
  let oy = 1.0   // origin y

  // X axis range: SNR from -6 to 8 dB
  let snr-min = -6.0
  let snr-max = 8.0

  // Y axis ranges
  let np-min = 0.0
  let np-max = 220.0
  let m-min = 0.0
  let m-max = 20.0

  // Transform functions
  let tx1(v) = ox1 + (v - snr-min) / (snr-max - snr-min) * pw
  let tx2(v) = ox2 + (v - snr-min) / (snr-max - snr-min) * pw
  let ty-np(v) = oy + (v - np-min) / (np-max - np-min) * ph
  let ty-m(v) = oy + (v - m-min) / (m-max - m-min) * ph

  // ============ LEFT PANEL: Optimal Np ============
  // Axes
  line((ox1, oy), (ox1 + pw, oy), stroke: 0.8pt + black90)
  line((ox1, oy), (ox1, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox1 + pw/2, oy - 0.7), text(fill: black90, size: 9pt)[SNR (dB)])

  // Y-axis label
  content(
    (ox1 - 1.0, oy + ph/2),
    text(fill: black90, size: 9pt)[Optimal $N_p$],
    angle: 90deg,
  )

  // X ticks
  let x-ticks = (-6, -4, -2, 0, 2, 4, 6, 8)
  for v in x-ticks {
    let x = tx1(v)
    line((x, oy), (x, oy - 0.08), stroke: 0.5pt + black70)
    content((x, oy - 0.28), text(fill: black70, size: 7pt)[#v])
    line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
  }

  // Y ticks for Np
  let np-ticks = (0, 50, 100, 150, 200)
  for v in np-ticks {
    let y = ty-np(v)
    line((ox1, y), (ox1 - 0.08, y), stroke: 0.5pt + black70)
    content((ox1 - 0.35, y), text(fill: black70, size: 7pt)[#v])
    line((ox1, y), (ox1 + pw, y), stroke: 0.2pt + black30)
  }

  // Plot Np data points and line
  let np-pts = data.map(d => (tx1(d.at(0)), ty-np(d.at(1))))
  for i in range(np-pts.len() - 1) {
    line(np-pts.at(i), np-pts.at(i + 1), stroke: 1.2pt + garnet)
  }
  for pt in np-pts {
    circle(pt, radius: 0.1, fill: garnet, stroke: 0.5pt + garnet.darken(20%))
  }

  // Clean baseline horizontal dashed line
  let clean-np-y = ty-np(clean-Np)
  line((ox1, clean-np-y), (ox1 + pw, clean-np-y), stroke: (paint: black50, thickness: 0.8pt, dash: "dashed"))
  content((ox1 + pw + 0.15, clean-np-y), text(fill: black50, size: 7pt)[clean], anchor: "west")

  // ============ RIGHT PANEL: Optimal m ============
  // Axes
  line((ox2, oy), (ox2 + pw, oy), stroke: 0.8pt + black90)
  line((ox2, oy), (ox2, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox2 + pw/2, oy - 0.7), text(fill: black90, size: 9pt)[SNR (dB)])

  // Y-axis label
  content(
    (ox2 - 1.0, oy + ph/2),
    text(fill: black90, size: 9pt)[Optimal $m$],
    angle: 90deg,
  )

  // X ticks
  for v in x-ticks {
    let x = tx2(v)
    line((x, oy), (x, oy - 0.08), stroke: 0.5pt + black70)
    content((x, oy - 0.28), text(fill: black70, size: 7pt)[#v])
    line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
  }

  // Y ticks for m
  let m-ticks = (0, 5, 10, 15, 20)
  for v in m-ticks {
    let y = ty-m(v)
    line((ox2, y), (ox2 - 0.08, y), stroke: 0.5pt + black70)
    content((ox2 - 0.35, y), text(fill: black70, size: 7pt)[#v])
    line((ox2, y), (ox2 + pw, y), stroke: 0.2pt + black30)
  }

  // Plot m data points and line
  let m-pts = data.map(d => (tx2(d.at(0)), ty-m(d.at(2))))
  for i in range(m-pts.len() - 1) {
    line(m-pts.at(i), m-pts.at(i + 1), stroke: 1.2pt + atlantic)
  }
  for pt in m-pts {
    circle(pt, radius: 0.1, fill: atlantic, stroke: 0.5pt + atlantic.darken(20%))
  }

  // Clean baseline horizontal dashed line
  let clean-m-y = ty-m(clean-m)
  line((ox2, clean-m-y), (ox2 + pw, clean-m-y), stroke: (paint: black50, thickness: 0.8pt, dash: "dashed"))
  content((ox2 + pw + 0.15, clean-m-y), text(fill: black50, size: 7pt)[clean], anchor: "west")

  // ============ LEGEND ============
  let lx = ox2 + pw - 2.5
  let ly = oy + ph - 0.3
  let ls = 0.4

  rect((lx - 0.15, ly + 0.25), (lx + 2.4, ly - 2*ls - 0.15), fill: white, stroke: 0.4pt + black30)

  // Np legend entry
  line((lx, ly), (lx + 0.4, ly), stroke: 1.2pt + garnet)
  circle((lx + 0.2, ly), radius: 0.08, fill: garnet, stroke: 0.4pt + garnet.darken(20%))
  content((lx + 0.55, ly), text(fill: black90, size: 8pt)[Optimal $N_p$], anchor: "west")

  // m legend entry
  line((lx, ly - ls), (lx + 0.4, ly - ls), stroke: 1.2pt + atlantic)
  circle((lx + 0.2, ly - ls), radius: 0.08, fill: atlantic, stroke: 0.4pt + atlantic.darken(20%))
  content((lx + 0.55, ly - ls), text(fill: black90, size: 8pt)[Optimal $m$], anchor: "west")
})
