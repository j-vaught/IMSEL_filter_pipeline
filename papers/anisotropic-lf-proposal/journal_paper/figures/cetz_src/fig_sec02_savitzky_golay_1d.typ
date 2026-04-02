#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  // Signal bar heights (11 bars, noisy with underlying trend)
  let heights = (1.0, 1.4, 0.9, 1.6, 2.0, 2.4, 2.1, 2.7, 2.5, 3.0, 3.2)
  let n = heights.len()
  let bar-w = 0.55
  let gap = 0.12
  let step = bar-w + gap

  // Window: bars 2..8 (7 bars)
  let win-start = 2
  let win-end = 8
  let center = 5

  // Draw window background rectangle — horseshoe green, transparent overlay
  let wx0 = win-start * step - gap / 2
  let wx1 = (win-end + 1) * step - gap / 2
  rect(
    (wx0, -0.1),
    (wx1, 3.5),
    fill: horseshoe.lighten(90%),
    stroke: 0.8pt + horseshoe,
  )

  // Draw all bars
  for i in range(n) {
    let x = i * step
    let h = heights.at(i)
    let clr = if i >= win-start and i <= win-end { garnet } else { black50 }
    rect(
      (x, 0),
      (x + bar-w, h),
      fill: clr,
      stroke: 0.4pt + black90,
    )
  }

  // Smooth polynomial curve over the window (degree 3 approximation)
  let poly-vals = (1.05, 1.45, 1.85, 2.18, 2.40, 2.52, 2.58)
  for i in range(6) {
    let idx0 = win-start + i
    let idx1 = win-start + i + 1
    let x0 = idx0 * step + bar-w / 2
    let x1 = idx1 * step + bar-w / 2
    let y0 = poly-vals.at(i)
    let y1 = poly-vals.at(i + 1)
    line((x0, y0), (x1, y1), stroke: 2.2pt + atlantic)
  }

  // Small dots on poly curve at bar centers
  for i in range(7) {
    let idx = win-start + i
    let x = idx * step + bar-w / 2
    let y = poly-vals.at(i)
    circle((x, y), radius: 0.05, fill: atlantic, stroke: none)
  }

  // "degree d" label above the polynomial curve, inside the window
  content(
    ((wx0 + wx1) / 2, 3.2),
    text(fill: atlantic, size: 8.5pt)[degree $d$ polynomial],
  )

  // Tangent arrow at center bar -- ABOVE the polynomial curve
  let cx = center * step + bar-w / 2
  let cy = poly-vals.at(center - win-start)
  // Slope from poly
  let slope = (poly-vals.at(4) - poly-vals.at(2)) / (2 * step)

  // Draw tangent as a short line segment with arrow, offset ABOVE curve
  let tang-half = 0.55
  let offset-y = 0.45  // above the curve
  line(
    (cx - tang-half, cy + offset-y - tang-half * slope),
    (cx + tang-half, cy + offset-y + tang-half * slope),
    stroke: 2pt + rose,
    mark: (end: "stealth", fill: rose, size: 0.22),
  )

  // Dashed connector from curve point up to tangent
  line(
    (cx, cy),
    (cx, cy + offset-y),
    stroke: (dash: "dotted", paint: rose, thickness: 0.6pt),
  )

  // Label derivative arrow
  content(
    (cx + tang-half + 0.15, cy + offset-y + tang-half * slope + 0.2),
    anchor: "west",
    text(fill: rose, size: 9pt, weight: "bold")[$f'_"center"$],
  )

  // Window bracket annotation below
  let brace-y = -0.5
  line(
    (wx0, brace-y),
    (wx1, brace-y),
    stroke: 0.8pt + black90,
    mark: (start: "|", end: "|"),
  )
  content(
    ((wx0 + wx1) / 2, brace-y - 0.35),
    text(fill: black90, size: 8.5pt)[window $= 2m + 1$],
  )
})
