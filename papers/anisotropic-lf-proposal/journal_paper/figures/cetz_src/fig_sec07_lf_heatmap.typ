#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

// ── Brand Colors ────────────────────────────────────────────────────────────
#let garnet   = rgb("#73000A")
#let black90  = rgb("#363636")
#let black70  = rgb("#5C5C5C")
#let black50  = rgb("#A2A2A2")
#let black30  = rgb("#C7C7C7")
#let black10  = rgb("#ECECEC")
#let white    = rgb("#FFFFFF")

// ── Hardcoded LF Data ───────────────────────────────────────────────────────
// ODS as function of Np (support size) vs m (line half-width)
// Averaged across Ns values (18, 36, 72)
#let np-values = (15, 25, 50, 75, 100, 150, 250)
#let m-values = (1, 2, 3, 5, 7, 10, 14, 20)

// ODS data as 2D array (rows=m, cols=Np)
#let ods-data = (
  (0.7288, 0.8062, 0.8273, 0.8365, 0.8330, 0.8063, 0.7647), // m=1
  (0.7826, 0.8222, 0.8296, 0.8304, 0.8271, 0.8006, 0.7609), // m=2
  (0.8090, 0.8287, 0.8223, 0.8188, 0.8145, 0.7903, 0.7542), // m=3
  (0.7998, 0.8030, 0.7906, 0.7812, 0.7764, 0.7553, 0.7320), // m=5
  (0.7655, 0.7625, 0.7471, 0.7384, 0.7341, 0.7177, 0.7016), // m=7
  (0.7096, 0.7043, 0.6919, 0.6856, 0.6817, 0.6715, 0.6645), // m=10
  (0.6518, 0.6452, 0.6372, 0.6344, 0.6314, 0.6265, 0.6207), // m=14
  (0.5969, 0.5922, 0.5854, 0.5844, 0.5820, 0.5782, 0.5714), // m=20
)

// ODS range for color scaling
#let ods-min = 0.57
#let ods-max = 0.84

// ── Layout Parameters ───────────────────────────────────────────────────────
#let cell-w = 0.6
#let cell-h = 0.5
#let n-cols = np-values.len()
#let n-rows = m-values.len()
#let left-pad = 0.8
#let bottom-pad = 0.6
#let bar-pad = 0.45

// ── Color Interpolation ─────────────────────────────────────────────────────
#let ods-to-color(ods) = {
  let t = (ods - ods-min) / (ods-max - ods-min)
  let t = calc.min(calc.max(t, 0.0), 1.0)
  // Sequential colormap: white (low) -> garnet (high)
  color.mix((garnet, t * 100%), (white, (1.0 - t) * 100%))
}

// ── Render ──────────────────────────────────────────────────────────────────
#cetz.canvas({
  import cetz.draw: *

  // Draw heatmap cells
  for row in range(n-rows) {
    for col in range(n-cols) {
      let x = left-pad + col * cell-w
      let y = bottom-pad + (n-rows - 1 - row) * cell-h
      let ods = ods-data.at(row).at(col)
      let fc = ods-to-color(ods)
      rect((x, y), (x + cell-w, y + cell-h), fill: fc, stroke: 0.3pt + black50)

      // Add ODS value text
      let text-color = if ods > 0.72 { white } else { black90 }
      content((x + cell-w / 2.0, y + cell-h / 2.0),
        text(fill: text-color, size: 7pt)[#calc.round(ods, digits: 2)])
    }
  }

  // X-axis labels (Np)
  for col in range(n-cols) {
    let x = left-pad + col * cell-w + cell-w / 2.0
    content((x, bottom-pad - 0.2),
      text(fill: black90, size: 8pt)[#np-values.at(col)])
  }

  // X-axis title
  content((left-pad + n-cols * cell-w / 2.0, bottom-pad - 0.5),
    text(fill: black90, size: 9pt)[$N_p$ (support size)])

  // Y-axis labels (m)
  for row in range(n-rows) {
    let y = bottom-pad + (n-rows - 1 - row) * cell-h + cell-h / 2.0
    content((left-pad - 0.2, y),
      text(fill: black90, size: 8pt)[#m-values.at(row)],
      anchor: "east")
  }

  // Y-axis title
  content((0.15, bottom-pad + n-rows * cell-h / 2.0),
    angle: 90deg,
    text(fill: black90, size: 9pt)[$m$ (line half-width)])

  // Colorbar
  let bar-x = left-pad + n-cols * cell-w + bar-pad
  let bar-w = 0.25
  let bar-h = n-rows * cell-h
  let n-steps = 50
  let step-h = bar-h / n-steps

  for s in range(n-steps) {
    let t = s / (n-steps - 1)
    let fc = ods-to-color(ods-min + t * (ods-max - ods-min))
    rect((bar-x, bottom-pad + s * step-h),
         (bar-x + bar-w, bottom-pad + (s + 1) * step-h),
         fill: fc, stroke: none)
  }
  rect((bar-x, bottom-pad), (bar-x + bar-w, bottom-pad + bar-h), stroke: 0.5pt + black50)

  // Colorbar labels
  content((bar-x + bar-w + 0.15, bottom-pad),
    text(fill: black90, size: 7.5pt)[#calc.round(ods-min, digits: 2)],
    anchor: "west")
  content((bar-x + bar-w + 0.15, bottom-pad + bar-h / 2.0),
    text(fill: black90, size: 7.5pt)[#calc.round((ods-min + ods-max) / 2.0, digits: 2)],
    anchor: "west")
  content((bar-x + bar-w + 0.15, bottom-pad + bar-h),
    text(fill: black90, size: 7.5pt)[#calc.round(ods-max, digits: 2)],
    anchor: "west")

  // Colorbar title
  content((bar-x + bar-w / 2.0 + 0.45, bottom-pad + bar-h + 0.3),
    text(fill: black90, size: 9pt)[ODS])
})
