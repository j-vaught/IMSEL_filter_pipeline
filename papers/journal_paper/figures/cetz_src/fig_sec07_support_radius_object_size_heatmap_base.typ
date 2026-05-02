#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black50, black30

#let white = rgb("#FFFFFF")

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let magnitude-to-color(value, min-value, max-value) = {
  let span = if max-value > min-value { max-value - min-value } else { 1.0 }
  let t = (value - min-value) / span
  let t = calc.min(calc.max(t, 0.0), 1.0)
  color.mix((garnet, t * 100%), (white, (1.0 - t) * 100%))
}

#let render(data-path, class-key) = {
  let data = json(data-path)
  let class-data = data.at("subclasses").at(class-key)
  let widths = class-data.at("heatmap").at("feature_widths")
  let radii = class-data.at("heatmap").at("radii")
  let matrix = class-data.at("heatmap").at("detection_magnitude")
  let title = data.at("title")
  let subtitle = data.at("subtitle") + " [" + class-data.at("label") + "]"

  let min-value = 1e30
  let max-value = 0.0
  for row in matrix {
    for value in row {
      if value < min-value { min-value = value }
      if value > max-value { max-value = value }
    }
  }

  let cell-w = 0.75
  let cell-h = 0.38
  let left-pad = 0.95
  let bottom-pad = 0.65
  let bar-pad = 0.50
  let n-cols = widths.len()
  let n-rows = radii.len()

  cetz.canvas({
    import cetz.draw: *

    content((left-pad + n-cols * cell-w / 2, bottom-pad + n-rows * cell-h + 1.00), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((left-pad + n-cols * cell-w / 2, bottom-pad + n-rows * cell-h + 0.52), text(fill: black70, size: 8.3pt)[#subtitle])

    for row in range(n-rows) {
      for col in range(n-cols) {
        let x = left-pad + col * cell-w
        let y = bottom-pad + (n-rows - 1 - row) * cell-h
        let value = matrix.at(row).at(col)
        let fill-color = magnitude-to-color(value, min-value, max-value)
        rect((x, y), (x + cell-w, y + cell-h), fill: fill-color, stroke: 0.28pt + black50)
      }
    }

    for col in range(n-cols) {
      let x = left-pad + col * cell-w + cell-w / 2
      content((x, bottom-pad - 0.22), text(fill: black90, size: 7.5pt)[#widths.at(col)])
    }
    content((left-pad + n-cols * cell-w / 2, bottom-pad - 0.55), text(fill: black90, size: 8.6pt)[Feature width (px)])

    for row in range(n-rows) {
      let y = bottom-pad + (n-rows - 1 - row) * cell-h + cell-h / 2
      content((left-pad - 0.22, y), text(fill: black90, size: 7.2pt)[#radii.at(row)], anchor: "east")
    }
    content((0.18, bottom-pad + n-rows * cell-h / 2), angle: 90deg, text(fill: black90, size: 8.6pt)[Support radius $r$ (px)])

    let bar-x = left-pad + n-cols * cell-w + bar-pad
    let bar-w = 0.24
    let bar-h = n-rows * cell-h
    let steps = 48
    let step-h = bar-h / steps
    for idx in range(steps) {
      let t = idx / (steps - 1)
      let value = min-value + t * (max-value - min-value)
      rect(
        (bar-x, bottom-pad + idx * step-h),
        (bar-x + bar-w, bottom-pad + (idx + 1) * step-h),
        fill: magnitude-to-color(value, min-value, max-value),
        stroke: none,
      )
    }
    rect((bar-x, bottom-pad), (bar-x + bar-w, bottom-pad + bar-h), stroke: 0.45pt + black50)
    content((bar-x + bar-w + 0.16, bottom-pad), text(fill: black90, size: 7.2pt)[#str(calc.round(min-value, digits: 4))], anchor: "west")
    content((bar-x + bar-w + 0.16, bottom-pad + bar-h / 2), text(fill: black90, size: 7.2pt)[#str(calc.round((min-value + max-value) / 2, digits: 4))], anchor: "west")
    content((bar-x + bar-w + 0.16, bottom-pad + bar-h), text(fill: black90, size: 7.2pt)[#str(calc.round(max-value, digits: 4))], anchor: "west")
    content((bar-x + 0.65, bottom-pad + bar-h + 0.22), text(fill: black90, size: 8.2pt)[Detection magnitude])
  })
}
