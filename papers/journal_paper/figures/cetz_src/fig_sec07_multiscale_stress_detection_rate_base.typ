#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black50, black30

#let white = rgb("#FFFFFF")

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let detection-color(value) = {
  let t = calc.min(calc.max(value, 0.0), 1.0)
  color.mix((garnet, t * 100%), (white, (1.0 - t) * 100%))
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.9A multi-scale stress detection rate"
  let heatmap = data.at("multiscale_stress").at("heatmap")
  let radii = heatmap.at("radii")
  let feature-scales = heatmap.at("feature_scales_px")
  let matrix = heatmap.at("detection_rate")
  let coverage = data.at("multiscale_stress").at("full_coverage_by_radius")

  let cell-w = 0.68
  let cell-h = 0.36
  let left-pad = 0.92
  let bottom-pad = 0.70
  let bar-pad = 0.52
  let n-cols = feature-scales.len()
  let n-rows = radii.len()

  cetz.canvas({
    import cetz.draw: *

    content((left-pad + n-cols * cell-w / 2, bottom-pad + n-rows * cell-h + 1.00), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((left-pad + n-cols * cell-w / 2, bottom-pad + n-rows * cell-h + 0.52), text(fill: black70, size: 8.2pt)[#subtitle])

    for row in range(n-rows) {
      for col in range(n-cols) {
        let x = left-pad + col * cell-w
        let y = bottom-pad + (n-rows - 1 - row) * cell-h
        let value = matrix.at(row).at(col)
        rect((x, y), (x + cell-w, y + cell-h), fill: detection-color(value), stroke: 0.28pt + black50)
      }
    }

    for col in range(n-cols) {
      let x = left-pad + col * cell-w + cell-w / 2
      content((x, bottom-pad - 0.22), text(fill: black90, size: 7.2pt)[#feature-scales.at(col)])
    }
    content((left-pad + n-cols * cell-w / 2, bottom-pad - 0.56), text(fill: black90, size: 8.6pt)[Feature scale (px)])

    for row in range(n-rows) {
      let y = bottom-pad + (n-rows - 1 - row) * cell-h + cell-h / 2
      content((left-pad - 0.18, y), text(fill: black90, size: 6.8pt)[#radii.at(row)], anchor: "east")
    }
    content((0.18, bottom-pad + n-rows * cell-h / 2), angle: 90deg, text(fill: black90, size: 8.6pt)[Support radius $r$ (px)])

    let bar-x = left-pad + n-cols * cell-w + bar-pad
    let bar-w = 0.24
    let bar-h = n-rows * cell-h
    let steps = 48
    let step-h = bar-h / steps
    for idx in range(steps) {
      let t = idx / (steps - 1)
      rect(
        (bar-x, bottom-pad + idx * step-h),
        (bar-x + bar-w, bottom-pad + (idx + 1) * step-h),
        fill: detection-color(t),
        stroke: none,
      )
    }
    rect((bar-x, bottom-pad), (bar-x + bar-w, bottom-pad + bar-h), stroke: 0.45pt + black50)
    content((bar-x + bar-w + 0.16, bottom-pad), text(fill: black90, size: 7.0pt)[0.0], anchor: "west")
    content((bar-x + bar-w + 0.16, bottom-pad + 0.5 * bar-h), text(fill: black90, size: 7.0pt)[0.5], anchor: "west")
    content((bar-x + bar-w + 0.16, bottom-pad + bar-h), text(fill: black90, size: 7.0pt)[1.0], anchor: "west")
    content((bar-x + 0.64, bottom-pad + bar-h + 0.22), text(fill: black90, size: 8.0pt)[Detection rate])

    let cover-pieces = ()
    for rec in coverage {
      let label = if rec.at("all_feature_scales_detected") { "yes" } else { "no" }
      cover-pieces.push("r=" + str(rec.at("radius")) + "→" + label)
    }
    content((left-pad + n-cols * cell-w / 2, bottom-pad + n-rows * cell-h + 0.18), text(fill: black70, size: 6.5pt)[Full coverage by radius. #cover-pieces.join(", ")])
  })
}
