#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.8pt)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.11 mean error magnitude by rotation angle"
  let rows = data.at("per_rotation_error")
  let max-error = 0.0
  for row in rows {
    if row.at("mean_error_magnitude") > max-error { max-error = row.at("mean_error_magnitude") }
  }
  let scale = if max-error > 0.0 { max-error } else { 1.0 }

  cetz.canvas({
    import cetz.draw: *

    let cx = 4.6
    let cy = 4.2
    let radius = 2.8

    content((cx, cy + radius + 1.05), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((cx, cy + radius + 0.60), text(fill: black70, size: 8.2pt)[#subtitle])

    for ring in (0.25, 0.5, 0.75, 1.0) {
      circle((cx, cy), radius: ring * radius, stroke: 0.28pt + black30, fill: none)
    }
    line((cx - radius - 0.2, cy), (cx + radius + 0.2, cy), stroke: 0.35pt + black30)
    line((cx, cy - radius - 0.2), (cx, cy + radius + 0.2), stroke: 0.35pt + black30)

    for angle in (0, 45, 90, 135, 180, 225, 270, 315) {
      let rad = angle * calc.pi / 180
      let px = cx + (radius + 0.32) * calc.cos(rad)
      let py = cy + (radius + 0.32) * calc.sin(rad)
      content((px, py), text(fill: black70, size: 6.4pt)[#angle], anchor: "center")
    }

    let points = ()
    for row in rows {
      let rad = row.at("angle_deg") * calc.pi / 180
      let rr = radius * row.at("mean_error_magnitude") / scale
      points.push((cx + rr * calc.cos(rad), cy + rr * calc.sin(rad)))
    }
    for idx in range(points.len() - 1) {
      line(points.at(idx), points.at(idx + 1), stroke: 1.15pt + garnet)
    }
    if points.len() > 1 {
      line(points.at(points.len() - 1), points.at(0), stroke: 1.15pt + garnet)
    }
    for point in points {
      circle(point, radius: 0.04, fill: garnet)
    }

    content((cx + radius + 1.15, cy + 0.22), text(fill: black90, size: 7.0pt, weight: "bold")[Reference])
    content((cx + radius + 1.15, cy - 0.02), text(fill: black70, size: 6.3pt)[max error = #str(calc.round(scale, digits: 6))], anchor: "center")
    content((cx, cy - radius - 0.62), text(fill: black70, size: 6.2pt)[Mean cross-rotation variance = #str(calc.round(data.at("mean_cross_rotation_variance"), digits: 6))])
    content((cx, cy - radius - 0.90), text(fill: black70, size: 6.2pt)[95th percentile variance = #str(calc.round(data.at("p95_cross_rotation_variance"), digits: 6))])
  })
}
