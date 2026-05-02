#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let series-style(index) = {
  let palette = (
    (paint: garnet, thickness: 1.6pt),
    (paint: atlantic, thickness: 1.4pt),
    (paint: congaree, thickness: 1.4pt),
    (paint: rose, thickness: 1.4pt),
  )
  palette.at(index)
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let series = data.at("arc_grad_rmse_plot")

  let x0 = 1
  let x1 = 9
  let y-values = ()
  for entry in series {
    for point in entry.at("points") {
      y-values.push(point.at("grad_rmse"))
    }
  }
  let y-min = calc.min(..y-values)
  let y-max = calc.max(..y-values)
  let y-pad = 0.06 * (y-max - y-min + 1e-12)
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 10.4
    let ph = 6.1
    let ox = 1.35
    let oy = 0.98

    let tx(v) = ox + (v - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.3pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.68), text(fill: black90, size: 9pt)[Polynomial degree $d$])
    content((ox - 1.16, oy + ph / 2), text(fill: black90, size: 9pt)[Arc gradient RMSE], angle: 90deg)

    let x-ticks = (1, 3, 5, 7, 9)
    for tick in x-ticks {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.0pt)[#tick])
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
    }

    let y-ticks = ()
    for alpha in range(6) {
      let value = y0 + alpha * (y1 - y0) / 5
      y-ticks.push(value)
    }
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.1, y), stroke: 0.45pt + black70)
      let label = calc.round(tick * 10000) / 10000
      content((ox - 0.64, y), text(fill: black70, size: 7.0pt)[#label])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    for i in range(series.len()) {
      let entry = series.at(i)
      let points = entry.at("points")
      let style = series-style(i)
      for j in range(points.len() - 1) {
        let a = points.at(j)
        let b = points.at(j + 1)
        line(
          (tx(a.at("degree")), ty(a.at("grad_rmse"))),
          (tx(b.at("degree")), ty(b.at("grad_rmse"))),
          stroke: style,
        )
      }
      for point in points {
        circle((tx(point.at("degree")), ty(point.at("grad_rmse"))), radius: 0.045, fill: style.paint)
      }
    }

    let lx0 = ox + 0.45
    let ly0 = oy + ph - 0.34
    rect((lx0 - 0.18, ly0 + 0.22), (lx0 + 2.2, ly0 - 1.00), fill: white, stroke: 0.4pt + black30)
    content((lx0 + 0.9, ly0 + 0.02), text(fill: black90, size: 8pt, weight: "bold")[Radius])
    for i in range(series.len()) {
      let entry = series.at(i)
      let y = ly0 - 0.24 * i
      line((lx0, y), (lx0 + 0.55, y), stroke: series-style(i))
      content((lx0 + 0.74, y), text(fill: black90, size: 7.1pt)[#entry.at("label")], anchor: "west")
    }
  })
}
