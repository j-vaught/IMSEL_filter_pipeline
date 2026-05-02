#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, horseshoe, honeycomb, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let series-style(index) = {
  let palette = (
    (paint: garnet, thickness: 1.5pt),
    (paint: rose, thickness: 1.3pt),
    (paint: atlantic, thickness: 1.3pt),
    (paint: congaree, thickness: 1.3pt),
    (paint: horseshoe, thickness: 1.3pt),
    (paint: honeycomb, thickness: 1.3pt),
    (paint: black90, thickness: 1.2pt, dash: "dashed"),
    (paint: black70, thickness: 1.2pt, dash: "dashed"),
    (paint: rose, thickness: 1.2pt, dash: "dotted"),
    (paint: atlantic, thickness: 1.2pt, dash: "dotted"),
    (paint: congaree, thickness: 1.2pt, dash: "dash-dotted"),
    (paint: horseshoe, thickness: 1.2pt, dash: "dash-dotted"),
    (paint: black50, thickness: 1.2pt, dash: "dotted"),
  )
  palette.at(index)
}

#let tx-map(radius) = calc.log(radius) / calc.log(2)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let series = data.at("rmse_series")

  let x0 = 1.0
  let x1 = 7.0
  let y-min = 1e30
  let y-max = 0.0
  for entry in series {
    for point in entry.at("points") {
      let y = point.at("grad_rmse")
      if y < y-min { y-min = y }
      if y > y-max { y-max = y }
    }
  }
  let y-span = if y-max > y-min { y-max - y-min } else { 1.0 }
  let y0 = y-min - 0.08 * y-span
  let y1 = y-max + 0.10 * y-span

  cetz.canvas({
    import cetz.draw: *

    let pw = 10.4
    let ph = 6.1
    let ox = 1.35
    let oy = 0.98

    let tx(v) = ox + (v - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.4pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.68), text(fill: black90, size: 9pt)[Support radius $r$ (px)])
    content((ox - 1.18, oy + ph / 2), text(fill: black90, size: 9pt)[Gradient-vector RMSE], angle: 90deg)

    let radius-ticks = (2, 4, 8, 16, 32, 64, 128)
    for radius in radius-ticks {
      let x = tx(tx-map(radius))
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.1pt)[#radius])
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
    }

    let y-ticks = (
      y0,
      y0 + 0.25 * (y1 - y0),
      y0 + 0.50 * (y1 - y0),
      y0 + 0.75 * (y1 - y0),
      y1,
    )
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.1, y), stroke: 0.45pt + black70)
      content((ox - 0.76, y), text(fill: black70, size: 7.1pt)[#str(calc.round(tick, digits: 5))])
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
          (tx(tx-map(a.at("radius"))), ty(a.at("grad_rmse"))),
          (tx(tx-map(b.at("radius"))), ty(b.at("grad_rmse"))),
          stroke: style,
        )
      }
      for point in points {
        circle((tx(tx-map(point.at("radius"))), ty(point.at("grad_rmse"))), radius: 0.04, fill: style.paint)
      }
    }

    let lx = ox + 0.28
    let ly = oy + ph - 0.36
    rect((lx - 0.18, ly + 0.22), (lx + 2.2, ly - 3.18), fill: white, stroke: 0.4pt + black30)
    content((lx + 1.0, ly + 0.02), text(fill: black90, size: 8pt, weight: "bold")[SNR])
    for i in range(series.len()) {
      let entry = series.at(i)
      let y = ly - 0.24 * i
      line((lx, y), (lx + 0.55, y), stroke: series-style(i))
      content((lx + 0.72, y), text(fill: black90, size: 6.8pt)[#entry.at("label")], anchor: "west")
    }
  })
}
