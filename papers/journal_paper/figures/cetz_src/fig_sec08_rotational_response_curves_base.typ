#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.3pt)

#let method-color(name) = {
  if name == "wvf" { return garnet }
  if name == "scharr" { return atlantic }
  if name == "square_sg" { return congaree }
  if name == "dog" { return horseshoe }
  if name == "farid_simoncelli" { return honeycomb }
  if name == "sobel" { return rose }
  if name == "prewitt" { return black70 }
  black90
}

#let fmt(v) = str(calc.round(v, digits: 4))

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let methods = data.at("methods")
  let order = data.at("method_order")

  cetz.canvas({
    import cetz.draw: *

    let cols = 2
    let rows = calc.ceil(order.len() / cols)
    let panel-w = 4.0
    let panel-h = 2.0
    let col-gap = 0.9
    let row-gap = 0.8
    let ox = 0.95
    let oy = 0.8
    let total-w = cols * panel-w + (cols - 1) * col-gap
    let total-h = rows * panel-h + (rows - 1) * row-gap

    content((ox + total-w / 2, oy + total-h + 1.0), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + total-w / 2, oy + total-h + 0.55), text(fill: black70, size: 8.0pt)[#subtitle])

    for idx in range(order.len()) {
      let method = order.at(idx)
      let method-data = methods.at(method)
      let curve = method-data.at("step").at("clean_curve")
      let panel-col = calc.rem(idx, cols)
      let panel-row = calc.floor(idx / cols)
      let px = ox + panel-col * (panel-w + col-gap)
      let py = oy + (rows - 1 - panel-row) * (panel-h + row-gap)
      let y-min = 1e30
      let y-max = -1e30
      for row in curve {
        let value = row.at("response")
        if value < y-min { y-min = value }
        if value > y-max { y-max = value }
      }
      let y-pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.01 }
      let y0 = y-min - y-pad
      let y1 = y-max + y-pad
      let anisotropy = method-data.at("step").at("anisotropy_by_snr").at("inf")
      let tx(theta) = px + theta / 180.0 * panel-w
      let ty(value) = py + (value - y0) / (y1 - y0) * panel-h

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.20), text(fill: black90, size: 7.6pt, weight: "bold")[#method-data.at("label")])
      content((px + panel-w / 2, py + panel-h + 0.02), text(fill: black70, size: 6.4pt)[A = #fmt(anisotropy)])

      let theta-ticks = (0.0, 45.0, 90.0, 135.0, 180.0)
      for tick in theta-ticks {
        let x = tx(tick)
        line((x, py), (x, py + panel-h), stroke: 0.18pt + black30)
        if tick < 180.0 {
          content((x, py - 0.18), text(fill: black70, size: 6.0pt)[#str(tick)])
        }
      }
      let y-ticks = (y0, y0 + 0.5 * (y1 - y0), y1)
      for tick in y-ticks {
        let y = ty(tick)
        line((px, y), (px + panel-w, y), stroke: 0.18pt + black30)
      }

      for point-idx in range(curve.len() - 1) {
        let a = curve.at(point-idx)
        let b = curve.at(point-idx + 1)
        line(
          (tx(a.at("theta_deg")), ty(a.at("response"))),
          (tx(b.at("theta_deg")), ty(b.at("response"))),
          stroke: 0.8pt + method-color(method),
        )
      }

      if panel-col == 0 {
        content((px - 0.18, py), text(fill: black70, size: 6.0pt)[#fmt(y0)], anchor: "east")
        content((px - 0.18, py + panel-h), text(fill: black70, size: 6.0pt)[#fmt(y1)], anchor: "east")
      }
      if panel-row == rows - 1 {
        content((px + panel-w / 2, py - 0.45), text(fill: black90, size: 7.0pt)[Orientation (deg)])
      }
    }

    content((0.18, oy + total-h / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Peak response $r(theta)$])
  })
}
