#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30, black50

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.2pt)

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

#let render(data-path) = {
  let data = json(data-path)
  let methods = data.at("methods")
  let order = data.at("method_order")
  let cols = 2
  let rows = calc.ceil(order.len() / cols)

  cetz.canvas({
    import cetz.draw: *

    let panel-w = 4.0
    let panel-h = 1.95
    let col-gap = 0.85
    let row-gap = 0.75
    let ox = 0.95
    let oy = 0.8
    let total-w = cols * panel-w + (cols - 1) * col-gap
    let total-h = rows * panel-h + (rows - 1) * row-gap

    content((ox + total-w / 2, oy + total-h + 0.95), text(fill: black90, size: 10pt, weight: "bold")[Clean phase-dependent localisation bias])
    content((ox + total-w / 2, oy + total-h + 0.52), text(fill: black70, size: 8.0pt)[Mean signed offset by sub-pixel phase])

    for idx in range(order.len()) {
      let method = order.at(idx)
      let method-data = methods.at(method)
      let curve = method-data.at("clean_phase_profile")
      let panel-col = calc.rem(idx, cols)
      let panel-row = calc.floor(idx / cols)
      let px = ox + panel-col * (panel-w + col-gap)
      let py = oy + (rows - 1 - panel-row) * (panel-h + row-gap)
      let y-min = 1e30
      let y-max = -1e30
      for row in curve {
        let value = row.at("mean_offset")
        if value < y-min { y-min = value }
        if value > y-max { y-max = value }
      }
      let y-pad = if y-max > y-min { 0.12 * (y-max - y-min) } else { 0.01 }
      let y0 = y-min - y-pad
      let y1 = y-max + y-pad
      let tx(v) = px + v * panel-w
      let ty(v) = py + (v - y0) / (y1 - y0) * panel-h

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.5pt, weight: "bold")[#method-data.at("label")])

      line((px, ty(0.0)), (px + panel-w, ty(0.0)), stroke: 0.25pt + black50)
      for idx2 in range(curve.len() - 1) {
        let a = curve.at(idx2)
        let b = curve.at(idx2 + 1)
        line((tx(a.at("phase_px")), ty(a.at("mean_offset"))), (tx(b.at("phase_px")), ty(b.at("mean_offset"))), stroke: 0.75pt + method-color(method))
      }

      if panel-col == 0 {
        content((px - 0.16, ty(y0)), text(fill: black70, size: 6.0pt)[#str(calc.round(y0, digits: 3))], anchor: "east")
        content((px - 0.16, ty(y1)), text(fill: black70, size: 6.0pt)[#str(calc.round(y1, digits: 3))], anchor: "east")
      }
      if panel-row == rows - 1 {
        content((px + panel-w / 2, py - 0.42), text(fill: black90, size: 7.0pt)[Phase (px)])
      }
    }

    content((0.18, oy + total-h / 2), angle: 90deg, text(fill: black90, size: 7.3pt)[Mean signed offset (px)])
  })
}
