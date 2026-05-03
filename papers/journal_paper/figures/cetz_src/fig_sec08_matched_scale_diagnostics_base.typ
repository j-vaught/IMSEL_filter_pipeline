#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, congaree, black90, black70, black30, black50

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.2pt)

#let rule-label(key) = {
  if key == "bounding_radius" { return "Bounding radius" }
  if key == "effective_response_width" { return "Effective response width" }
  if key == "support_cardinality" { return "Support cardinality" }
  if key == "white_noise_gain" { return "White-noise gain" }
  "Effective second moment"
}

#let render(data-path) = {
  let data = json(data-path)
  let rules = data.at("rules")
  let order = data.at("rule_order")
  let radii = data.at("config").at("radius_schedule")
  let cols = 2
  let rows = calc.ceil(order.len() / cols)
  let y-min = 1e30
  let y-max = -1e30
  for rule in order {
    for row in rules.at(rule).at("rows") {
      let a = row.at("wvf").at("anisotropy_ratio")
      let b = row.at("square_sg").at("anisotropy_ratio")
      if a < y-min { y-min = a }
      if a > y-max { y-max = a }
      if b < y-min { y-min = b }
      if b > y-max { y-max = b }
    }
  }

  cetz.canvas({
    import cetz.draw: *

    let panel-w = 4.0
    let panel-h = 2.1
    let col-gap = 0.85
    let row-gap = 0.82
    let ox = 0.95
    let oy = 0.8
    let total-w = cols * panel-w + (cols - 1) * col-gap
    let total-h = rows * panel-h + (rows - 1) * row-gap

    content((ox + total-w / 2, oy + total-h + 0.95), text(fill: black90, size: 10pt, weight: "bold")[Matched-scale diagnostics])
    content((ox + total-w / 2, oy + total-h + 0.52), text(fill: black70, size: 8.0pt)[WVF versus square SG at AWGN 10 dB, one panel per matching rule])

    for idx in range(order.len()) {
      let rule = order.at(idx)
      let rows-data = rules.at(rule).at("rows")
      let panel-col = calc.rem(idx, cols)
      let panel-row = calc.floor(idx / cols)
      let px = ox + panel-col * (panel-w + col-gap)
      let py = oy + (rows - 1 - panel-row) * (panel-h + row-gap)
      let tx(v) = px + (v - radii.first()) / (radii.last() - radii.first()) * panel-w
      let ty(v) = py + (v - y-min) / (y-max - y-min) * panel-h

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.4pt, weight: "bold")[#rule-label(rule)])

      for radius in radii {
        let x = tx(radius)
        line((x, py), (x, py + panel-h), stroke: 0.18pt + black30)
        if panel-row == rows - 1 {
          content((x, py - 0.18), text(fill: black70, size: 6.0pt)[#str(radius)])
        }
      }
      for tick in (y-min, y-min + 0.5 * (y-max - y-min), y-max) {
        let y = ty(tick)
        line((px, y), (px + panel-w, y), stroke: 0.18pt + black30)
        if panel-col == 0 {
          content((px - 0.16, y), text(fill: black70, size: 6.0pt)[#str(calc.round(tick, digits: 3))], anchor: "east")
        }
      }

      for point-idx in range(rows-data.len() - 1) {
        let a = rows-data.at(point-idx)
        let b = rows-data.at(point-idx + 1)
        line((tx(a.at("radius")), ty(a.at("wvf").at("anisotropy_ratio"))), (tx(b.at("radius")), ty(b.at("wvf").at("anisotropy_ratio"))), stroke: 0.78pt + garnet)
        line((tx(a.at("radius")), ty(a.at("square_sg").at("anisotropy_ratio"))), (tx(b.at("radius")), ty(b.at("square_sg").at("anisotropy_ratio"))), stroke: 0.78pt + congaree)
      }
    }

    let lx = ox + total-w - 1.00
    let ly = oy + total-h - 0.05
    line((lx, ly), (lx + 0.18, ly), stroke: 0.78pt + garnet)
    content((lx + 0.26, ly), text(fill: black70, size: 6.2pt)[WVF], anchor: "west")
    line((lx, ly - 0.22), (lx + 0.18, ly - 0.22), stroke: 0.78pt + congaree)
    content((lx + 0.26, ly - 0.22), text(fill: black70, size: 6.2pt)[Square SG], anchor: "west")

    content((ox + total-w / 2, oy - 0.56), text(fill: black90, size: 7.3pt)[WVF radius $r$])
    content((0.18, oy + total-h / 2), angle: 90deg, text(fill: black90, size: 7.3pt)[Anisotropy ratio])
  })
}
