#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, congaree, horseshoe, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.0pt)

#let rule-label(key) = {
  if key == "bounding_radius" { return "Bounding radius" }
  if key == "effective_response_width" { return "Effective response width" }
  if key == "support_cardinality" { return "Support cardinality" }
  if key == "white_noise_gain" { return "White-noise gain" }
  "Effective second moment"
}

#let comparator-color(name) = {
  if name == "square_sg" { return congaree }
  if name == "dog" { return horseshoe }
  black70
}

#let comparator-label(name) = {
  if name == "square_sg" { return "Square SG" }
  if name == "dog" { return "DoG" }
  name
}

#let render(data-path) = {
  let data = json(data-path)
  let comparators = data.at("comparators")
  let comparator-order = data.at("comparator_order")
  let rule-order = data.at("rule_order")
  let radii = data.at("config").at("radius_schedule")
  let cols = rule-order.len()
  let rows = comparator-order.len()
  let y-min = 1e30
  let y-max = -1e30
  for comparator in comparator-order {
    for rule in rule-order {
      for row in comparators.at(comparator).at("rules").at(rule).at("rows") {
        let a = row.at("wvf").at("anisotropy_ratio")
        let b = row.at("comparator").at("anisotropy_ratio")
        if a < y-min { y-min = a }
        if a > y-max { y-max = a }
        if b < y-min { y-min = b }
        if b > y-max { y-max = b }
      }
    }
  }

  cetz.canvas({
    import cetz.draw: *

    let panel-w = 2.28
    let panel-h = 1.88
    let col-gap = 0.42
    let row-gap = 0.74
    let ox = 1.10
    let oy = 0.86
    let total-w = cols * panel-w + (cols - 1) * col-gap
    let total-h = rows * panel-h + (rows - 1) * row-gap

    content((ox + total-w / 2, oy + total-h + 0.98), text(fill: black90, size: 10pt, weight: "bold")[Matched-scale diagnostics])
    content((ox + total-w / 2, oy + total-h + 0.56), text(fill: black70, size: 7.8pt)[WVF versus square SG and DoG at AWGN 10 dB, one panel per matching rule])

    for row-idx in range(comparator-order.len()) {
      let comparator = comparator-order.at(row-idx)
      let py = oy + (rows - 1 - row-idx) * (panel-h + row-gap)
      content((ox - 0.38, py + panel-h / 2), angle: 90deg, text(fill: black90, size: 7.0pt, weight: "bold")[#comparator-label(comparator)], anchor: "south")
      for col-idx in range(rule-order.len()) {
        let rule = rule-order.at(col-idx)
        let rows-data = comparators.at(comparator).at("rules").at(rule).at("rows")
        let px = ox + col-idx * (panel-w + col-gap)
        let tx(v) = px + (v - radii.first()) / (radii.last() - radii.first()) * panel-w
        let ty(v) = py + (v - y-min) / (y-max - y-min) * panel-h

        rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
        if row-idx == 0 {
          content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.0pt, weight: "bold")[#rule-label(rule)])
        }

        for radius in radii {
          let x = tx(radius)
          line((x, py), (x, py + panel-h), stroke: 0.16pt + black30)
          if row-idx == rows - 1 {
            content((x, py - 0.18), text(fill: black70, size: 5.8pt)[#str(radius)])
          }
        }
        for tick in (y-min, y-min + 0.5 * (y-max - y-min), y-max) {
          let y = ty(tick)
          line((px, y), (px + panel-w, y), stroke: 0.16pt + black30)
          if col-idx == 0 {
            content((px - 0.14, y), text(fill: black70, size: 5.8pt)[#str(calc.round(tick, digits: 3))], anchor: "east")
          }
        }

        for point-idx in range(rows-data.len() - 1) {
          let a = rows-data.at(point-idx)
          let b = rows-data.at(point-idx + 1)
          line((tx(a.at("radius")), ty(a.at("wvf").at("anisotropy_ratio"))), (tx(b.at("radius")), ty(b.at("wvf").at("anisotropy_ratio"))), stroke: 0.76pt + garnet)
          line((tx(a.at("radius")), ty(a.at("comparator").at("anisotropy_ratio"))), (tx(b.at("radius")), ty(b.at("comparator").at("anisotropy_ratio"))), stroke: 0.76pt + comparator-color(comparator))
        }
      }
    }

    let lx = ox + total-w - 1.10
    let ly = oy + total-h + 0.02
    line((lx, ly), (lx + 0.18, ly), stroke: 0.76pt + garnet)
    content((lx + 0.26, ly), text(fill: black70, size: 6.0pt)[WVF], anchor: "west")
    line((lx, ly - 0.20), (lx + 0.18, ly - 0.20), stroke: 0.76pt + congaree)
    content((lx + 0.26, ly - 0.20), text(fill: black70, size: 6.0pt)[Square SG], anchor: "west")
    line((lx, ly - 0.40), (lx + 0.18, ly - 0.40), stroke: 0.76pt + horseshoe)
    content((lx + 0.26, ly - 0.40), text(fill: black70, size: 6.0pt)[DoG], anchor: "west")

    content((ox + total-w / 2, oy - 0.54), text(fill: black90, size: 7.2pt)[WVF radius $r$])
    content((0.20, oy + total-h / 2), angle: 90deg, text(fill: black90, size: 7.2pt)[Anisotropy ratio])
  })
}
