#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.4pt)

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
  let max-v = 0.0
  for method in order {
    for key in ("l_corner", "x_junction") {
      let value = methods.at(method).at("junctions").at(key).at("branch_isotropy_mean")
      if value > max-v { max-v = value }
    }
  }
  let y1 = 1.1 * max-v

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.9
    let oy = 0.82
    let pw = 6.9
    let ph = 4.25
    let group-gap = 0.26
    let bar-w = 0.18
    let within = 0.05
    let ty(v) = oy + v / y1 * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Junction branch isotropy])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[L-corner and X-junction, 4-phase orientation mean])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0) {
      let y = ty(tick * y1)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.14, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick * y1, digits: 2))], anchor: "east")
    }

    for idx in range(order.len()) {
      let method = order.at(idx)
      let xg = ox + 0.28 + idx * (2 * bar-w + within + group-gap)
      let l-val = methods.at(method).at("junctions").at("l_corner").at("branch_isotropy_mean")
      let x-val = methods.at(method).at("junctions").at("x_junction").at("branch_isotropy_mean")
      rect((xg, oy), (xg + bar-w, ty(l-val)), fill: method-color(method), stroke: none)
      rect((xg + bar-w + within, oy), (xg + 2 * bar-w + within, ty(x-val)), fill: black70, stroke: none)
      content((xg + bar-w, oy - 0.24), angle: 45deg, text(fill: black90, size: 6.2pt)[#methods.at(method).at("label")])
    }

    content((ox + pw / 2, oy - 0.74), text(fill: black90, size: 7.3pt)[Method])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.3pt)[Branch isotropy ratio])

    let lx = ox + pw - 1.2
    let ly = oy + ph - 0.12
    rect((lx, ly - 0.10), (lx + 0.14, ly), fill: garnet, stroke: none)
    content((lx + 0.22, ly - 0.05), text(fill: black70, size: 6.2pt)[L-corner], anchor: "west")
    rect((lx, ly - 0.34), (lx + 0.14, ly - 0.24), fill: black70, stroke: none)
    content((lx + 0.22, ly - 0.29), text(fill: black70, size: 6.2pt)[X-junction], anchor: "west")
  })
}
