#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.5pt)

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
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let methods = data.at("methods")
  let order = data.at("method_order")
  let max-v = 0.0
  for method in order {
    let value = methods.at(method).at("step").at("anisotropy_by_snr").at("10")
    if value > max-v { max-v = value }
  }
  let y1 = 1.1 * max-v

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.9
    let oy = 0.82
    let pw = 6.6
    let ph = 4.25
    let bar-w = 0.46
    let gap = 0.28
    let tx(idx) = ox + 0.35 + idx * (bar-w + gap)
    let ty(v) = oy + v / y1 * ph

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[Step-edge anisotropy at SNR 10 dB])
    content((ox + pw / 2, oy + ph + 0.52), text(fill: black70, size: 8.0pt)[#subtitle])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    let y-ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    for tick in y-ticks {
      let y = ty(tick * y1)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.14, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick * y1, digits: 2))], anchor: "east")
    }

    for idx in range(order.len()) {
      let method = order.at(idx)
      let value = methods.at(method).at("step").at("anisotropy_by_snr").at("10")
      let x0 = tx(idx)
      rect((x0, oy), (x0 + bar-w, ty(value)), fill: method-color(method), stroke: none)
      content((x0 + bar-w / 2, oy - 0.24), angle: 45deg, text(fill: black90, size: 6.4pt)[#methods.at(method).at("label")])
    }

    content((ox + pw / 2, oy - 0.74), text(fill: black90, size: 7.4pt)[Method])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Anisotropy ratio $A$])
  })
}
