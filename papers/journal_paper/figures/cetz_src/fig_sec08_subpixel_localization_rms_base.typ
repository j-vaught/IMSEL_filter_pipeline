#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30, black50

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
  let snr-order = ("inf", "20", "15", "10", "5", "0")
  let y-min = 1e30
  let y-max = -1e30
  for method in order {
    for snr in snr-order {
      let value = methods.at(method).at("rms_by_snr").at(snr)
      if value < y-min { y-min = value }
      if value > y-max { y-max = value }
    }
  }

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.92
    let oy = 0.82
    let pw = 6.6
    let ph = 4.25
    let tx(idx) = ox + idx / (snr-order.len() - 1) * pw
    let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Sub-pixel localisation RMS])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[Validation-tuned methods on smoothed step edges])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for idx in range(snr-order.len()) {
      let x = tx(idx)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.0pt)[#snr-order.at(idx)])
    }
    let y-ticks = (y-min, y-min + 0.5 * (y-max - y-min), y-max)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.0pt)[#str(calc.round(tick, digits: 3))], anchor: "east")
    }

    for method in order {
      for idx in range(snr-order.len() - 1) {
        let a = methods.at(method).at("rms_by_snr").at(snr-order.at(idx))
        let b = methods.at(method).at("rms_by_snr").at(snr-order.at(idx + 1))
        line((tx(idx), ty(a)), (tx(idx + 1), ty(b)), stroke: 0.75pt + method-color(method))
      }
    }

    let lx = ox + pw - 1.25
    let ly = oy + ph - 0.05
    for idx in range(order.len()) {
      let method = order.at(idx)
      let y = ly - 0.20 * idx
      line((lx, y), (lx + 0.16, y), stroke: 0.75pt + method-color(method))
      content((lx + 0.24, y), text(fill: black70, size: 6.0pt)[#methods.at(method).at("label")], anchor: "west")
    }

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[SNR (dB)])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Localisation RMS (px)])
  })
}
