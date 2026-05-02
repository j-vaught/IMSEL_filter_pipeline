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
  let snr-order = ("inf", "30", "25", "20", "15", "12", "10", "7p5", "5", "2p5", "1", "0p5", "0")
  let y-min = 1e30
  let y-max = -1e30
  for method in order {
    for snr in snr-order {
      let value = methods.at(method).at("snr_metrics").at(snr).at("grad_rmse")
      if value < y-min { y-min = value }
      if value > y-max { y-max = value }
    }
  }
  let ly0 = calc.log(y-min) / calc.log(10)
  let ly1 = calc.log(y-max) / calc.log(10)

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.92
    let oy = 0.82
    let pw = 6.6
    let ph = 4.25
    let tx(idx) = ox + idx / (snr-order.len() - 1) * pw
    let ty(v) = oy + (calc.log(v) / calc.log(10) - ly0) / (ly1 - ly0) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Gradient RMSE versus SNR])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[Validation-tuned methods on smoothed step edges])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for idx in range(snr-order.len()) {
      let x = tx(idx)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.0pt)[#snr-order.at(idx)])
    }
    let y-ticks = (y-min, calc.pow(10, (ly0 + ly1) / 2), y-max)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.0pt)[#str(tick)], anchor: "east")
    }

    for method in order {
      for idx in range(snr-order.len() - 1) {
        let a = methods.at(method).at("snr_metrics").at(snr-order.at(idx)).at("grad_rmse")
        let b = methods.at(method).at("snr_metrics").at(snr-order.at(idx + 1)).at("grad_rmse")
        line((tx(idx), ty(a)), (tx(idx + 1), ty(b)), stroke: 0.72pt + method-color(method))
      }
    }

    let lx = ox + pw - 1.25
    let ly = oy + ph - 0.05
    for idx in range(order.len()) {
      let method = order.at(idx)
      let y = ly - 0.20 * idx
      line((lx, y), (lx + 0.16, y), stroke: 0.72pt + method-color(method))
      content((lx + 0.24, y), text(fill: black70, size: 6.0pt)[#methods.at(method).at("label")], anchor: "west")
    }

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[SNR / severity])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Gradient RMSE (log scale)])
  })
}
