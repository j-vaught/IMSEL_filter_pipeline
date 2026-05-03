#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.4pt)

#let method-color(name) = {
  if name == "wvf" { return garnet }
  if name == "scharr" { return atlantic }
  if name == "square_sg" { return congaree }
  if name == "square_sg_degmatch_n21_d11" { return black50 }
  if name == "square_sg_degmatch_n25_d11" { return black90 }
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
  let radii = data.at("config").at("curvature_radii")
  let y-min = 1e30
  let y-max = -1e30
  for method in order {
    for radius in radii {
      let value = methods.at(method).at("curvature_metrics").at(str(radius)).at("grad_rmse")
      if value < y-min { y-min = value }
      if value > y-max { y-max = value }
    }
  }
  let lx0 = calc.log(radii.first()) / calc.log(10)
  let lx1 = calc.log(radii.last()) / calc.log(10)
  let ly0 = calc.log(y-min) / calc.log(10)
  let ly1 = calc.log(y-max) / calc.log(10)

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.92
    let oy = 0.82
    let pw = 6.4
    let ph = 4.25
    let tx(v) = ox + (calc.log(v) / calc.log(10) - lx0) / (lx1 - lx0) * pw
    let ty(v) = oy + (calc.log(v) / calc.log(10) - ly0) / (ly1 - ly0) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Curvature handling at AWGN 10 dB])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[Curved-arc and S-curve bank])
    content((ox + pw / 2, oy + ph + 0.22), text(fill: black50, size: 6.2pt)[Farid-Simoncelli uses 7-tap support and is approximately matched-scale to WVF $r = 3$, not the validated $r = 50$ point.])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for radius in radii {
      let x = tx(radius)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.0pt)[#str(radius)])
    }
    let y-ticks = (y-min, calc.pow(10, (ly0 + ly1) / 2), y-max)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.0pt)[#str(tick)], anchor: "east")
    }

    for method in order {
      for idx in range(radii.len() - 1) {
        let a-r = radii.at(idx)
        let b-r = radii.at(idx + 1)
        let a-v = methods.at(method).at("curvature_metrics").at(str(a-r)).at("grad_rmse")
        let b-v = methods.at(method).at("curvature_metrics").at(str(b-r)).at("grad_rmse")
        line((tx(a-r), ty(a-v)), (tx(b-r), ty(b-v)), stroke: 0.72pt + method-color(method))
      }
    }

    let lx = ox + pw - 1.28
    let ly = oy + ph - 0.05
    for idx in range(order.len()) {
      let method = order.at(idx)
      let y = ly - 0.20 * idx
      line((lx, y), (lx + 0.16, y), stroke: 0.72pt + method-color(method))
      content((lx + 0.24, y), text(fill: black70, size: 6.0pt)[#methods.at(method).at("label")], anchor: "west")
    }

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[Curvature radius $rho$ (px)])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Gradient RMSE (log scale)])
  })
}
