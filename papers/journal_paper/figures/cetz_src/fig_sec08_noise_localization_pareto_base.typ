#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black50, black30

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
  let traces = data.at("traces")
  let trace-order = ("wvf", "dog", "square_sg")
  let x-min = 1e30
  let x-max = -1e30
  let y-min = 1e30
  let y-max = -1e30
  for method in order {
    let clean = methods.at(method).at("snr_metrics").at("inf")
    let x = clean.at("white_noise_gain")
    let y = clean.at("fwhm")
    if x < x-min { x-min = x }
    if x > x-max { x-max = x }
    if y < y-min { y-min = y }
    if y > y-max { y-max = y }
  }
  for trace-name in trace-order {
    for row in traces.at(trace-name).at("frontier") {
      let x = row.at("white_noise_gain")
      let y = row.at("fwhm")
      if x < x-min { x-min = x }
      if x > x-max { x-max = x }
      if y < y-min { y-min = y }
      if y > y-max { y-max = y }
    }
  }
  let lx0 = calc.log(x-min) / calc.log(10)
  let lx1 = calc.log(x-max) / calc.log(10)
  let y0 = y-min - 0.08 * (y-max - y-min)
  let y1 = y-max + 0.08 * (y-max - y-min)

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.92
    let oy = 0.82
    let pw = 6.2
    let ph = 4.3
    let tx(v) = ox + (calc.log(v) / calc.log(10) - lx0) / (lx1 - lx0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Noise-localisation Pareto])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[Clean FWHM versus white-noise gain. Parametric traces are shown for WVF, DoG, and square SG.])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    let y-ticks = (y0, y0 + 0.5 * (y1 - y0), y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick, digits: 3))], anchor: "east")
    }
    let x-ticks = (x-min, calc.pow(10, (lx0 + lx1) / 2), x-max)
    for tick in x-ticks {
      let x = tx(tick)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.2pt)[#str(tick)])
    }

    for trace-name in trace-order {
      let frontier = traces.at(trace-name).at("frontier")
      for idx in range(frontier.len() - 1) {
        let a = frontier.at(idx)
        let b = frontier.at(idx + 1)
        line(
          (tx(a.at("white_noise_gain")), ty(a.at("fwhm"))),
          (tx(b.at("white_noise_gain")), ty(b.at("fwhm"))),
          stroke: 0.8pt + method-color(trace-name),
        )
      }
      for row in frontier {
        circle((tx(row.at("white_noise_gain")), ty(row.at("fwhm"))), radius: 0.06, fill: method-color(trace-name), stroke: none)
      }
    }

    for method in order {
      let clean = methods.at(method).at("snr_metrics").at("inf")
      let x = tx(clean.at("white_noise_gain"))
      let y = ty(clean.at("fwhm"))
      circle((x, y), radius: 0.08, fill: method-color(method), stroke: 0.3pt + black90)
      content((x + 0.10, y + 0.08), text(fill: black70, size: 6.0pt)[#methods.at(method).at("label")], anchor: "west")
    }

    let lx = ox + pw - 1.55
    let ly = oy + ph - 0.04
    for idx in range(trace-order.len()) {
      let trace-name = trace-order.at(idx)
      let y = ly - 0.20 * idx
      line((lx, y), (lx + 0.18, y), stroke: 0.8pt + method-color(trace-name))
      content((lx + 0.26, y), text(fill: black70, size: 6.0pt)[#trace-name trace], anchor: "west")
    }

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[White-noise gain (log scale)])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[FWHM])
  })
}
