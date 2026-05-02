#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")
#let point-palette = (black90, garnet, rose, atlantic, congaree)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let records = data.at("records")

  let x-min = 1e30
  let x-max = 0.0
  let y-min = 1e30
  let y-max = 0.0
  for rec in records {
    let x = rec.at("fwhm")
    let y = rec.at("white_noise_gain")
    if x < x-min { x-min = x }
    if x > x-max { x-max = x }
    if y < y-min { y-min = y }
    if y > y-max { y-max = y }
  }

  let x-pad = if x-max > x-min { 0.08 * (x-max - x-min) } else { 0.5 }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 1e-6 }
  let x0 = x-min - x-pad
  let x1 = x-max + x-pad
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 7.4
    let ph = 5.2
    let ox = 1.25
    let oy = 0.95

    let tx(v) = ox + (v - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.92), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.2pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.64), text(fill: black90, size: 8.8pt)[FWHM (px)])
    content((ox - 1.08, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.8pt)[White-noise gain])

    let x-ticks = (
      calc.round(x0, digits: 2),
      calc.round(x0 + 0.25 * (x1 - x0), digits: 2),
      calc.round(x0 + 0.50 * (x1 - x0), digits: 2),
      calc.round(x0 + 0.75 * (x1 - x0), digits: 2),
      calc.round(x1, digits: 2),
    )
    for tick in x-ticks {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.08), stroke: 0.45pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
      content((x, oy - 0.26), text(fill: black70, size: 6.8pt)[#str(tick)])
    }

    let y-ticks = (
      y0,
      y0 + 0.25 * (y1 - y0),
      y0 + 0.50 * (y1 - y0),
      y0 + 0.75 * (y1 - y0),
      y1,
    )
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.08, y), stroke: 0.45pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
      content((ox - 0.66, y), text(fill: black70, size: 6.6pt)[#str(calc.round(tick, digits: 6))])
    }

    for idx in range(records.len() - 1) {
      let a = records.at(idx)
      let b = records.at(idx + 1)
      line(
        (tx(a.at("fwhm")), ty(a.at("white_noise_gain"))),
        (tx(b.at("fwhm")), ty(b.at("white_noise_gain"))),
        stroke: 1.15pt + garnet,
      )
    }

    for idx in range(records.len()) {
      let rec = records.at(idx)
      let x = tx(rec.at("fwhm"))
      let y = ty(rec.at("white_noise_gain"))
      let color = point-palette.at(idx)
      circle((x, y), radius: 0.06, fill: color, stroke: 0.35pt + white)
      content((x + 0.12, y + 0.08), text(fill: color, size: 6.8pt)[#rec.at("sigma_label")], anchor: "west")
    }

    let lx = ox + pw - 2.45
    let ly = oy + ph - 0.16
    rect((lx - 0.15, ly + 0.16), (lx + 2.22, ly - 1.16), fill: white, stroke: 0.35pt + black30)
    content((lx + 0.92, ly - 0.02), text(fill: black90, size: 7.6pt, weight: "bold")[Metrics])
    content((lx, ly - 0.28), text(fill: black70, size: 6.5pt)[Points ordered from uniform to narrow weighting.], anchor: "west")
    content((lx, ly - 0.50), text(fill: black70, size: 6.5pt)[Left/down is tighter and quieter.], anchor: "west")
    content((lx, ly - 0.72), text(fill: black70, size: 6.5pt)[Right/up is broader and noisier.], anchor: "west")
  })
}
