#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, honeycomb, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")
#let line-style(mode) = {
  if mode == "reflect" {
    return (paint: garnet, thickness: 1.5pt)
  }
  if mode == "zero" {
    return (paint: black90, thickness: 1.3pt, dash: "dashed")
  }
  if mode == "constant_value" {
    return (paint: honeycomb, thickness: 1.3pt, dash: "dotted")
  }
  (paint: atlantic, thickness: 1.3pt)
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let records = data.at("records")

  let filtered = ()
  for rec in records {
    if rec.at("snr_db") == "10 dB" {
      filtered.push(rec)
    }
  }

  let x-labels = ("0", "r", "2r", "4r", "interior")
  let x-values = (0, 1, 2, 3, 4)
  let y-min = 1e30
  let y-max = 0.0
  for rec in filtered {
    let y = rec.at("grad_rmse")
    if y < y-min { y-min = y }
    if y > y-max { y-max = y }
  }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 0.01 }
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 7.4
    let ph = 5.2
    let ox = 1.15
    let oy = 0.95

    let tx(v) = ox + v / 4 * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.46), text(fill: black70, size: 8.2pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.62), text(fill: black90, size: 8.8pt)[Edge offset from border])
    content((ox - 0.98, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.8pt)[Gradient RMSE])

    for idx in range(x-labels.len()) {
      let x = tx(x-values.at(idx))
      line((x, oy), (x, oy - 0.08), stroke: 0.45pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
      content((x, oy - 0.24), text(fill: black70, size: 6.8pt)[#x-labels.at(idx)])
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
      content((ox - 0.74, y), text(fill: black70, size: 6.6pt)[#str(calc.round(tick, digits: 4))])
    }

    let modes = ("reflect", "zero", "constant_value", "edge")
    for mode in modes {
      let style = line-style(mode)
      let series = ()
      for label in ("0", "15", "30", "60", "interior") {
        for rec in filtered {
          if rec.at("padding_mode") == mode and rec.at("edge_offset_px") == label {
            series.push((label, rec.at("grad_rmse")))
          }
        }
      }
      for idx in range(series.len() - 1) {
        let a = series.at(idx)
        let b = series.at(idx + 1)
        line(
          (tx(x-values.at(idx)), ty(a.at(1))),
          (tx(x-values.at(idx + 1)), ty(b.at(1))),
          stroke: style,
        )
      }
      for idx in range(series.len()) {
        let point = series.at(idx)
        circle((tx(x-values.at(idx)), ty(point.at(1))), radius: 0.05, fill: style.paint)
      }
    }

    let lx = ox + pw - 1.88
    let ly = oy + ph - 0.12
    rect((lx - 0.16, ly + 0.14), (lx + 1.78, ly - 1.08), fill: white, stroke: 0.35pt + black30)
    content((lx + 0.76, ly - 0.02), text(fill: black90, size: 7.4pt, weight: "bold")[Padding])
    let modes = ("reflect", "zero", "constant_value", "edge")
    for idx in range(modes.len()) {
      let mode = modes.at(idx)
      let style = line-style(mode)
      let y = ly - 0.24 * (idx + 1)
      line((lx, y), (lx + 0.46, y), stroke: style)
      content((lx + 0.62, y), text(fill: black90, size: 6.7pt)[#{
        if mode == "reflect" { "Reflection" }
        else if mode == "zero" { "Zero" }
        else if mode == "constant_value" { "Border-constant" }
        else { "Clamp" }
      }], anchor: "west")
    }
  })
}
