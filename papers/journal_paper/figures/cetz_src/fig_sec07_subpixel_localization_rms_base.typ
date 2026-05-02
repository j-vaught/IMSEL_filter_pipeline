#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.8pt)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.10 localisation RMS versus SNR"
  let rows = data.at("rms_by_snr")
  let x-labels = ("clean", "30 dB", "20 dB", "10 dB")
  let x-values = (0, 1, 2, 3)
  let y-min = 0.0
  let y-max = 0.0
  for row in rows {
    if row.at("localisation_rms") > y-max { y-max = row.at("localisation_rms") }
  }
  let y1 = if y-max > 0.0 { 1.10 * y-max } else { 0.1 }

  cetz.canvas({
    import cetz.draw: *

    let pw = 6.0
    let ph = 4.2
    let ox = 1.00
    let oy = 0.88
    let tx(v) = ox + v / 3 * pw
    let ty(v) = oy + (v - y-min) / (y1 - y-min) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.2pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    for idx in range(x-labels.len()) {
      let x = tx(x-values.at(idx))
      line((x, oy), (x, oy - 0.08), stroke: 0.4pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.24), text(fill: black70, size: 6.8pt)[#x-labels.at(idx)])
    }
    content((ox + pw / 2, oy - 0.58), text(fill: black90, size: 8.5pt)[SNR])

    let y-ticks = (
      0.0,
      0.25 * y1,
      0.5 * y1,
      0.75 * y1,
      y1,
    )
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.08, y), stroke: 0.4pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.6pt)[#str(calc.round(tick, digits: 4))], anchor: "east")
    }
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.4pt)[Localisation RMS (px)])

    let points = ()
    for row in rows {
      points.push(row.at("localisation_rms"))
    }
    for idx in range(points.len() - 1) {
      line((tx(x-values.at(idx)), ty(points.at(idx))), (tx(x-values.at(idx + 1)), ty(points.at(idx + 1))), stroke: 1.25pt + garnet)
    }
    for idx in range(points.len()) {
      circle((tx(x-values.at(idx)), ty(points.at(idx))), radius: 0.06, fill: garnet)
    }
  })
}
