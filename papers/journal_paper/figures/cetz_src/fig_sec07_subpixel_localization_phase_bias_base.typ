#import "@preview/cetz:0.3.4"
#import "../../colors.typ": atlantic, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.8pt)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.10 clean per-phase localisation bias"
  let rows = data.at("clean_phase_curve")
  let x0 = rows.at(0).at("phase_px")
  let x1 = rows.at(rows.len() - 1).at("phase_px")
  let y-min = 1e30
  let y-max = -1e30
  for row in rows {
    let value = row.at("mean_localisation_offset")
    if value < y-min { y-min = value }
    if value > y-max { y-max = value }
  }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 0.01 }
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 6.2
    let ph = 4.2
    let ox = 1.00
    let oy = 0.88
    let tx(v) = ox + (v - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.2pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    for row in rows {
      let x = tx(row.at("phase_px"))
      line((x, oy), (x, oy - 0.08), stroke: 0.4pt + black70)
      content((x, oy - 0.24), text(fill: black70, size: 6.4pt)[#str(calc.round(row.at("phase_px"), digits: 3))])
    }
    content((ox + pw / 2, oy - 0.58), text(fill: black90, size: 8.5pt)[Sub-pixel phase (px)])

    let y-ticks = (
      y0,
      y0 + 0.25 * (y1 - y0),
      y0 + 0.5 * (y1 - y0),
      y0 + 0.75 * (y1 - y0),
      y1,
    )
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.08, y), stroke: 0.4pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.6pt)[#str(calc.round(tick, digits: 4))], anchor: "east")
    }
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.4pt)[Mean localisation offset (px)])

    for idx in range(rows.len() - 1) {
      let a = rows.at(idx)
      let b = rows.at(idx + 1)
      line((tx(a.at("phase_px")), ty(a.at("mean_localisation_offset"))), (tx(b.at("phase_px")), ty(b.at("mean_localisation_offset"))), stroke: 1.25pt + atlantic)
    }
    for row in rows {
      circle((tx(row.at("phase_px")), ty(row.at("mean_localisation_offset"))), radius: 0.05, fill: atlantic)
    }
  })
}
