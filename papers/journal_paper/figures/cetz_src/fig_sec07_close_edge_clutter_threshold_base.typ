#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.9pt)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.9B close-edge clutter resolution threshold"
  let rows = data.at("close_edge_clutter").at("threshold_by_radius")
  let points = ()
  for row in rows {
    if row.at("resolution_threshold_px") != none {
      points.push((row.at("radius"), row.at("resolution_threshold_px")))
    }
  }
  let x0 = calc.log(2) / calc.log(2)
  let x1 = calc.log(128) / calc.log(2)
  let y0 = calc.log(1) / calc.log(2)
  let y1 = calc.log(32) / calc.log(2)

  cetz.canvas({
    import cetz.draw: *

    let pw = 7.0
    let ph = 4.7
    let ox = 1.05
    let oy = 0.88
    let tx(v) = ox + (calc.log(v) / calc.log(2) - x0) / (x1 - x0) * pw
    let ty(v) = oy + (calc.log(v) / calc.log(2) - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.92), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.2pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    let x-ticks = (2, 4, 8, 16, 32, 64, 128)
    for tick in x-ticks {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.08), stroke: 0.4pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.24), text(fill: black70, size: 6.8pt)[#tick])
    }
    content((ox + pw / 2, oy - 0.58), text(fill: black90, size: 8.5pt)[Support radius $r$ (px)])

    let y-ticks = (1, 2, 4, 8, 16, 32)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.08, y), stroke: 0.4pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.18, y), text(fill: black70, size: 6.8pt)[#tick], anchor: "east")
    }
    content((0.22, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.5pt)[Resolution threshold (px)])

    for idx in range(points.len() - 1) {
      let a = points.at(idx)
      let b = points.at(idx + 1)
      line((tx(a.at(0)), ty(a.at(1))), (tx(b.at(0)), ty(b.at(1))), stroke: 1.25pt + garnet)
    }
    for point in points {
      circle((tx(point.at(0)), ty(point.at(1))), radius: 0.06, fill: garnet)
    }

    content((ox + pw - 1.62, oy + ph - 0.18), text(fill: black70, size: 6.4pt)[Log-log axes])
    content((ox + pw - 1.62, oy + ph - 0.42), text(fill: black70, size: 6.4pt)[Threshold = smallest separation])
    content((ox + pw - 1.62, oy + ph - 0.64), text(fill: black70, size: 6.4pt)[with two resolved peaks])
  })
}
