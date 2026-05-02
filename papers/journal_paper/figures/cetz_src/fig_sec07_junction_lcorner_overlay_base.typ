#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let shape-style(name) = {
  if name == "disk" {
    (paint: garnet, thickness: 1.5pt)
  } else {
    (paint: atlantic, thickness: 1.3pt)
  }
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let shape-order = data.at("shape_order")
  let shapes = data.at("shapes")

  let x-max = 350.0
  let y-max = 0.0
  let y-min = 1e30
  for name in shape-order {
    let curve = shapes.at(name).at("curve")
    for row in curve {
      let value = row.at("junction_magnitude")
      if value > y-max { y-max = value }
      if value < y-min { y-min = value }
    }
  }
  let y-pad = 0.04 * (y-max - y-min)
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 10.1
    let ph = 6.0
    let ox = 1.4
    let oy = 1.0

    let tx(v) = ox + v / x-max * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.4pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.68), text(fill: black90, size: 9pt)[$theta$ (deg)])
    content((ox - 1.22, oy + ph / 2), text(fill: black90, size: 9pt)[Junction-center magnitude], angle: 90deg)

    for tick in (0, 60, 120, 180, 240, 300, 360) {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.4pt)[#tick])
      if tick > 0 and tick < 360 {
        line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
      }
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
      line((ox, y), (ox - 0.1, y), stroke: 0.45pt + black70)
      content((ox - 0.60, y), text(fill: black70, size: 7.1pt)[#str(calc.round(tick, digits: 4))])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    for name in shape-order {
      let curve = shapes.at(name).at("curve")
      let style = shape-style(name)
      for i in range(curve.len() - 1) {
        let a = curve.at(i)
        let b = curve.at(i + 1)
        line(
          (tx(a.at("theta_deg")), ty(a.at("junction_magnitude"))),
          (tx(b.at("theta_deg")), ty(b.at("junction_magnitude"))),
          stroke: style,
        )
      }
    }

    rect((ox + pw - 2.95, oy + ph - 0.18), (ox + pw - 0.18, oy + ph - 0.98), fill: white, stroke: 0.4pt + black30)
    content((ox + pw - 1.56, oy + ph - 0.34), text(fill: black90, size: 8pt, weight: "bold")[Branch isotropy])
    for i in range(shape-order.len()) {
      let name = shape-order.at(i)
      let shape = shapes.at(name)
      let y = oy + ph - 0.62 - 0.22 * i
      content(
        (ox + pw - 1.56, y),
        text(fill: if name == "disk" { garnet } else { atlantic }, size: 7.2pt)[
          #shape.at("label") = #str(calc.round(shape.at("branch_isotropy_ratio_mean"), digits: 4))
        ],
      )
    }

    rect((ox + 0.34, oy + ph - 0.36), (ox + 2.26, oy + ph - 0.88), fill: white, stroke: 0.4pt + black30)
    line((ox + 0.54, oy + ph - 0.56), (ox + 1.10, oy + ph - 0.56), stroke: shape-style("disk"))
    content((ox + 1.28, oy + ph - 0.56), text(fill: black90, size: 7.4pt)[Disk], anchor: "west")
    line((ox + 0.54, oy + ph - 0.78), (ox + 1.10, oy + ph - 0.78), stroke: shape-style("square"))
    content((ox + 1.28, oy + ph - 0.78), text(fill: black90, size: 7.4pt)[Square], anchor: "west")
  })
}
