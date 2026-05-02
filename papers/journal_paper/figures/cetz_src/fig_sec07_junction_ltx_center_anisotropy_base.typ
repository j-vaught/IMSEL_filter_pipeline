#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let render(data-path) = {
  let data = json(data-path)
  let junction-order = data.at("junction_order")
  let shapes = data.at("shapes")
  let junctions = data.at("junctions")

  let y-max = 0.0
  let y-min = 1e30
  for name in junction-order {
    let cell = junctions.at(name).at("shapes")
    for shape-name in ("disk", "square") {
      let value = cell.at(shape-name).at("junction_center_anisotropy_ratio")
      if value > y-max { y-max = value }
      if value < y-min { y-min = value }
    }
  }
  let y-pad = 0.07 * (y-max - y-min)
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 7.9
    let ph = 5.4
    let ox = 1.25
    let oy = 0.95
    let x-step = pw / (junction-order.len() - 1)

    let tx(i) = ox + i * x-step
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.88), text(fill: black90, size: 10pt, weight: "bold")[Section 7.3 junction-center anisotropy])
    content((ox + pw / 2, oy + ph + 0.42), text(fill: black70, size: 8.4pt)[Disk vs square across L, T, X. Matched bounding rule, $d = 3$, normalize_coords = True])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.64), text(fill: black90, size: 9pt)[Junction type])
    content((ox - 1.10, oy + ph / 2), text(fill: black90, size: 9pt)[Anisotropy ratio], angle: 90deg)

    for i in range(junction-order.len()) {
      let name = junction-order.at(i)
      let label = junctions.at(name).at("label")
      let x = tx(i)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.28), text(fill: black70, size: 7.5pt)[#label])
      if i > 0 and i < junction-order.len() - 1 {
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
      content((ox - 0.55, y), text(fill: black70, size: 7.1pt)[#str(calc.round(tick, digits: 4))])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    let disk-points = ()
    let square-points = ()
    for i in range(junction-order.len()) {
      let name = junction-order.at(i)
      disk-points.push((tx(i), ty(junctions.at(name).at("shapes").at("disk").at("junction_center_anisotropy_ratio"))))
      square-points.push((tx(i), ty(junctions.at(name).at("shapes").at("square").at("junction_center_anisotropy_ratio"))))
    }

    for i in range(disk-points.len() - 1) {
      line(disk-points.at(i), disk-points.at(i + 1), stroke: (paint: garnet, thickness: 1.5pt))
      line(square-points.at(i), square-points.at(i + 1), stroke: (paint: atlantic, thickness: 1.3pt))
    }
    for point in disk-points {
      circle(point, radius: 0.07, fill: garnet, stroke: none)
    }
    for point in square-points {
      rect((point.at(0) - 0.07, point.at(1) - 0.07), (point.at(0) + 0.07, point.at(1) + 0.07), fill: atlantic, stroke: none)
    }

    rect((ox + pw - 2.12, oy + ph - 0.18), (ox + pw - 0.18, oy + ph - 0.82), fill: white, stroke: 0.4pt + black30)
    line((ox + pw - 1.90, oy + ph - 0.38), (ox + pw - 1.35, oy + ph - 0.38), stroke: (paint: garnet, thickness: 1.5pt))
    circle((ox + pw - 1.625, oy + ph - 0.38), radius: 0.07, fill: garnet, stroke: none)
    content((ox + pw - 1.10, oy + ph - 0.38), text(fill: black90, size: 7.5pt)[#shapes.at("disk").at("label")], anchor: "west")
    line((ox + pw - 1.90, oy + ph - 0.62), (ox + pw - 1.35, oy + ph - 0.62), stroke: (paint: atlantic, thickness: 1.3pt))
    rect((ox + pw - 1.695, oy + ph - 0.69), (ox + pw - 1.555, oy + ph - 0.55), fill: atlantic, stroke: none)
    content((ox + pw - 1.10, oy + ph - 0.62), text(fill: black90, size: 7.5pt)[#shapes.at("square").at("label")], anchor: "west")
  })
}
