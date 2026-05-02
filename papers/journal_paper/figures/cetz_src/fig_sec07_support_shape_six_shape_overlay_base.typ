#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, horseshoe, honeycomb, rose, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let shape-style(name) = {
  if name == "triangle" {
    (paint: black90, thickness: 1.3pt, dash: "dashed")
  } else if name == "square" {
    (paint: atlantic, thickness: 1.3pt)
  } else if name == "diamond" {
    (paint: congaree, thickness: 1.3pt, dash: "dotted")
  } else if name == "hexagon" {
    (paint: horseshoe, thickness: 1.3pt)
  } else if name == "octagon" {
    (paint: honeycomb, thickness: 1.3pt, dash: "dash-dotted")
  } else {
    (paint: garnet, thickness: 1.5pt)
  }
}

#let label-style(name) = {
  if name == "triangle" {
    black90
  } else if name == "square" {
    atlantic
  } else if name == "diamond" {
    congaree
  } else if name == "hexagon" {
    horseshoe
  } else if name == "octagon" {
    honeycomb
  } else {
    garnet
  }
}

#let render(data-path) = {
  let data = json(data-path)
  let shape-order = data.at("shape_order")
  let shapes = data.at("shapes")

  let x-max = 179.5
  let y-max = 0.0
  let y-min = 1e30
  for name in shape-order {
    let curve = shapes.at(name).at("curve")
    for row in curve {
      let value = row.at("response_magnitude")
      if value > y-max { y-max = value }
      if value < y-min { y-min = value }
    }
  }
  let y-pad = 0.04 * (y-max - y-min)
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 10.2
    let ph = 6.0
    let ox = 1.45
    let oy = 1.0

    let tx(v) = ox + v / x-max * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content(
      (ox + pw / 2, oy + ph + 0.95),
      text(fill: black90, size: 10pt, weight: "bold")[
        Section 7.3 support-shape six-shape sweep
      ],
    )
    content(
      (ox + pw / 2, oy + ph + 0.48),
      text(fill: black70, size: 8.4pt)[
        Matched bounding rule, $d = 3$, normalize_coords = True, clean smoothed step edge
      ],
    )

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.68), text(fill: black90, size: 9pt)[$theta$ (deg)])
    content(
      (ox - 1.18, oy + ph / 2),
      text(fill: black90, size: 9pt)[Peak directional response],
      angle: 90deg,
    )

    for tick in (0, 30, 60, 90, 120, 150, 180) {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.4pt)[#tick])
      if tick > 0 and tick < 180 {
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
          (tx(a.at("theta_deg")), ty(a.at("response_magnitude"))),
          (tx(b.at("theta_deg")), ty(b.at("response_magnitude"))),
          stroke: style,
        )
      }
    }

    let bx0 = ox + pw - 3.35
    let by0 = oy + ph - 0.18
    rect(
      (bx0, by0),
      (ox + pw - 0.18, oy + ph - 2.02),
      fill: white,
      stroke: 0.4pt + black30,
    )
    content(
      ((bx0 + ox + pw - 0.18) / 2, oy + ph - 0.34),
      text(fill: black90, size: 8pt, weight: "bold")[
        Anisotropy ratio
      ],
    )
    for i in range(shape-order.len()) {
      let name = shape-order.at(i)
      let label = shapes.at(name).at("label")
      let ratio = shapes.at(name).at("anisotropy_ratio")
      content(
        ((bx0 + ox + pw - 0.18) / 2, oy + ph - 0.66 - 0.22 * i),
        text(fill: label-style(name), size: 7.1pt)[#label = #str(calc.round(ratio, digits: 4))],
      )
    }

    let lx = ox + 0.45
    let ly = oy + ph - 0.36
    rect((lx - 0.18, ly + 0.22), (lx + 2.55, ly - 1.55), fill: white, stroke: 0.4pt + black30)
    for i in range(shape-order.len()) {
      let name = shape-order.at(i)
      let label = shapes.at(name).at("label")
      let y = ly - 0.26 * i
      line((lx, y), (lx + 0.58, y), stroke: shape-style(name))
      content((lx + 0.76, y), text(fill: black90, size: 7.3pt)[#label], anchor: "west")
    }
  })
}
