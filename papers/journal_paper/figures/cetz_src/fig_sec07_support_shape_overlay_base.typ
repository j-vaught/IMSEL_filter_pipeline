#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let render(data-path) = {
  let data = json(data-path)
  let disk = data.at("disk")
  let square = data.at("square")
  let disk-curve = disk.at("curve")
  let square-curve = square.at("curve")

  let x-max = 179.5
  let y-max = 0.0
  let y-min = 1e30
  for row in disk-curve {
    let value = row.at("response_magnitude")
    if value > y-max { y-max = value }
    if value < y-min { y-min = value }
  }
  for row in square-curve {
    let value = row.at("response_magnitude")
    if value > y-max { y-max = value }
    if value < y-min { y-min = value }
  }
  let y-pad = 0.04 * (y-max - y-min)
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 9.5
    let ph = 5.8
    let ox = 1.4
    let oy = 1.0

    let tx(v) = ox + v / x-max * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content(
      (ox + pw / 2, oy + ph + 0.9),
      text(fill: black90, size: 10pt, weight: "bold")[
        Section 7.3 support-shape first pass
      ],
    )
    content(
      (ox + pw / 2, oy + ph + 0.45),
      text(fill: black70, size: 8.5pt)[
        Disk $r = 15$ vs. square $h = 15$, $d = 3$, normalize_coords = True
      ],
    )

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.65), text(fill: black90, size: 9pt)[$theta$ (deg)])
    content(
      (ox - 1.15, oy + ph / 2),
      text(fill: black90, size: 9pt)[Peak directional response],
      angle: 90deg,
    )

    for tick in (0, 30, 60, 90, 120, 150, 180) {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.5pt)[#tick])
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
      content((ox - 0.56, y), text(fill: black70, size: 7.2pt)[#str(calc.round(tick, digits: 4))])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    for i in range(disk-curve.len() - 1) {
      let a = disk-curve.at(i)
      let b = disk-curve.at(i + 1)
      line(
        (tx(a.at("theta_deg")), ty(a.at("response_magnitude"))),
        (tx(b.at("theta_deg")), ty(b.at("response_magnitude"))),
        stroke: 1.4pt + garnet,
      )
    }
    for i in range(square-curve.len() - 1) {
      let a = square-curve.at(i)
      let b = square-curve.at(i + 1)
      line(
        (tx(a.at("theta_deg")), ty(a.at("response_magnitude"))),
        (tx(b.at("theta_deg")), ty(b.at("response_magnitude"))),
        stroke: (paint: atlantic, thickness: 1.4pt, dash: "dashed"),
      )
    }

    rect(
      (ox + pw - 3.25, oy + ph - 0.18),
      (ox + pw - 0.18, oy + ph - 1.26),
      fill: white,
      stroke: 0.4pt + black30,
    )
    content(
      (ox + pw - 1.72, oy + ph - 0.34),
      text(fill: black90, size: 8pt, weight: "bold")[
        Anisotropy ratio
      ],
    )
    content(
      (ox + pw - 1.72, oy + ph - 0.65),
      text(fill: garnet, size: 7.7pt)[Disk = #str(calc.round(disk.at("anisotropy_ratio"), digits: 4))],
    )
    content(
      (ox + pw - 1.72, oy + ph - 0.96),
      text(fill: atlantic, size: 7.7pt)[Square = #str(calc.round(square.at("anisotropy_ratio"), digits: 4))],
    )

    let lx = ox + 0.5
    let ly = oy + ph - 0.4
    rect((lx - 0.18, ly + 0.22), (lx + 2.1, ly - 0.62), fill: white, stroke: 0.4pt + black30)
    line((lx, ly), (lx + 0.55, ly), stroke: 1.4pt + garnet)
    content((lx + 0.72, ly), text(fill: black90, size: 7.7pt)[Disk], anchor: "west")
    line((lx, ly - 0.32), (lx + 0.55, ly - 0.32), stroke: (paint: atlantic, thickness: 1.4pt, dash: "dashed"))
    content((lx + 0.72, ly - 0.32), text(fill: black90, size: 7.7pt)[Square], anchor: "west")
  })
}
