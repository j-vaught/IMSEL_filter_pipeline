#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, horseshoe, honeycomb, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.7pt)

#let white = rgb("#FFFFFF")
#let palette = (black90, garnet, rose, atlantic, congaree, horseshoe, honeycomb)

#let curve-color(idx, count) = {
  if count <= palette.len() {
    return palette.at(calc.rem(idx, palette.len()))
  }
  let t = idx / calc.max(count - 1, 1)
  color.mix((garnet, t * 100%), (black90, (1.0 - t) * 100%))
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = "Section 7.9B close-edge clutter peak-amplitude retention"
  let rows = data.at("close_edge_clutter").at("rows")
  let thresholds = data.at("close_edge_clutter").at("threshold_by_radius")
  let separations = data.at("config").at("close_edge_separations_px")
  let radii = data.at("config").at("radii")

  let y0 = 0.0
  let y1 = 1.05
  let x0 = calc.log(separations.at(0)) / calc.log(2)
  let x1 = calc.log(separations.at(separations.len() - 1)) / calc.log(2)

  cetz.canvas({
    import cetz.draw: *

    let pw = 7.2
    let ph = 4.9
    let ox = 1.00
    let oy = 0.88
    let tx(v) = ox + (calc.log(v) / calc.log(2) - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.92), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.1pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    for tick in separations {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.08), stroke: 0.4pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.24), text(fill: black70, size: 6.8pt)[#tick])
    }
    content((ox + pw / 2, oy - 0.58), text(fill: black90, size: 8.5pt)[Bar width / edge separation $s$ (px)])

    let y-ticks = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox - 0.08, y), stroke: 0.4pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.6pt)[#str(calc.round(tick, digits: 2))], anchor: "east")
    }
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.4pt)[Retention ratio $rho(s, r)$])

    line((ox, ty(0.9)), (ox + pw, ty(0.9)), stroke: (paint: black50, thickness: 0.75pt, dash: "dashed"))
    content((ox + pw - 0.08, ty(0.9) + 0.12), text(fill: black70, size: 6.4pt)[$rho = 0.9$], anchor: "east")

    for idx in range(radii.len()) {
      let radius = radii.at(idx)
      let color = curve-color(idx, radii.len())
      let points = ()
      for sep in separations {
        for row in rows {
          if row.at("radius") == radius and row.at("separation_px") == sep {
            points.push((sep, row.at("retention_ratio")))
          }
        }
      }
      for pt-idx in range(points.len() - 1) {
        let a = points.at(pt-idx)
        let b = points.at(pt-idx + 1)
        line((tx(a.at(0)), ty(a.at(1))), (tx(b.at(0)), ty(b.at(1))), stroke: 0.95pt + color)
      }
      for point in points {
        circle((tx(point.at(0)), ty(point.at(1))), radius: 0.04, fill: color)
      }
    }

    let threshold-pieces = ()
    for rec in thresholds {
      let shown = if rec.at("resolution_threshold_px") == none { "none" } else { str(rec.at("resolution_threshold_px")) }
      threshold-pieces.push("r=" + str(rec.at("radius")) + "→" + shown)
    }
    content((ox + pw / 2, oy + ph + 0.16), text(fill: black70, size: 6.3pt)[Thresholds at $rho > 0.9$. #threshold-pieces.join(", ")])

    let lx = ox + pw + 0.30
    let ly = oy + ph - 0.06
    let legend-h = 0.18 * radii.len() + 0.36
    rect((lx - 0.12, ly + 0.16), (lx + 1.36, ly - legend-h), fill: white, stroke: 0.35pt + black30)
    content((lx + 0.56, ly - 0.02), text(fill: black90, size: 7.0pt, weight: "bold")[Radius])
    for idx in range(radii.len()) {
      let y = ly - 0.15 * (idx + 1)
      let color = curve-color(idx, radii.len())
      line((lx, y), (lx + 0.24, y), stroke: 0.95pt + color)
      content((lx + 0.34, y), text(fill: black90, size: 6.0pt)[#radii.at(idx)], anchor: "west")
    }
  })
}
