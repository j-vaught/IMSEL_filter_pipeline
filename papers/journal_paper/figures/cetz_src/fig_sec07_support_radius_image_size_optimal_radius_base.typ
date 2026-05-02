#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let series-style(index) = {
  let palette = (
    (paint: garnet, thickness: 1.6pt),
    (paint: atlantic, thickness: 1.4pt),
    (paint: congaree, thickness: 1.4pt),
    (paint: rose, thickness: 1.4pt),
  )
  palette.at(index)
}

#let lx(v) = calc.log(v) / calc.log(2)

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let series = data.at("optimal_radius_plot")

  let x0 = lx(1)
  let x1 = lx(1024)
  let y0 = lx(2)
  let y1 = lx(512)

  cetz.canvas({
    import cetz.draw: *

    let pw = 10.6
    let ph = 6.2
    let ox = 1.35
    let oy = 0.98

    let tx(v) = ox + (v - x0) / (x1 - x0) * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.3pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.68), text(fill: black90, size: 9pt)[Feature scale (px)])
    content((ox - 1.18, oy + ph / 2), text(fill: black90, size: 9pt)[Optimal radius (px)], angle: 90deg)

    let x-ticks = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    for tick in x-ticks {
      let x = tx(lx(tick))
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.0pt)[#tick])
      line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
    }

    let y-ticks = (2, 4, 8, 16, 32, 64, 128, 256, 512)
    for tick in y-ticks {
      let y = ty(lx(tick))
      line((ox, y), (ox - 0.1, y), stroke: 0.45pt + black70)
      content((ox - 0.54, y), text(fill: black70, size: 7.0pt)[#tick])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    for i in range(series.len()) {
      let entry = series.at(i)
      let points = entry.at("points")
      let style = series-style(i)
      for j in range(points.len() - 1) {
        let a = points.at(j)
        let b = points.at(j + 1)
        line(
          (tx(lx(a.at("feature_scale_px"))), ty(lx(a.at("optimal_radius")))),
          (tx(lx(b.at("feature_scale_px"))), ty(lx(b.at("optimal_radius")))),
          stroke: style,
        )
      }
      for point in points {
        circle((tx(lx(point.at("feature_scale_px"))), ty(lx(point.at("optimal_radius")))), radius: 0.045, fill: style.paint)
      }
    }

    let lx0 = ox + 0.35
    let ly0 = oy + ph - 0.34
    rect((lx0 - 0.18, ly0 + 0.22), (lx0 + 2.25, ly0 - 1.00), fill: white, stroke: 0.4pt + black30)
    content((lx0 + 0.95, ly0 + 0.02), text(fill: black90, size: 8pt, weight: "bold")[Image size])
    for i in range(series.len()) {
      let entry = series.at(i)
      let y = ly0 - 0.24 * i
      line((lx0, y), (lx0 + 0.55, y), stroke: series-style(i))
      content((lx0 + 0.74, y), text(fill: black90, size: 7.1pt)[#entry.at("label")], anchor: "west")
    }
  })
}
