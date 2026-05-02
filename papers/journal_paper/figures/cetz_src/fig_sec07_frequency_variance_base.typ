#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.8pt)

#let render(data-path) = {
  let data = json(data-path)
  let subtitle = data.at("subtitle")
  let records = data.at("variance_records")
  let groups = (
    ("awgn", "AWGN", garnet),
    ("poisson", "Poisson", atlantic),
    ("speckle", "Speckle", congaree),
  )

  let y-min = 1e30
  let y-max = -1e30
  for rec in records {
    let value = rec.at("empirical_variance")
    if value < y-min { y-min = value }
    if value > y-max { y-max = value }
    if rec.at("analytical_variance") != none and rec.at("analytical_variance") > y-max {
      y-max = rec.at("analytical_variance")
    }
  }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 1e-6 }
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let pw = 8.2
    let ph = 5.1
    let ox = 1.05
    let oy = 0.92

    let positions = (
      0, 1, 2,
      4, 5, 6,
      8, 9, 10,
    )
    let tx(v) = ox + v / 10 * pw
    let ty(v) = oy + (v - y0) / (y1 - y0) * ph

    content((ox + pw / 2, oy + ph + 0.92), text(fill: black90, size: 10pt, weight: "bold")[Section 7.8 noise variance])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8pt)[#subtitle])

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)
    content((ox + pw / 2, oy - 0.64), text(fill: black90, size: 8.4pt)[Noise severity])
    content((ox - 0.90, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.4pt)[Empirical output variance])

    let x-labels = ("30", "20", "10", "1000", "100", "10", "0.1", "0.4", "0.8")
    for idx in range(x-labels.len()) {
      let x = tx(positions.at(idx))
      line((x, oy), (x, oy - 0.08), stroke: 0.4pt + black70)
      content((x, oy - 0.24), text(fill: black70, size: 6.4pt)[#x-labels.at(idx)])
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
      line((ox, y), (ox - 0.08, y), stroke: 0.4pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.76, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick, digits: 6))])
    }

    for group-idx in range(groups.len()) {
      let group = groups.at(group-idx)
      let base-pos = group-idx * 4
      let series = ()
      for rec in records {
        if rec.at("noise_type") == group.at(0) {
          series.push(rec)
        }
      }
      for idx in range(series.len() - 1) {
        let a = series.at(idx)
        let b = series.at(idx + 1)
        line(
          (tx(base-pos + idx), ty(a.at("empirical_variance"))),
          (tx(base-pos + idx + 1), ty(b.at("empirical_variance"))),
          stroke: 1.15pt + group.at(2),
        )
      }
      for idx in range(series.len()) {
        let rec = series.at(idx)
        let x = tx(base-pos + idx)
        circle((x, ty(rec.at("empirical_variance"))), radius: 0.06, fill: group.at(2))
        if rec.at("analytical_variance") != none {
          circle((x, ty(rec.at("analytical_variance"))), radius: 0.045, stroke: 0.55pt + black90)
        }
      }
      content((tx(base-pos + 1), oy + ph + 0.12), text(fill: group.at(2), size: 7.2pt, weight: "bold")[#group.at(1)])
    }

    let lx = ox + pw - 2.25
    let ly = oy + ph - 0.12
    rect((lx - 0.15, ly + 0.16), (lx + 2.05, ly - 0.84), fill: white, stroke: 0.35pt + black30)
    content((lx + 0.92, ly - 0.02), text(fill: black90, size: 7.2pt, weight: "bold")[Legend])
    content((lx, ly - 0.28), text(fill: black70, size: 6.2pt)[Filled circles are empirical variance.], anchor: "west")
    content((lx, ly - 0.50), text(fill: black70, size: 6.2pt)[Open circles are the AWGN analytical reference.], anchor: "west")
  })
}
