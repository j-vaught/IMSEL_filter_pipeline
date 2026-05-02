#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black, black90, black70, black50, black30, black10

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let white = rgb("#FFFFFF")

#let render(data-path) = {
  let data = json(data-path)
  let radius = data.at("radius")
  let degree = data.at("degree")
  let normalized = data.at("normalize_coords")
  let pass-count = data.at("pass_count")
  let total-count = data.at("total_count")
  let plot = data.at("plot")
  let records = data.at("records")

  let x-ticks = plot.at("x_ticks")
  let y-ticks = plot.at("y_ticks")
  let log-min = plot.at("log10_min")
  let log-max = plot.at("log10_max")

  let mode-label = if normalized {
    [normalize_coords = True]
  } else {
    [normalize_coords = False]
  }

  cetz.canvas({
    import cetz.draw: *

    let pw = 9.2
    let ph = 5.6
    let ox = 1.4
    let oy = 1.0

    let tx(v) = ox + v / 175 * pw
    let ty(v) = oy + (v - log-min) / (log-max - log-min) * ph

    content(
      (ox + pw / 2, oy + ph + 0.9),
      text(fill: black90, size: 10pt, weight: "bold")[
        Steerability residual. Disk WVF.
      ],
    )
    content(
      (ox + pw / 2, oy + ph + 0.45),
      text(fill: black70, size: 8.5pt)[
        $r$ = #radius, $d$ = #degree, #mode-label
      ],
    )

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

    content((ox + pw / 2, oy - 0.65), text(fill: black90, size: 9pt)[$theta$ (deg)])
    content(
      (ox - 1.05, oy + ph / 2),
      text(fill: black90, size: 9pt)[Max kernel residual],
      angle: 90deg,
    )

    for tick in x-ticks {
      let x = tx(tick)
      line((x, oy), (x, oy - 0.1), stroke: 0.45pt + black70)
      content((x, oy - 0.3), text(fill: black70, size: 7.5pt)[#tick])
      if tick > 0 {
        line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
      }
    }

    for tick in y-ticks {
      let y = ty(tick.at("log10_value"))
      line((ox, y), (ox - 0.1, y), stroke: 0.45pt + black70)
      content((ox - 0.52, y), text(fill: black70, size: 7.5pt)[#tick.at("label")])
      line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
    }

    for i in range(records.len() - 1) {
      let first = records.at(i)
      let second = records.at(i + 1)
      line(
        (tx(first.at("theta_deg")), ty(first.at("log10_plot_residual"))),
        (tx(second.at("theta_deg")), ty(second.at("log10_plot_residual"))),
        stroke: 1.2pt + garnet,
      )
    }

    for record in records {
      let cx = tx(record.at("theta_deg"))
      let cy = ty(record.at("log10_plot_residual"))
      rect(
        (cx - 0.055, cy + 0.055),
        (cx + 0.055, cy - 0.055),
        fill: garnet,
        stroke: 0.35pt + black,
      )
    }

    rect(
      (ox + pw - 2.55, oy + ph - 0.15),
      (ox + pw - 0.15, oy + ph - 0.95),
      fill: white,
      stroke: 0.4pt + black30,
    )
    content(
      (ox + pw - 1.35, oy + ph - 0.37),
      text(fill: black90, size: 8pt, weight: "bold")[
        #pass-count / #total-count pass
      ],
    )
    content(
      (ox + pw - 1.35, oy + ph - 0.67),
      text(fill: black70, size: 7.5pt)[criterion uses $10 epsilon_(64)$],
    )
  })
}
