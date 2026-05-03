#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black, black90, black70, black50, black30, rose, atlantic, congaree, horseshoe, honeycomb, sandstorm

#set page(width: 12.25in, height: 9.5in, margin: 14pt)
#set text(font: "New Computer Modern", size: 7.4pt)

#let method-color(name) = {
  if name == "wvf" { return garnet }
  if name == "dog" { return horseshoe }
  if name == "square_sg" { return congaree }
  if name == "farid_simoncelli" { return honeycomb }
  if name == "scharr" { return atlantic }
  if name == "sobel" { return rose }
  if name == "prewitt" { return black70 }
  if name == "roberts" { return black50 }
  black90
}

#let fmt(v, digits: 4) = str(calc.round(v, digits: digits))
#let thumb(path, width: 1.38in) = image("../" + path, width: width)

#let image-grid(data, heading, asset-key, show-input: false) = {
  let images = data.at("images")
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = (
    box(width: 1.1in, height: 1.1in, inset: 2pt)[text(fill: black90, size: 7pt, weight: "bold")[Method]],
  )
  for row in images {
    cells.push(
      stack(
        spacing: 3pt,
        align(center)[
          text(fill: black90, size: 6.5pt, weight: "bold")[row.at("image_name")]
          if show-input { thumb(row.at("input_asset_path")) }
        ],
      ),
    )
  }
  for method in order {
    let method-data = methods.at(method)
    cells.push(
      box(width: 1.1in, inset: 2pt, fill: sandstorm, stroke: 0.35pt + black30)[
        text(fill: method-color(method), size: 6.8pt, weight: "bold")[method-data.at("label")]
      ],
    )
    for row in images {
      let asset = method-data.at("clean_assets").at(row.at("image_key")).at(asset-key)
      cells.push(box(stroke: 0.35pt + black30, inset: 1pt)[thumb(asset)])
    }
  }
  [
    text(fill: black90, size: 9.2pt, weight: "bold")[heading]
    v(4pt)
    grid(columns: 1 + images.len(), column-gutter: 5pt, row-gutter: 5pt, ..cells)
  ]
}

#let bar-chart(data, metric-key, title, y-label, log-scale: false) = {
  let methods = data.at("methods")
  let order = data.at("method_order")
  let values = (
    ..for method in order (
      methods.at(method).at(metric-key)
    )
  )
  let mapped = if log-scale {
    (
      ..for value in values (
        calc.log(calc.max(value, 1e-12), base: 10)
      )
    )
  } else { values }
  let y-min = 0.0
  let y-max = calc.max(..mapped)
  let pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.1 }
  let y1 = y-max + pad

  cetz.canvas({
    import cetz.draw: *
    let w = 5.0
    let h = 2.3
    let ox = 0.8
    let oy = 0.55
    let bar-w = 0.42
    let gap = 0.18
    let stride = (w - gap) / order.len()
    let ty(v) = oy + v / y1 * h
    rect((ox, oy), (ox + w, oy + h), stroke: 0.45pt + black30)
    content((ox + w / 2, oy + h + 0.22), text(fill: black90, size: 7.8pt, weight: "bold")[title])
    let y-ticks = (0.0, 0.5 * y1, y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + w, y), stroke: 0.18pt + black30)
      let label = if log-scale {
        "1e" + fmt(tick, digits: 2)
      } else { fmt(tick) }
      content((ox - 0.12, y), text(fill: black70, size: 6pt)[label], anchor: "east")
    }
    for idx in range(order.len()) {
      let method = order.at(idx)
      let x0 = ox + gap / 2 + idx * stride
      let value = mapped.at(idx)
      rect((x0, oy), (x0 + bar-w, ty(value)), fill: method-color(method), stroke: none)
      content((x0 + bar-w / 2, oy - 0.15), angle: 60deg, text(fill: black70, size: 5.6pt)[methods.at(method).at("label")], anchor: "north")
    }
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.7pt)[y-label])
  })
}

#let summary-table(data) = {
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = ()
  cells.push([Metric])
  for method in order { cells.push([text(fill: method-color(method), weight: "bold")[methods.at(method).at("label")]]) }
  cells.push([White-noise gain])
  for method in order { cells.push([fmt(methods.at(method).at("white_noise_gain"))]) }
  cells.push([Background grad. MAD])
  for method in order { cells.push([fmt(methods.at(method).at("background_gradient_mad_mean"))]) }
  cells.push([Background grad. median])
  for method in order { cells.push([fmt(methods.at(method).at("background_gradient_median_mean"))]) }
  table(columns: 1 + order.len(), stroke: 0.35pt + black30, inset: 3pt, align: center, ..cells)
}

#let legend(data) = {
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = ()
  for method in order {
    cells.push([
      box(width: 10pt, height: 10pt, fill: method-color(method), stroke: 0.2pt + black)
      h(4pt)
      text(fill: black90, size: 6.4pt)[methods.at(method).at("label")]
    ])
  }
  grid(columns: 4, column-gutter: 8pt, row-gutter: 4pt, ..cells)
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let config = data.at("config")

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[subtitle]
      linebreak()
      text(fill: black70, size: 6.8pt)[Five native noisy fluorescence fields. Quantitative summary combines analytical white-noise gain with a dark-background gradient-stability proxy.]
    ]
    v(8pt)
    image-grid(data, [Gradient magnitude, native fluorescence images], "magnitude_path", show-input: true)
  ]

  pagebreak()

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[Orientation maps. Hue encodes edge angle modulo $pi$ and brightness follows normalized magnitude.]
    ]
    v(8pt)
    image-grid(data, [Gradient orientation, native fluorescence images], "orientation_path", show-input: false)
  ]

  pagebreak()

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[Analytical and empirical stability summary. The empirical proxy is the median absolute deviation of gradient magnitude on the darkest percentile pixels.]
    ]
    v(8pt)
    grid(
      columns: 2,
      column-gutter: 10pt,
      [
        bar-chart(data, "white_noise_gain", [Analytical white-noise gain], [log10 WNG], log-scale: true)
      ],
      [
        bar-chart(data, "background_gradient_mad_mean", [Background gradient MAD], [MAD], log-scale: false)
      ],
    )
    v(10pt)
    legend(data)
    v(10pt)
    summary-table(data)
  ]
}
