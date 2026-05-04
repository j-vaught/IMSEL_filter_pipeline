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
#let radius-label(radius) = str(radius)
#let degree-label(degree) = str(degree)

#let heat-color(t) = {
  if t <= 0.16 { return garnet }
  if t <= 0.32 { return rose }
  if t <= 0.48 { return honeycomb }
  if t <= 0.64 { return atlantic }
  if t <= 0.80 { return black50 }
  black10
}

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
  let values = ()
  for method in order {
    values.push(methods.at(method).at(metric-key))
  }
  let mapped = ()
  for value in values {
    mapped.push(if log-scale {
      calc.log(calc.max(value, 1e-12), base: 10)
    } else {
      value
    })
  }
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

#let wvf-trace-chart(data, metric-key, title, y-label, log-scale: false) = {
  let methods = data.at("methods")
  let order = data.at("method_order")
  let points = data.at("wvf_trace").at("points")
  let values = ()
  for point in points {
    values.push(point.at(metric-key))
  }
  for method in order {
    if method != "wvf" {
      values.push(methods.at(method).at(metric-key))
    }
  }
  let mapped = ()
  for value in values {
    mapped.push(if log-scale {
      calc.log(calc.max(value, 1e-12), base: 10)
    } else {
      value
    })
  }
  let y-min = calc.min(..mapped)
  let y-max = calc.max(..mapped)
  let pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.1 }
  let y0 = y-min - pad
  let y1 = y-max + pad

  cetz.canvas({
    import cetz.draw: *
    let w = 5.0
    let h = 2.0
    let ox = 0.82
    let oy = 0.58
    let span = if points.len() > 1 { points.len() - 1 } else { 1 }
    let tx(i) = ox + i / span * w
    let ty(v) = oy + (v - y0) / (y1 - y0) * h

    rect((ox, oy), (ox + w, oy + h), stroke: 0.45pt + black30)
    content((ox + w / 2, oy + h + 0.22), text(fill: black90, size: 7.4pt, weight: "bold")[title])

    for idx in range(points.len()) {
      let x = tx(idx)
      line((x, oy), (x, oy + h), stroke: 0.18pt + black30)
      content((x, oy - 0.17), text(fill: black70, size: 5.7pt)[r=#radius-label(points.at(idx).at("radius"))], anchor: "north")
    }
    let y-ticks = (y0, y0 + 0.5 * (y1 - y0), y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + w, y), stroke: 0.18pt + black30)
      let label = if log-scale {
        "1e" + fmt(tick, digits: 2)
      } else { fmt(tick) }
      content((ox - 0.12, y), text(fill: black70, size: 5.7pt)[label], anchor: "east")
    }
    for method in order {
      if method != "wvf" {
        let raw = methods.at(method).at(metric-key)
        let value = if log-scale { calc.log(calc.max(raw, 1e-12), base: 10) } else { raw }
        line((ox, ty(value)), (ox + w, ty(value)), stroke: 0.35pt + method-color(method))
      }
    }
    for idx in range(points.len() - 1) {
      let raw-a = points.at(idx).at(metric-key)
      let raw-b = points.at(idx + 1).at(metric-key)
      let a = if log-scale { calc.log(calc.max(raw-a, 1e-12), base: 10) } else { raw-a }
      let b = if log-scale { calc.log(calc.max(raw-b, 1e-12), base: 10) } else { raw-b }
      line((tx(idx), ty(a)), (tx(idx + 1), ty(b)), stroke: 1.1pt + garnet)
    }
    for idx in range(points.len()) {
      let point = points.at(idx)
      let raw = point.at(metric-key)
      let value = if log-scale { calc.log(calc.max(raw, 1e-12), base: 10) } else { raw }
      let highlight = point.at("comparison").at(metric-key).at("overtakes_best_baseline")
      if highlight {
        circle((tx(idx), ty(value)), radius: 0.062, fill: rgb("#FFFFFF"), stroke: 0.5pt + black90)
      }
      circle((tx(idx), ty(value)), radius: 0.038, fill: garnet, stroke: none)
    }

    content((ox + w / 2, oy - 0.42), text(fill: black90, size: 6.4pt)[WVF radius])
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.4pt)[y-label])
  })
}

#let wvf-grid-heatmap(data, title) = {
  let grid = data.at("wvf_grid")
  let radii = grid.at("grid_radii")
  let degrees = grid.at("grid_degrees")
  let cells = grid.at("cells")
  let primary-metric = grid.at("primary_metric_key")
  let optimum = grid.at("annotated_optimum")
  let values = ()
  for cell in cells {
    values.push(cell.at(primary-metric))
  }
  let v-min = calc.min(..values)
  let v-max = calc.max(..values)
  let span = calc.max(v-max - v-min, 1e-12)
  let cell-map = (:)
  for cell in cells {
    cell-map.insert(str(cell.at("radius")) + "-" + str(cell.at("degree")), cell)
  }

  cetz.canvas({
    import cetz.draw: *
    let w = 5.2
    let h = 3.0
    let ox = 0.9
    let oy = 0.55
    let cols = radii.len()
    let rows = degrees.len()
    let cw = w / cols
    let ch = h / rows

    rect((ox, oy), (ox + w, oy + h), stroke: 0.45pt + black30)
    content((ox + w / 2, oy + h + 0.24), text(fill: black90, size: 8.0pt, weight: "bold")[title])
    for cidx in range(cols) {
      let radius = radii.at(cidx)
      let x0 = ox + cidx * cw
      content((x0 + cw / 2, oy - 0.18), text(fill: black70, size: 5.8pt)[r=#radius-label(radius)], anchor: "north")
    }
    for ridx in range(rows) {
      let degree = degrees.at(ridx)
      let y0 = oy + (rows - 1 - ridx) * ch
      content((ox - 0.12, y0 + ch / 2), text(fill: black70, size: 5.8pt)[d=#degree-label(degree)], anchor: "east")
    }
    for ridx in range(rows) {
      let degree = degrees.at(ridx)
      for cidx in range(cols) {
        let radius = radii.at(cidx)
        let x0 = ox + cidx * cw
        let y0 = oy + (rows - 1 - ridx) * ch
        let key = str(radius) + "-" + str(degree)
        if key in cell-map {
          let cell = cell-map.at(key)
          let raw = cell.at(primary-metric)
          let norm = (raw - v-min) / span
          rect((x0, y0), (x0 + cw, y0 + ch), fill: heat-color(norm), stroke: 0.25pt + black30)
          if radius == optimum.at("radius") and degree == optimum.at("degree") {
            rect((x0 + 0.02, y0 + 0.02), (x0 + cw - 0.02, y0 + ch - 0.02), fill: none, stroke: 0.55pt + black)
          }
        } else {
          rect((x0, y0), (x0 + cw, y0 + ch), fill: sandstorm, stroke: 0.25pt + black30)
          line((x0 + 0.05, y0 + 0.05), (x0 + cw - 0.05, y0 + ch - 0.05), stroke: 0.25pt + black50)
          line((x0 + 0.05, y0 + ch - 0.05), (x0 + cw - 0.05, y0 + 0.05), stroke: 0.25pt + black50)
        }
      }
    }
    content((ox + w / 2, oy - 0.42), text(fill: black90, size: 6.4pt)[WVF radius])
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.4pt)[Polynomial degree])
  })
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

  if "wvf_trace" in data {
    pagebreak()

    [
      align(center)[
        text(fill: black90, size: 11pt, weight: "bold")[title]
        linebreak()
        text(fill: black70, size: 8pt)[WVF radius-trace follow-up on the fixed fluorescence image set.]
        linebreak()
        text(fill: black70, size: 6.8pt)[Garnet trace points sweep WVF across $(r,d)=(3,5),(5,9),(9,11),(15,11),(25,11),(50,11)$. Thin horizontal lines are the fixed baseline methods. Outlined WVF points beat the best fixed baseline on that metric.]
      ]
      v(8pt)
      grid(
        columns: 2,
        column-gutter: 10pt,
        row-gutter: 10pt,
        [
          wvf-trace-chart(data, "white_noise_gain", [Analytical WNG versus WVF radius], [log10 WNG], log-scale: true)
        ],
        [
          wvf-trace-chart(data, "background_gradient_mad_mean", [Background gradient MAD versus WVF radius], [MAD], log-scale: false)
        ],
        [
          wvf-trace-chart(data, "background_gradient_median_mean", [Background gradient median versus WVF radius], [Median], log-scale: false)
        ],
      )
      v(8pt)
      legend(data)
    ]
  }

  if "wvf_grid" in data {
    pagebreak()

    [
      align(center)[
        text(fill: black90, size: 11pt, weight: "bold")[title]
        linebreak()
        text(fill: black70, size: 8pt)[Fine-grained WVF radius and degree sweep on the fixed fluorescence image set.]
        linebreak()
        text(fill: black70, size: 6.8pt)[The heatmap shows native-noise background-gradient MAD across the feasible $(r,d)$ grid. Crossed cells were skipped by the Section 7 conditioning gate. The outlined cell is the annotated optimum.]
      ]
      v(8pt)
      wvf-grid-heatmap(data, [Background gradient MAD across the feasible WVF grid])
      v(10pt)
      align(left)[
        text(fill: black90, size: 7.2pt, weight: "bold")[Annotated optimum. ]
        text(fill: black90, size: 7.2pt)[#data.at("wvf_grid").at("annotated_optimum").at("label")]
        text(fill: black90, size: 7.2pt)[ with MAD #fmt(data.at("wvf_grid").at("annotated_optimum").at("value"), digits: 6).]
        linebreak()
        text(fill: black90, size: 7.2pt, weight: "bold")[Driver assessment. ]
        text(fill: black90, size: 7.2pt)[#data.at("wvf_grid").at("driver_assessment").at("rationale")]
      ]
    ]
  }
}
