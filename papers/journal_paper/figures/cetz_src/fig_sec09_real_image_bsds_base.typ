#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black, black90, black70, black50, black30, black10, rose, atlantic, congaree, horseshoe, honeycomb, sandstorm

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

#let snr-label(slug) = {
  if slug == "inf" { return "clean" }
  if slug == "20" { return "20 dB" }
  if slug == "10" { return "10 dB" }
  if slug == "5" { return "5 dB" }
  slug
}

#let thumb(path, width: 1.38in) = image("../" + path, width: width)

#let image-grid(data, heading, asset-key, show-input: false) = {
  let crops = data.at("crops")
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = (
    box(width: 1.1in, height: 1.1in, inset: 2pt)[#text(fill: black90, size: 7.0pt, weight: "bold")[Method]],
  )
  for crop in crops {
    cells.push(
      stack(
        spacing: 3pt,
        align(center)[
          #text(fill: black90, size: 6.5pt, weight: "bold")[#crop.at("label")]
          #if show-input {
            #thumb(crop.at("input_asset_path"))
          }
        ],
      ),
    )
  }
  for method in order {
    let method-data = methods.at(method)
    cells.push(
      box(
        width: 1.1in,
        inset: 2pt,
        fill: sandstorm,
        stroke: 0.35pt + black30,
      )[
        #text(fill: method-color(method), size: 6.8pt, weight: "bold")[#method-data.at("label")]
      ],
    )
    for crop in crops {
      let asset = method-data.at("clean_assets").at(crop.at("crop_key")).at(asset-key)
      cells.push(box(stroke: 0.35pt + black30, inset: 1pt)[#thumb(asset)])
    }
  }

  [
    #text(fill: black90, size: 9.2pt, weight: "bold")[#heading]
    #v(4pt)
    #grid(
      columns: 1 + crops.len(),
      column-gutter: 5pt,
      row-gutter: 5pt,
      ..cells,
    )
  ]
}

#let metric-chart(data, metric-key, title, y-label) = {
  let methods = data.at("methods")
  let order = data.at("method_order")
  let x-order = ("inf", "20", "10", "5")
  let y-min = 1e30
  let y-max = -1e30
  for method in order {
    let method-data = methods.at(method)
    for slug in x-order {
      let value = method-data.at("snr_metrics").at(slug).at(metric-key)
      if value < y-min { y-min = value }
      if value > y-max { y-max = value }
    }
  }
  let pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.05 }
  let y0 = y-min - pad
  let y1 = y-max + pad

  cetz.canvas({
    import cetz.draw: *

    let w = 5.1
    let h = 2.25
    let ox = 0.85
    let oy = 0.6
    let tx(i) = ox + i / 3.0 * w
    let ty(v) = oy + (v - y0) / (y1 - y0) * h

    rect((ox, oy), (ox + w, oy + h), stroke: 0.45pt + black30)
    content((ox + w / 2, oy + h + 0.24), text(fill: black90, size: 8pt, weight: "bold")[#title])

    for idx in range(4) {
      let x = tx(idx)
      line((x, oy), (x, oy + h), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.0pt)[#snr-label(x-order.at(idx))], anchor: "north")
    }
    let y-ticks = (y0, y0 + 0.5 * (y1 - y0), y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + w, y), stroke: 0.18pt + black30)
      content((ox - 0.12, y), text(fill: black70, size: 6.0pt)[#fmt(tick)], anchor: "east")
    }

    for method in order {
      let method-data = methods.at(method)
      for idx in range(3) {
        let slug-a = x-order.at(idx)
        let slug-b = x-order.at(idx + 1)
        let a = method-data.at("snr_metrics").at(slug-a).at(metric-key)
        let b = method-data.at("snr_metrics").at(slug-b).at(metric-key)
        line(
          (tx(idx), ty(a)),
          (tx(idx + 1), ty(b)),
          stroke: 0.85pt + method-color(method),
        )
      }
      for idx in range(4) {
        let slug = x-order.at(idx)
        let value = method-data.at("snr_metrics").at(slug).at(metric-key)
        circle((tx(idx), ty(value)), radius: 0.035, fill: method-color(method), stroke: none)
      }
    }

    content((ox + w / 2, oy - 0.48), text(fill: black90, size: 6.8pt)[Input SNR])
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.8pt)[#y-label])
  })
}

#let legend(data) = {
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = ()
  for method in order {
    cells.push([
      #box(width: 10pt, height: 10pt, fill: method-color(method), stroke: 0.2pt + black)
      #h(4pt)
      #text(fill: black90, size: 6.4pt)[#methods.at(method).at("label")]
    ])
  }
  grid(
    columns: 4,
    column-gutter: 8pt,
    row-gutter: 4pt,
    ..cells,
  )
}

#let summary-table(data) = {
  let order = data.at("method_order")
  let methods = data.at("methods")
  let cells = ()
  cells.push([Metric])
  for method in order {
    let method-data = methods.at(method)
    cells.push([#text(fill: method-color(method), weight: "bold")[#method-data.at("label")]])
  }
  cells.push([ODS clean])
  for method in order { cells.push([#fmt(methods.at(method).at("snr_metrics").at("inf").at("ods_f_score"))]) }
  cells.push([ODS 10 dB])
  for method in order { cells.push([#fmt(methods.at(method).at("snr_metrics").at("10").at("ods_f_score"))]) }
  cells.push([RMSE clean])
  for method in order { cells.push([#fmt(methods.at(method).at("snr_metrics").at("inf").at("gradient_vector_rmse_mean"))]) }
  cells.push([RMSE 10 dB])
  for method in order { cells.push([#fmt(methods.at(method).at("snr_metrics").at("10").at("gradient_vector_rmse_mean"))]) }
  table(
    columns: 1 + order.len(),
    stroke: 0.35pt + black30,
    inset: 3pt,
    align: center,
    ..cells,
  )
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let config = data.at("config")

  [
    #align(center)[
      #text(fill: black90, size: 11pt, weight: "bold")[#title]
      #linebreak()
      #text(fill: black70, size: 8pt)[#subtitle]
      #linebreak()
      #text(fill: black70, size: 6.8pt)[Five BSDS500 test crops. Clean qualitative panels. Metrics over SNR levels clean, 20 dB, 10 dB, and 5 dB.]
    ]
    #v(8pt)
    #image-grid(data, [Gradient magnitude, clean real images], "magnitude_path", show-input: true)
  ]

  #pagebreak()

  [
    #align(center)[
      #text(fill: black90, size: 11pt, weight: "bold")[#title]
      #linebreak()
      #text(fill: black70, size: 8pt)[Orientation maps. Hue encodes edge angle modulo $pi$ and brightness follows normalized magnitude.]
    ]
    #v(8pt)
    #image-grid(data, [Gradient orientation, clean real images], "orientation_path", show-input: false)
  ]

  #pagebreak()

  [
    #align(center)[
      #text(fill: black90, size: 11pt, weight: "bold")[#title]
      #linebreak()
      #text(fill: black70, size: 8pt)[Quantitative summary across #str(config.at("crop_count")) crops. Noisy metrics average over #str(config.at("noise_draws")) AWGN draws per crop.]
    ]
    #v(8pt)
    #grid(
      columns: 2,
      column-gutter: 10pt,
      [
        #metric-chart(data, "ods_f_score", [Soft-label ODS F-score versus SNR], [ODS F-score])
      ],
      [
        #metric-chart(data, "gradient_vector_rmse_mean", [Gradient-vector RMSE versus SNR], [Vector RMSE])
      ],
    )
    #v(10pt)
    #legend(data)
    #v(10pt)
    #summary-table(data)
  ]
}
