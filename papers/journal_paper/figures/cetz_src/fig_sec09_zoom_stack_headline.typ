#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black, black90, black70, black50, black30, rose, atlantic, congaree, horseshoe, honeycomb, sandstorm

#set page(width: 12.25in, height: 9.5in, margin: 14pt)
#set text(font: "New Computer Modern", size: 7.0pt)

#let DATA_PATH = "../data/sec09_real_image_zoom_stack/sec09_real_image_zoom_stack_summary.json"

#let method-color(name) = {
  if name == "wvf" { return garnet }
  if name == "dog" { return horseshoe }
  if name == "farid_simoncelli" { return honeycomb }
  if name == "sobel" { return rose }
  black70
}

#let fmt(v, digits: 4) = str(calc.round(v, digits: digits))

#let snr-label(slug) = {
  if slug == "inf" { return "clean" }
  if slug == "10" { return "10 dB" }
  slug
}

#let short-method(name) = {
  if name == "farid_simoncelli" { return "F-S" }
  if name == "dog" { return "DoG" }
  if name == "sobel" { return "Sobel" }
  if name == "wvf" { return "WVF" }
  name
}

#let thumb(path, width: 1.55in) = image("../" + path, width: width)

#let column-header(zoom) = [
  text(fill: black90, size: 6.7pt, weight: "bold")[#zoom.at("label")]
  linebreak()
  text(fill: black70, size: 5.8pt)[#zoom.at("effective_vessel_diameter_px")]
]

#let image-cell(assets) = stack(
  spacing: 2pt,
  box(stroke: 0.28pt + black30, inset: 1pt)[thumb(assets.at("magnitude_path"))],
  box(stroke: 0.28pt + black30, inset: 1pt)[thumb(assets.at("orientation_path"))],
)

#let input-cell(zoom) = stack(
  spacing: 2pt,
  box(stroke: 0.28pt + black30, inset: 1pt)[thumb(zoom.at("input_asset_path"))],
  box(stroke: 0.28pt + black30, inset: 1pt)[thumb(zoom.at("vessel_mask_asset_path"))],
)

#let method-grid(data) = {
  let zoom-order = data.at("zoom_order")
  let zooms = data.at("zooms")
  let method-order = data.at("method_order")
  let cells = ()
  cells.push(box(fill: sandstorm, stroke: 0.35pt + black30, inset: 3pt)[text(fill: black90, size: 6.6pt, weight: "bold")[View]])
  for zoom-key in zoom-order {
    let zoom = zooms.at(zoom-key)
    cells.push(box(fill: sandstorm, stroke: 0.35pt + black30, inset: 3pt)[#column-header(zoom)])
  }
  cells.push(box(fill: sandstorm, stroke: 0.35pt + black30, inset: 3pt)[text(fill: black90, size: 6.8pt, weight: "bold")[Input]])
  for zoom-key in zoom-order {
    cells.push(input-cell(zooms.at(zoom-key)))
  }
  for method in method-order {
    let label = data.at("zooms").at(zoom-order.at(0)).at("methods").at(method).at("label")
    cells.push(
      box(fill: sandstorm, stroke: 0.35pt + black30, inset: 3pt)[
        text(fill: method-color(method), size: 6.8pt, weight: "bold")[#label]
        linebreak()
        text(fill: black70, size: 5.5pt)[mag / ori]
      ],
    )
    for zoom-key in zoom-order {
      let assets = zooms.at(zoom-key).at("methods").at(method).at("clean_assets")
      cells.push(image-cell(assets))
    }
  }
  grid(columns: 1 + zoom-order.len(), column-gutter: 5pt, row-gutter: 5pt, ..cells)
}

#let delta-chart(data, metric-key, title, y-label) = {
  let zoom-order = data.at("zoom_order")
  let zooms = data.at("zooms")
  let values = ()
  for zoom-key in zoom-order {
    let deltas = zooms.at(zoom-key).at("delta_small_stencil_minus_best_wvf")
    values.push(deltas.at("inf").at(metric-key))
    values.push(deltas.at("10").at(metric-key))
  }
  let y-min = calc.min(..values)
  let y-max = calc.max(..values)
  let pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.05 }
  let y0 = y-min - pad
  let y1 = y-max + pad

  cetz.canvas({
    import cetz.draw: *
    let w = 4.0
    let h = 1.45
    let ox = 0.85
    let oy = 0.52
    let span = if zoom-order.len() > 1 { zoom-order.len() - 1 } else { 1 }
    let tx(i) = ox + i / span * w
    let ty(v) = oy + (v - y0) / (y1 - y0) * h
    rect((ox, oy), (ox + w, oy + h), stroke: 0.42pt + black30)
    content((ox + w / 2, oy + h + 0.18), text(fill: black90, size: 7.0pt, weight: "bold")[title])
    for idx in range(zoom-order.len()) {
      let zoom = zooms.at(zoom-order.at(idx))
      let x = tx(idx)
      line((x, oy), (x, oy + h), stroke: 0.18pt + black30)
      content((x, oy - 0.15), text(fill: black70, size: 5.6pt)[x#str(zoom.at("downsample_factor"))], anchor: "north")
    }
    let y-ticks = (y0, 0.0, y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + w, y), stroke: 0.18pt + black30)
      content((ox - 0.12, y), text(fill: black70, size: 5.6pt)[#fmt(tick)], anchor: "east")
    }
    for idx in range(zoom-order.len() - 1) {
      let inf-a = zooms.at(zoom-order.at(idx)).at("delta_small_stencil_minus_best_wvf").at("inf").at(metric-key)
      let inf-b = zooms.at(zoom-order.at(idx + 1)).at("delta_small_stencil_minus_best_wvf").at("inf").at(metric-key)
      line((tx(idx), ty(inf-a)), (tx(idx + 1), ty(inf-b)), stroke: 1.05pt + garnet)
      let ten-a = zooms.at(zoom-order.at(idx)).at("delta_small_stencil_minus_best_wvf").at("10").at(metric-key)
      let ten-b = zooms.at(zoom-order.at(idx + 1)).at("delta_small_stencil_minus_best_wvf").at("10").at(metric-key)
      line((tx(idx), ty(ten-a)), (tx(idx + 1), ty(ten-b)), stroke: 0.95pt + atlantic)
    }
    for idx in range(zoom-order.len()) {
      let inf-v = zooms.at(zoom-order.at(idx)).at("delta_small_stencil_minus_best_wvf").at("inf").at(metric-key)
      let ten-v = zooms.at(zoom-order.at(idx)).at("delta_small_stencil_minus_best_wvf").at("10").at(metric-key)
      circle((tx(idx), ty(inf-v)), radius: 0.036, fill: garnet, stroke: none)
      circle((tx(idx), ty(ten-v)), radius: 0.036, fill: atlantic, stroke: none)
    }
    content((ox + w / 2, oy - 0.4), text(fill: black90, size: 6.2pt)[Downsample factor])
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.2pt)[#y-label])
  })
}

#let delta-legend() = grid(
  columns: 2,
  column-gutter: 10pt,
  [
    box(width: 10pt, height: 10pt, fill: garnet, stroke: 0.2pt + black30)
    h(4pt)
    text(fill: black90, size: 6.1pt)[clean]
  ],
  [
    box(width: 10pt, height: 10pt, fill: atlantic, stroke: 0.2pt + black30)
    h(4pt)
    text(fill: black90, size: 6.1pt)[10 dB]
  ],
)

#let summary-table(data) = {
  let zoom-order = data.at("zoom_order")
  let zooms = data.at("zooms")
  let cells = ()
  cells.push([Zoom])
  cells.push([Best WVF, clean])
  cells.push([Best WVF, 10 dB])
  cells.push([Baseline delta, clean])
  cells.push([Baseline delta, 10 dB])
  for zoom-key in zoom-order {
    let zoom = zooms.at(zoom-key)
    let clean-best = zoom.at("methods").at("wvf").at("best_by_snr").at("inf")
    let ten-best = zoom.at("methods").at("wvf").at("best_by_snr").at("10")
    let clean-base = zoom.at("best_baseline_by_snr").at("inf")
    let ten-base = zoom.at("best_baseline_by_snr").at("10")
    let clean-delta = zoom.at("delta_small_stencil_minus_best_wvf").at("inf")
    let ten-delta = zoom.at("delta_small_stencil_minus_best_wvf").at("10")
    cells.push([text(fill: black90, weight: "bold")[#zoom.at("label")] linebreak() text(fill: black70, size: 5.6pt)[#zoom.at("effective_vessel_diameter_px")]])
    cells.push([
      text(fill: garnet, weight: "bold")[#clean-best.at("label")]
      linebreak()
      text(fill: black90, size: 5.6pt)[RMSE #fmt(clean-best.at("metrics").at("gradient_vector_rmse_mean"))]
      linebreak()
      text(fill: black90, size: 5.6pt)[MAE #fmt(clean-best.at("metrics").at("orientation_mae_deg_mean")) deg]
    ])
    cells.push([
      text(fill: garnet, weight: "bold")[#ten-best.at("label")]
      linebreak()
      text(fill: black90, size: 5.6pt)[RMSE #fmt(ten-best.at("metrics").at("gradient_vector_rmse_mean"))]
      linebreak()
      text(fill: black90, size: 5.6pt)[MAE #fmt(ten-best.at("metrics").at("orientation_mae_deg_mean")) deg]
    ])
    cells.push([
      text(fill: method-color(clean-base.at("method")), weight: "bold")[#clean-base.at("label")]
      linebreak()
      text(fill: black90, size: 5.6pt)[RMSE +#fmt(clean-delta.at("gradient_vector_rmse_mean"))]
      linebreak()
      text(fill: black90, size: 5.6pt)[MAE +#fmt(clean-delta.at("orientation_mae_deg_mean")) deg]
    ])
    cells.push([
      text(fill: method-color(ten-base.at("method")), weight: "bold")[#ten-base.at("label")]
      linebreak()
      text(fill: black90, size: 5.6pt)[RMSE +#fmt(ten-delta.at("gradient_vector_rmse_mean"))]
      linebreak()
      text(fill: black90, size: 5.6pt)[MAE +#fmt(ten-delta.at("orientation_mae_deg_mean")) deg]
    ])
  }
  table(columns: 5, stroke: 0.35pt + black30, inset: 3pt, align: left, ..cells)
}

#let render(data-path) = {
  let data = json(data-path)
  let selected = data.at("selected_image")
  let asset-rendering = data.at("asset_rendering")

  [
    align(center)[
      text(fill: black90, size: 10.8pt, weight: "bold")[#data.at("title")]
      linebreak()
      text(fill: black70, size: 7.7pt)[#data.at("subtitle")]
      linebreak()
      text(fill: black70, size: 6.3pt)[HRF image #selected.at("image_id") from the #selected.at("condition_class").replace("_", " ") class. Each view is center-cropped to 512 by 512 pixels and preview-rendered at #str(asset-rendering.at("asset_max_width_px")) px width.]
    ]
    v(6pt)
    method-grid(data)
    v(8pt)
    grid(
      columns: 2,
      column-gutter: 10pt,
      [
        summary-table(data)
      ],
      [
        stack(
          spacing: 8pt,
          delta-chart(data, "gradient_vector_rmse_mean", [Best-WVF RMSE advantage over best baseline], [Delta RMSE]),
          delta-chart(data, "orientation_mae_deg_mean", [Best-WVF orientation advantage over best baseline], [Delta MAE (deg)]),
          delta-legend(),
        )
      ],
    )
  ]
}

#render(DATA_PATH)
