#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black30, atlantic
#import "fig_sec09_real_image_drive_base.typ": method-color, fmt, image-grid, metric-chart, legend, summary-table, wvf-trace-chart

#set page(width: 12.25in, height: 9.5in, margin: 14pt)
#set text(font: "New Computer Modern", size: 7.4pt)

#let snr-label(slug) = {
  if slug == "inf" { return "clean" }
  if slug == "20" { return "20 dB" }
  if slug == "10" { return "10 dB" }
  if slug == "5" { return "5 dB" }
  slug
}

#let delta-chart(data, metric-key, title, y-label) = {
  let drive = data.at("comparison_to_drive").at("drive_deltas").at(metric-key).at("per_snr")
  let hrf = data.at("comparison_to_drive").at("hrf_deltas").at(metric-key).at("per_snr")
  let slugs = ("inf", "20", "10", "5")
  let values = ()
  for slug in slugs {
    values.push(drive.at(slug).at("delta_small_minus_wvf"))
    values.push(hrf.at(slug).at("delta_small_minus_wvf"))
  }
  let y-min = calc.min(..values)
  let y-max = calc.max(..values)
  let pad = if y-max > y-min { 0.08 * (y-max - y-min) } else { 0.05 }
  let y0 = y-min - pad
  let y1 = y-max + pad

  cetz.canvas({
    import cetz.draw: *
    let w = 4.9
    let h = 2.0
    let ox = 0.85
    let oy = 0.58
    let tx(i) = ox + i / 3.0 * w
    let ty(v) = oy + (v - y0) / (y1 - y0) * h
    rect((ox, oy), (ox + w, oy + h), stroke: 0.45pt + black30)
    content((ox + w / 2, oy + h + 0.22), text(fill: black90, size: 7.6pt, weight: "bold")[title])
    for idx in range(4) {
      let x = tx(idx)
      line((x, oy), (x, oy + h), stroke: 0.18pt + black30)
      content((x, oy - 0.17), text(fill: black70, size: 5.8pt)[#snr-label(slugs.at(idx))], anchor: "north")
    }
    let y-ticks = (y0, y0 + 0.5 * (y1 - y0), y1)
    for tick in y-ticks {
      let y = ty(tick)
      line((ox, y), (ox + w, y), stroke: 0.18pt + black30)
      content((ox - 0.12, y), text(fill: black70, size: 5.8pt)[#fmt(tick)], anchor: "east")
    }
    for idx in range(3) {
      let a0 = drive.at(slugs.at(idx)).at("delta_small_minus_wvf")
      let a1 = drive.at(slugs.at(idx + 1)).at("delta_small_minus_wvf")
      line((tx(idx), ty(a0)), (tx(idx + 1), ty(a1)), stroke: 0.9pt + atlantic)
      let b0 = hrf.at(slugs.at(idx)).at("delta_small_minus_wvf")
      let b1 = hrf.at(slugs.at(idx + 1)).at("delta_small_minus_wvf")
      line((tx(idx), ty(b0)), (tx(idx + 1), ty(b1)), stroke: 1.05pt + garnet)
    }
    for idx in range(4) {
      let drive-v = drive.at(slugs.at(idx)).at("delta_small_minus_wvf")
      let hrf-v = hrf.at(slugs.at(idx)).at("delta_small_minus_wvf")
      circle((tx(idx), ty(drive-v)), radius: 0.038, fill: atlantic, stroke: none)
      circle((tx(idx), ty(hrf-v)), radius: 0.042, fill: garnet, stroke: none)
    }
    content((ox + w / 2, oy - 0.42), text(fill: black90, size: 6.4pt)[Input SNR])
    content((0.14, oy + h / 2), angle: 90deg, text(fill: black90, size: 6.4pt)[y-label])
  })
}

#let delta-legend() = {
  grid(
    columns: 2,
    column-gutter: 12pt,
    [
      box(width: 10pt, height: 10pt, fill: atlantic, stroke: 0.2pt + black30)
      h(4pt)
      text(fill: black90, size: 6.4pt)[DRIVE 565×584]
    ],
    [
      box(width: 10pt, height: 10pt, fill: garnet, stroke: 0.2pt + black30)
      h(4pt)
      text(fill: black90, size: 6.4pt)[HRF 3504×2336]
    ],
  )
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
      text(fill: black70, size: 6.8pt)[Five HRF images across healthy, diabetic-retinopathy, and glaucoma cases. Filtering uses the green channel only. Noisy metrics average over #str(config.at("noise_draws")) AWGN draws.]
    ]
    v(8pt)
    image-grid(data, [Gradient magnitude, clean HRF retinal images], "magnitude_path", show-input: true)
  ]

  pagebreak()

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[Orientation maps. Hue encodes edge angle modulo $pi$ and brightness follows normalized magnitude.]
    ]
    v(8pt)
    image-grid(data, [Gradient orientation, clean HRF retinal images], "orientation_path", show-input: false)
  ]

  pagebreak()

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[Quantitative summary on vessel-boundary ODS, boundary-vector RMSE, and centerline tangent orientation MAE.]
    ]
    v(8pt)
    grid(
      columns: 2,
      column-gutter: 10pt,
      [
        metric-chart(data, "ods_f_score", [Soft-label ODS F-score versus SNR], [ODS F-score])
      ],
      [
        metric-chart(data, "gradient_vector_rmse_mean", [Gradient-vector RMSE versus SNR], [Vector RMSE])
      ],
      [
        metric-chart(data, "orientation_mae_deg_mean", [Centerline tangent MAE versus SNR], [Angle MAE (deg)])
      ],
      [
        summary-table(data)
      ],
    )
    v(10pt)
    legend(data)
  ]

  if "wvf_trace" in data {
    pagebreak()

    [
      align(center)[
        text(fill: black90, size: 11pt, weight: "bold")[title]
        linebreak()
        text(fill: black70, size: 8pt)[WVF radius-trace follow-up on the fixed HRF image set.]
        linebreak()
        text(fill: black70, size: 6.8pt)[The garnet trace sweeps WVF across $(r,d)=(3,5),(5,9),(9,11),(15,11),(25,11),(50,11)$. Thin horizontal lines are the fixed baseline methods.]
      ]
      v(8pt)
      grid(
        columns: 2,
        column-gutter: 10pt,
        row-gutter: 10pt,
        [
          wvf-trace-chart(data, "gradient_vector_rmse_mean", "inf", [Vector RMSE versus WVF radius, clean], [Vector RMSE])
        ],
        [
          wvf-trace-chart(data, "orientation_mae_deg_mean", "inf", [Orientation MAE versus WVF radius, clean], [Angle MAE (deg)])
        ],
        [
          wvf-trace-chart(data, "gradient_vector_rmse_mean", "10", [Vector RMSE versus WVF radius, 10 dB], [Vector RMSE])
        ],
        [
          wvf-trace-chart(data, "orientation_mae_deg_mean", "10", [Orientation MAE versus WVF radius, 10 dB], [Angle MAE (deg)])
        ],
      )
      v(8pt)
      legend(data)
    ]
  }

  pagebreak()

  [
    align(center)[
      text(fill: black90, size: 11pt, weight: "bold")[title]
      linebreak()
      text(fill: black70, size: 8pt)[Best-WVF versus best-small-stencil deltas. Positive values mean the best WVF grid cell has lower error than the best 3×3 stencil baseline.]
    ]
    v(8pt)
    grid(
      columns: 2,
      column-gutter: 10pt,
      [
        delta-chart(data, "gradient_vector_rmse_mean", [Vector RMSE delta, small-stencil minus best WVF], [RMSE delta])
      ],
      [
        delta-chart(data, "orientation_mae_deg_mean", [Orientation MAE delta, small-stencil minus best WVF], [Angle delta (deg)])
      ],
    )
    v(10pt)
    delta-legend()
  ]
}
