#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, black90, black70, black50, black30, atlantic, sandstorm

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.3pt)

#let white = rgb("#FFFFFF")

#let metric-defs = (
  ("step_grad_rmse", "Step gradient RMSE", false),
  ("arc_orientation_mae_deg", "Arc orientation MAE (deg)", false),
)

#let clamp01(x) = calc.min(calc.max(x, 0.0), 1.0)

#let metric-color(value, min-value, max-value) = {
  if max-value <= min-value {
    return white
  }
  let mapped = clamp01((value - min-value) / (max-value - min-value))
  color.mix((white, mapped * 100%), (garnet, (1.0 - mapped) * 100%))
}

#let fmt-short(value) = {
  if value >= 1000 or value < 0.001 {
    return str(value)
  }
  str(calc.round(value, digits: 4))
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let config = data.at("config")
  let widths = config.at("edge_widths_px")
  let trace = config.at("wvf_trace")
  let radii = ()
  for item in trace {
    radii.push(item.at("radius"))
  }
  let cells = data.at("cells")
  let best-by-width = data.at("best_by_width")

  let bounds = ()
  for metric in metric-defs {
    let key = metric.at(0)
    let min-value = none
    let max-value = none
    for cell in cells {
      let value = cell.at(key)
      if min-value == none or value < min-value { min-value = value }
      if max-value == none or value > max-value { max-value = value }
    }
    bounds.push((key, min-value, max-value))
  }

  cetz.canvas({
    import cetz.draw: *

    let cell-w = 0.72
    let cell-h = 0.42
    let panel-w = widths.len() * cell-w
    let panel-h = radii.len() * cell-h
    let col-gap = 1.30
    let left-pad = 0.92
    let bottom-pad = 1.18
    let total-w = 2 * panel-w + col-gap
    let total-h = panel-h

    content((left-pad + total-w / 2, bottom-pad + total-h + 0.98), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((left-pad + total-w / 2, bottom-pad + total-h + 0.60), text(fill: black70, size: 8pt)[#subtitle])
    content((left-pad + total-w / 2, bottom-pad + total-h + 0.28), text(fill: black70, size: 6.8pt)[AWGN 10 dB, #str(config.at("noise_draws")) draws, 36 orientations, 4 phases, curved-bank $rho = 4r$. Darker cells are lower error.])

    for metric-index in range(metric-defs.len()) {
      let metric-key = metric-defs.at(metric-index).at(0)
      let metric-label = metric-defs.at(metric-index).at(1)
      let bound = bounds.at(metric-index)
      let min-value = bound.at(1)
      let max-value = bound.at(2)
      let px = left-pad + metric-index * (panel-w + col-gap)
      let py = bottom-pad

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.8pt, weight: "bold")[#metric-label])

      for row-index in range(radii.len()) {
        let radius = radii.at(row-index)
        for col-index in range(widths.len()) {
          let edge-width = widths.at(col-index)
          let found = none
          for cell in cells {
            if cell.at("radius") == radius and cell.at("edge_width_px") == edge-width {
              found = cell
            }
          }
          if found == none {
            continue
          }
          let x0 = px + col-index * cell-w
          let y0 = py + (radii.len() - 1 - row-index) * cell-h
          let value = found.at(metric-key)
          rect(
            (x0, y0),
            (x0 + cell-w, y0 + cell-h),
            fill: metric-color(value, min-value, max-value),
            stroke: 0.24pt + black50,
          )

          for badge in best-by-width {
            if badge.at("edge_width_px") != edge-width {
              continue
            }
            let best-radius = if metric-key == "step_grad_rmse" { badge.at("best_step_radius") } else { badge.at("best_arc_radius") }
            if int(best-radius) == int(radius) {
              rect(
                (x0 + 0.03, y0 + 0.03),
                (x0 + cell-w - 0.03, y0 + cell-h - 0.03),
                stroke: 0.40pt + atlantic,
              )
            }
          }
        }
      }

      for col-index in range(widths.len()) {
        let x = px + col-index * cell-w + cell-w / 2
        content((x, py - 0.18), text(fill: black90, size: 6.7pt)[#widths.at(col-index)])
      }
      content((px + panel-w / 2, py - 0.48), text(fill: black90, size: 7.3pt)[Conceptual edge width $w$ (px)])

      if metric-index == 0 {
        for row-index in range(radii.len()) {
          let y = py + (radii.len() - 1 - row-index) * cell-h + cell-h / 2
          content((px - 0.16, y), text(fill: black90, size: 6.7pt)[#radii.at(row-index)], anchor: "east")
        }
        content((0.18, py + panel-h / 2), angle: 90deg, text(fill: black90, size: 7.5pt)[WVF radius $r$])
      }

      let bar-x = px + panel-w + 0.16
      let bar-w = 0.16
      let steps = 40
      let step-h = panel-h / steps
      for idx in range(steps) {
        let t = idx / (steps - 1)
        let value = min-value + t * (max-value - min-value)
        rect(
          (bar-x, py + idx * step-h),
          (bar-x + bar-w, py + (idx + 1) * step-h),
          fill: metric-color(value, min-value, max-value),
          stroke: none,
        )
      }
      rect((bar-x, py), (bar-x + bar-w, py + panel-h), stroke: 0.35pt + black50)
      content((bar-x + bar-w + 0.10, py), text(fill: black70, size: 6.3pt)[#fmt-short(min-value)], anchor: "west")
      content((bar-x + bar-w + 0.10, py + panel-h), text(fill: black70, size: 6.3pt)[#fmt-short(max-value)], anchor: "west")
    }

    let table-x = left-pad
    let table-y = bottom-pad - 1.00
    let columns = 5
    let table-cells = ()
    table-cells.push([w (px)])
    table-cells.push([3w-5w band])
    table-cells.push([best $r$ by step RMSE])
    table-cells.push([best $r$ by arc MAE])
    table-cells.push([in band?])
    for row in best-by-width {
      table-cells.push([str(row.at("edge_width_px"))])
      let band = row.at("hypothesis_radius_band_px")
      table-cells.push([str(calc.round(band.at(0), digits: 1)) + "-" + str(calc.round(band.at(1), digits: 1))])
      table-cells.push([str(row.at("best_step_radius"))])
      table-cells.push([str(row.at("best_arc_radius"))])
      let verdict = (if row.at("best_step_in_band") { "step" } else { "" }) + (if row.at("best_step_in_band") and row.at("best_arc_in_band") { ", " } else { "" }) + (if row.at("best_arc_in_band") { "arc" } else { "" })
      table-cells.push([(if verdict == "" { "neither" } else { verdict })])
    }
    content((table-x, table-y), table(columns: columns, stroke: 0.35pt + black30, inset: 3pt, align: center, ..table-cells), anchor: "south-west")
  })
}
