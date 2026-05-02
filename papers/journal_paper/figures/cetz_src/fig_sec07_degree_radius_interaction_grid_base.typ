#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.3pt)

#let white = rgb("#FFFFFF")

#let metric-labels = (
  ("arc_grad_rmse", "Arc RMSE"),
  ("kappa_design_matrix", "kappa(A)"),
  ("rank_deficient_count", "Rank-def count"),
  ("white_noise_gain", "White-noise gain"),
)

#let clamp01(x) = calc.min(calc.max(x, 0.0), 1.0)

#let metric-color(metric, value, min-value, max-value) = {
  if max-value <= min-value {
    return white
  }
  let mapped = if metric == "kappa_design_matrix" {
    let lv = calc.log(value) / calc.log(10)
    let lmin = calc.log(min-value) / calc.log(10)
    let lmax = calc.log(max-value) / calc.log(10)
    clamp01((lv - lmin) / (lmax - lmin))
  } else {
    clamp01((value - min-value) / (max-value - min-value))
  }
  color.mix((garnet, mapped * 100%), (white, (1.0 - mapped) * 100%))
}

#let fmt-short(value) = {
  if value >= 1000 or value < 0.001 {
    return str(value)
  }
  return str(calc.round(value, digits: 4))
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let config = data.at("config")
  let states = data.at("states")
  let radii = config.at("radii")
  let degrees = config.at("degrees")

  let metric-bounds = ()
  for metric in metric-labels {
    let key = metric.at(0)
    let min-value = none
    let max-value = none
    for state in states {
      for cell in state.at("cells") {
        let value = cell.at(key)
        if value == none {
          continue
        }
        if min-value == none or value < min-value {
          min-value = value
        }
        if max-value == none or value > max-value {
          max-value = value
        }
      }
    }
    metric-bounds.push((key, min-value, max-value))
  }

  #cetz.canvas({
    import cetz.draw: *

    let cell-w = 0.48
    let cell-h = 0.38
    let panel-w = degrees.len() * cell-w
    let panel-h = radii.len() * cell-h
    let col-gap = 1.05
    let row-gap = 0.90
    let left-pad = 0.80
    let bottom-pad = 0.80
    let total-w = 2 * panel-w + col-gap
    let total-h = 4 * panel-h + 3 * row-gap

    content((left-pad + total-w / 2, bottom-pad + total-h + 1.15), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((left-pad + total-w / 2, bottom-pad + total-h + 0.68), text(fill: black70, size: 8pt)[#subtitle])

    for state-index in range(states.len()) {
      let state = states.at(state-index)
      let px = left-pad + state-index * (panel-w + col-gap)
      let py = bottom-pad + total-h + 0.16
      let badge = state.at("recommendation")
      let pieces = ()
      for rec in badge {
        let shown = if rec.at("max_useful_degree") == none { "none" } else { str(rec.at("max_useful_degree")) }
        pieces.push("r=" + str(rec.at("radius")) + "→d=" + shown)
      }
      let label = if state.at("normalize_coords") { "normalize_coords = True" } else { "normalize_coords = False" }
      content((px + panel-w / 2, py), text(fill: black90, size: 8.2pt, weight: "bold")[#label])
      content((px + panel-w / 2, py - 0.28), text(fill: black70, size: 6.8pt)[#pieces.join(", ")])
    }

    for metric-index in range(metric-labels.len()) {
      let metric-key = metric-labels.at(metric-index).at(0)
      let metric-text = metric-labels.at(metric-index).at(1)
      let bound = metric-bounds.at(metric-index)
      let min-value = bound.at(1)
      let max-value = bound.at(2)
      for state-index in range(states.len()) {
        let state = states.at(state-index)
        let px = left-pad + state-index * (panel-w + col-gap)
        let py = bottom-pad + (metric-labels.len() - 1 - metric-index) * (panel-h + row-gap)

        rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
        content((px + panel-w / 2, py + panel-h + 0.20), text(fill: black90, size: 7.8pt, weight: "bold")[#metric-text])

        for row-index in range(radii.len()) {
          let radius = radii.at(row-index)
          for col-index in range(degrees.len()) {
            let degree = degrees.at(col-index)
            let found = none
            for cell in state.at("cells") {
              if cell.at("radius") == radius and cell.at("degree") == degree {
                found = cell
              }
            }
            if found == none {
              continue
            }
            let x0 = px + col-index * cell-w
            let y0 = py + (radii.len() - 1 - row-index) * cell-h
            let value = found.at(metric-key)
            if value == none {
              rect(
                (x0, y0),
                (x0 + cell-w, y0 + cell-h),
                fill: white,
                stroke: 0.22pt + black50,
              )
              line((x0, y0), (x0 + cell-w, y0 + cell-h), stroke: 0.18pt + atlantic)
              line((x0 + cell-w, y0), (x0, y0 + cell-h), stroke: 0.18pt + atlantic)
              continue
            }
            rect(
              (x0, y0),
              (x0 + cell-w, y0 + cell-h),
              fill: metric-color(metric-key, value, min-value, max-value),
              stroke: 0.22pt + black50,
            )
          }
        }

        for col-index in range(degrees.len()) {
          let x = px + col-index * cell-w + cell-w / 2
          content((x, py - 0.18), text(fill: black90, size: 6.7pt)[#degrees.at(col-index)])
        }
        if metric-index == metric-labels.len() - 1 {
          content((px + panel-w / 2, py - 0.50), text(fill: black90, size: 7.5pt)[Polynomial degree $d$])
        }

        if state-index == 0 {
          for row-index in range(radii.len()) {
            let y = py + (radii.len() - 1 - row-index) * cell-h + cell-h / 2
            content((px - 0.16, y), text(fill: black90, size: 6.7pt)[#radii.at(row-index)], anchor: "east")
          }
          content((0.18, py + panel-h / 2), angle: 90deg, text(fill: black90, size: 7.6pt)[Support radius $r$])
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
            fill: metric-color(metric-key, value, min-value, max-value),
            stroke: none,
          )
        }
        rect((bar-x, py), (bar-x + bar-w, py + panel-h), stroke: 0.35pt + black50)
        if min-value != none and max-value != none {
          content((bar-x + bar-w + 0.10, py), text(fill: black70, size: 6.3pt)[#fmt-short(min-value)], anchor: "west")
          content((bar-x + bar-w + 0.10, py + panel-h), text(fill: black70, size: 6.3pt)[#fmt-short(max-value)], anchor: "west")
        }
        if metric-key == "kappa_design_matrix" {
          content((bar-x + 0.34, py + panel-h + 0.08), text(fill: atlantic, size: 6.5pt)[log scale])
        }
      }
    }
  })
}
