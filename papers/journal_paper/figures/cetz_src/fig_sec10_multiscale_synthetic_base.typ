#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black50, black30

#let white = rgb("#FFFFFF")

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.6pt)

#let heat-color(value, max-abs) = {
  if max-abs <= 1e-15 {
    return white
  }
  let t = calc.min(calc.max(calc.abs(value) / max-abs, 0.0), 1.0)
  if value >= 0.0 {
    color.mix((garnet, t * 100%), (white, (1.0 - t) * 100%))
  } else {
    color.mix((atlantic, t * 100%), (white, (1.0 - t) * 100%))
  }
}

#let render(data-path) = {
  let data = json(data-path)
  let heatmap = data.at("heatmap")
  let strategy-labels = heatmap.at("strategy_labels")
  let stimulus-labels = heatmap.at("stimulus_labels")
  let delta-matrix = heatmap.at("rmse_delta_matrix")
  let percent-matrix = heatmap.at("percent_improvement_matrix")

  let max-abs = 1e-15
  for row in delta-matrix {
    for value in row {
      if calc.abs(value) > max-abs {
        max-abs = calc.abs(value)
      }
    }
  }

  let cols = stimulus-labels.len()
  let rows = strategy-labels.len()
  let cell-w = 1.22
  let cell-h = 0.52
  let left-pad = 1.45
  let bottom-pad = 0.84
  let bar-pad = 0.64
  let total-w = cols * cell-w
  let total-h = rows * cell-h

  cetz.canvas({
    import cetz.draw: *

    content((left-pad + total-w / 2, bottom-pad + total-h + 0.98), text(fill: black90, size: 10pt, weight: "bold")[Section 10 multi-scale synthetic validation])
    content((left-pad + total-w / 2, bottom-pad + total-h + 0.54), text(fill: black70, size: 7.8pt)[Positive cells mean the multi-scale strategy lowers 10 dB step-edge RMSE relative to the best single-scale WVF trace point.])

    for row in range(rows) {
      for col in range(cols) {
        let x = left-pad + col * cell-w
        let y = bottom-pad + (rows - 1 - row) * cell-h
        let delta = delta-matrix.at(row).at(col)
        let pct = percent-matrix.at(row).at(col)
        rect((x, y), (x + cell-w, y + cell-h), fill: heat-color(delta, max-abs), stroke: 0.34pt + black50)
        content((x + cell-w / 2, y + cell-h / 2 + 0.06), text(fill: black90, size: 6.8pt, weight: "bold")[#str(calc.round(delta, digits: 4))])
        content((x + cell-w / 2, y + cell-h / 2 - 0.12), text(fill: black70, size: 5.8pt)[#str(calc.round(pct, digits: 1)) + "%"])
      }
    }

    for col in range(cols) {
      let x = left-pad + col * cell-w + cell-w / 2
      content((x, bottom-pad - 0.20), text(fill: black90, size: 6.8pt)[#stimulus-labels.at(col)])
    }
    content((left-pad + total-w / 2, bottom-pad - 0.54), text(fill: black90, size: 7.8pt)[Stimulus])

    for row in range(rows) {
      let y = bottom-pad + (rows - 1 - row) * cell-h + cell-h / 2
      content((left-pad - 0.18, y), text(fill: black90, size: 6.8pt)[#strategy-labels.at(row)], anchor: "east")
    }
    content((0.22, bottom-pad + total-h / 2), angle: 90deg, text(fill: black90, size: 7.8pt)[Combination strategy])

    let bar-x = left-pad + total-w + bar-pad
    let bar-y = bottom-pad
    let bar-h = total-h
    let bar-w = 0.24
    let steps = 64
    let step-h = bar-h / steps
    for idx in range(steps) {
      let alpha = idx / (steps - 1)
      let value = max-abs * (2.0 * alpha - 1.0)
      rect(
        (bar-x, bar-y + idx * step-h),
        (bar-x + bar-w, bar-y + (idx + 1) * step-h),
        fill: heat-color(value, max-abs),
        stroke: none,
      )
    }
    rect((bar-x, bar-y), (bar-x + bar-w, bar-y + bar-h), stroke: 0.42pt + black50)
    content((bar-x + bar-w + 0.16, bar-y), text(fill: black90, size: 6.4pt)[#str(calc.round(-max-abs, digits: 4))], anchor: "west")
    content((bar-x + bar-w + 0.16, bar-y + bar-h / 2), text(fill: black90, size: 6.4pt)[0.0], anchor: "west")
    content((bar-x + bar-w + 0.16, bar-y + bar-h), text(fill: black90, size: 6.4pt)[#str(calc.round(max-abs, digits: 4))], anchor: "west")
    content((bar-x + 0.70, bar-y + bar-h + 0.18), text(fill: black90, size: 7.0pt)[RMSE delta])

    content((left-pad + total-w / 2, bottom-pad + total-h + 0.20), text(fill: black70, size: 6.2pt)[Cell text shows raw RMSE delta on the first line and relative improvement on the second.])
  })
}
