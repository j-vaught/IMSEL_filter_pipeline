#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, rose, atlantic, congaree, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.5pt)

#let white = rgb("#FFFFFF")
#let neg-palette = (atlantic, congaree)
#let pos-palette = (rose, garnet)

#let color-for(value, max-abs) = {
  if max-abs <= 1e-18 {
    return white
  }
  let t = calc.clamp(value / max-abs, -1, 1)
  if t >= 0 {
    return if t < 0.5 { rose } else { garnet }
  }
  return if t > -0.5 { atlantic } else { congaree }
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let records = data.at("records")

  cetz.canvas({
    import cetz.draw: *

    let cols = 3
    let panel-w = 3.35
    let panel-h = 3.35
    let gap-x = 0.55
    let gap-y = 0.70
    let ox = 0.45
    let oy = 0.65

    let total-w = cols * panel-w + (cols - 1) * gap-x
    content((ox + total-w / 2, oy + 8.1), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + total-w / 2, oy + 7.66), text(fill: black70, size: 8pt)[#subtitle])

    for idx in range(records.len()) {
      let rec = records.at(idx)
      let row = calc.floor(idx / cols)
      let col = calc.rem(idx, cols)
      let px0 = ox + col * (panel-w + gap-x)
      let py0 = oy + (1 - row) * (panel-h + gap-y)
      let kernel = rec.at("kernel_x")

      rect((px0, py0), (px0 + panel-w, py0 + panel-h), stroke: 0.45pt + black30)
      content((px0 + panel-w / 2, py0 + panel-h + 0.22), text(fill: black90, size: 8pt, weight: "bold")[r = #rec.at("radius"), d_max = #rec.at("degree_max")])

      if kernel == none {
        content((px0 + panel-w / 2, py0 + panel-h / 2), text(fill: black70, size: 8pt)[construction failed])
      } else {
        let rows = kernel.len()
        let cols-k = kernel.at(0).len()
        let cell-w = panel-w / cols-k
        let cell-h = panel-h / rows
        let max-abs = rec.at("kernel_max")
        for iy in range(rows) {
          let row-data = kernel.at(iy)
          for ix in range(cols-k) {
            let value = row-data.at(ix)
            let x0 = px0 + ix * cell-w
            let y0 = py0 + (rows - 1 - iy) * cell-h
            rect((x0, y0), (x0 + cell-w, y0 + cell-h), fill: color-for(value, max-abs))
          }
        }
      }

      let badge-y = py0 - 0.18
      content((px0, badge-y), text(fill: black70, size: 6.6pt)[|S|=#rec.at("support_cardinality"), M=#rec.at("coefficient_count")], anchor: "west")
      content((px0, badge-y - 0.18), text(fill: black70, size: 6.6pt)[kappa=#str(calc.round(rec.at("kappa") * 100) / 100)], anchor: "west")
      content((px0, badge-y - 0.36), text(fill: black70, size: 6.6pt)[sigma_min=#str(calc.round(rec.at("sigma_min") * 1e12) / 1e12)], anchor: "west")
      content((px0, badge-y - 0.54), text(fill: black70, size: 6.6pt)[step RMSE=#str(calc.round(rec.at("step_grad_rmse") * 1e6) / 1e6)], anchor: "west")
      content((px0, badge-y - 0.72), text(fill: black70, size: 6.6pt)[arc RMSE=#str(calc.round(rec.at("arc_grad_rmse") * 1e6) / 1e6)], anchor: "west")
      content((px0, badge-y - 0.90), text(fill: black70, size: 6.6pt)[method=#rec.at("application_method")], anchor: "west")
    }
  })
}
