#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.6pt)

#let white = rgb("#FFFFFF")

#let line-style(snr) = {
  if snr == "clean" {
    return (paint: black90, thickness: 1.35pt, dash: "dashed")
  }
  if snr == "20 dB" {
    return (paint: atlantic, thickness: 1.25pt)
  }
  (paint: garnet, thickness: 1.35pt)
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let records = data.at("records")
  let modes = ("reflect", "zero", "constant_value", "edge")
  let mode-labels = (
    "reflect": "Reflection",
    "zero": "Zero",
    "constant_value": "Border-constant",
    "edge": "Clamp",
  )
  let snr-values = ("clean", "20 dB", "10 dB")
  let offset-labels = ("0", "15", "30", "60", "interior")
  let x-labels = ("0", "r", "2r", "4r", "interior")
  let x-values = (0, 1, 2, 3, 4)

  let y-min = 1e30
  let y-max = -1e30
  for rec in records {
    let value = rec.at("grad_rmse")
    if value < y-min { y-min = value }
    if value > y-max { y-max = value }
  }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 0.01 }
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let panel-w = 3.3
    let panel-h = 2.8
    let gap-x = 0.70
    let gap-y = 0.82
    let ox = 1.00
    let oy = 0.95

    let total-w = 2 * panel-w + gap-x
    let total-h = 2 * panel-h + gap-y

    content((ox + total-w / 2, oy + total-h + 0.92), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + total-w / 2, oy + total-h + 0.48), text(fill: black70, size: 8pt)[#subtitle])

    for mode-index in range(modes.len()) {
      let mode = modes.at(mode-index)
      let row = calc.floor(mode-index / 2)
      let col = calc.rem(mode-index, 2)
      let px = ox + col * (panel-w + gap-x)
      let py = oy + (1 - row) * (panel-h + gap-y)

      let tx(v) = px + v / 4 * panel-w
      let ty(v) = py + (v - y0) / (y1 - y0) * panel-h

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.6pt, weight: "bold")[#mode-labels.at(mode)])

      for idx in range(x-labels.len()) {
        let x = tx(x-values.at(idx))
        line((x, py), (x, py - 0.08), stroke: 0.40pt + black70)
        line((x, py), (x, py + panel-h), stroke: 0.18pt + black30)
        content((x, py - 0.24), text(fill: black70, size: 6.2pt)[#x-labels.at(idx)])
      }

      let y-ticks = (
        y0,
        y0 + 0.25 * (y1 - y0),
        y0 + 0.50 * (y1 - y0),
        y0 + 0.75 * (y1 - y0),
        y1,
      )
      for tick in y-ticks {
        let y = ty(tick)
        line((px, y), (px - 0.08, y), stroke: 0.40pt + black70)
        line((px, y), (px + panel-w, y), stroke: 0.18pt + black30)
        if col == 0 {
          content((px - 0.76, y), text(fill: black70, size: 6.1pt)[#str(calc.round(tick, digits: 4))])
        }
      }

      for snr in snr-values {
        let style = line-style(snr)
        let series = ()
        for offset in offset-labels {
          for rec in records {
            if rec.at("padding_mode") == mode and rec.at("snr_db") == snr and rec.at("edge_offset_px") == offset {
              series.push((offset, rec.at("grad_rmse")))
            }
          }
        }
        for idx in range(series.len() - 1) {
          let a = series.at(idx)
          let b = series.at(idx + 1)
          line(
            (tx(x-values.at(idx)), ty(a.at(1))),
            (tx(x-values.at(idx + 1)), ty(b.at(1))),
            stroke: style,
          )
        }
        for idx in range(series.len()) {
          let point = series.at(idx)
          circle((tx(x-values.at(idx)), ty(point.at(1))), radius: 0.05, fill: style.paint)
        }
      }
    }

    content((ox + total-w / 2, oy - 0.60), text(fill: black90, size: 8.4pt)[Edge offset from border])
    content((ox - 0.86, oy + total-h / 2), angle: 90deg, text(fill: black90, size: 8.4pt)[Gradient RMSE])

    let lx = ox + total-w + 0.28
    let ly = oy + total-h - 0.08
    rect((lx - 0.14, ly + 0.16), (lx + 1.62, ly - 0.92), fill: white, stroke: 0.35pt + black30)
    content((lx + 0.60, ly - 0.02), text(fill: black90, size: 7.2pt, weight: "bold")[SNR])
    for idx in range(snr-values.len()) {
      let snr = snr-values.at(idx)
      let y = ly - 0.24 * (idx + 1)
      line((lx, y), (lx + 0.42, y), stroke: line-style(snr))
      content((lx + 0.56, y), text(fill: black90, size: 6.6pt)[#snr], anchor: "west")
    }

    content((lx + 0.72, oy + 0.22), text(fill: black70, size: 6.2pt)[Offset 0 isolates the immediate border case.], anchor: "center")
  })
}
