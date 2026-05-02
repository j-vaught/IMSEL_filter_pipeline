#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black50, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.6pt)

#let render(data-path, noise-type, figure-title) = {
  let data = json(data-path)
  let subtitle = data.at("subtitle")
  let records = ()
  let severities = ()
  for rec in data.at("noise_records") {
    if rec.at("noise_type") == noise-type {
      records.push(rec)
      if not severities.contains(rec.at("severity_label")) {
        severities.push(rec.at("severity_label"))
      }
    }
  }

  let frequencies = ()
  for rec in records {
    if not frequencies.contains(rec.at("frequency_cyc_px")) {
      frequencies.push(rec.at("frequency_cyc_px"))
    }
  }
  frequencies = frequencies.sorted()

  let y-min = 1e30
  let y-max = -1e30
  for rec in records {
    let value = rec.at("mag_error")
    if value < y-min { y-min = value }
    if value > y-max { y-max = value }
  }
  let y-pad = if y-max > y-min { 0.10 * (y-max - y-min) } else { 0.01 }
  let y0 = y-min - y-pad
  let y1 = y-max + y-pad

  cetz.canvas({
    import cetz.draw: *

    let panel-w = 3.15
    let panel-h = 4.2
    let gap = 0.48
    let ox = 0.90
    let oy = 0.88

    content((ox + 1.5 * panel-w + gap, oy + panel-h + 0.94), text(fill: black90, size: 10pt, weight: "bold")[#figure-title])
    content((ox + 1.5 * panel-w + gap, oy + panel-h + 0.48), text(fill: black70, size: 8pt)[#subtitle])

    for panel-idx in range(severities.len()) {
      let severity = severities.at(panel-idx)
      let px = ox + panel-idx * (panel-w + gap)
      let py = oy

      let tx(v) = px + calc.log(v / frequencies.first()) / calc.log(frequencies.last() / frequencies.first()) * panel-w
      let ty(v) = py + (v - y0) / (y1 - y0) * panel-h

      rect((px, py), (px + panel-w, py + panel-h), stroke: 0.45pt + black30)
      content((px + panel-w / 2, py + panel-h + 0.18), text(fill: black90, size: 7.6pt, weight: "bold")[#severity])

      for freq in frequencies {
        let x = tx(freq)
        line((x, py), (x, py - 0.08), stroke: 0.4pt + black70)
        line((x, py), (x, py + panel-h), stroke: 0.18pt + black30)
        content((x, py - 0.24), text(fill: black70, size: 6.2pt)[#str(calc.round(freq, digits: 4))])
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
        line((px, y), (px - 0.08, y), stroke: 0.4pt + black70)
        line((px, y), (px + panel-w, y), stroke: 0.18pt + black30)
        if panel-idx == 0 {
          content((px - 0.74, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick, digits: 4))])
        }
      }

      let orientations = ()
      for rec in records {
        if rec.at("severity_label") == severity and not orientations.contains(rec.at("orientation_deg")) {
          orientations.push(rec.at("orientation_deg"))
        }
      }
      orientations = orientations.sorted()
      let mean-series = ()
      for freq in frequencies {
        let values = ()
        for rec in records {
          if rec.at("severity_label") == severity and rec.at("frequency_cyc_px") == freq {
            values.push(rec.at("mag_error"))
          }
        }
        let total = 0.0
        for value in values {
          total += value
        }
        mean-series.push((freq, total / values.len()))
      }

      for orientation in orientations {
        let series = ()
        for freq in frequencies {
          for rec in records {
            if rec.at("severity_label") == severity and rec.at("orientation_deg") == orientation and rec.at("frequency_cyc_px") == freq {
              series.push((freq, rec.at("mag_error")))
            }
          }
        }
        for idx in range(series.len() - 1) {
          let a = series.at(idx)
          let b = series.at(idx + 1)
          line((tx(a.at(0)), ty(a.at(1))), (tx(b.at(0)), ty(b.at(1))), stroke: 0.28pt + atlantic)
        }
      }

      for idx in range(mean-series.len() - 1) {
        let a = mean-series.at(idx)
        let b = mean-series.at(idx + 1)
        line((tx(a.at(0)), ty(a.at(1))), (tx(b.at(0)), ty(b.at(1))), stroke: 1.0pt + garnet)
      }
    }

    content((ox + 1.5 * panel-w + gap, oy - 0.58), text(fill: black90, size: 8.2pt)[Frequency (cyc/px)])
    content((ox - 0.72, oy + panel-h / 2), angle: 90deg, text(fill: black90, size: 8.2pt)[Magnitude error])
    content((ox + 1.5 * panel-w + gap, oy + panel-h + 0.16), text(fill: black50, size: 6.4pt)[Thin lines show orientations. Thick garnet shows the orientation mean.])
  })
}
