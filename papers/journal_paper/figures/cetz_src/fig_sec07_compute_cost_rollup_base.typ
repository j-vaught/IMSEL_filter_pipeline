#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, congaree, honeycomb, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.6pt)

#let backend-color(name) = {
  if name == "scipy_cpu" { return black90 }
  if name == "cpu_fft" { return atlantic }
  if name == "vkfft_cuda" { return garnet }
  congaree
}

#let shape-color(idx) = {
  let colors = (congaree, atlantic, honeycomb, garnet, black90)
  colors.at(calc.rem(idx, colors.len()))
}

#let render(data-path) = {
  let data = json(data-path)
  let title = data.at("title")
  let subtitle = data.at("subtitle")
  let rows = data.at("rows")
  let main = ()
  let alt = ()
  let max-main = 0.0
  let max-alt = 0.0
  for row in rows {
    if row.at("panel") == "main" {
      main.push(row)
      if row.at("throughput_mp_s") > max-main { max-main = row.at("throughput_mp_s") }
    } else {
      alt.push(row)
      if row.at("throughput_mp_s") > max-alt { max-alt = row.at("throughput_mp_s") }
    }
  }
  let main-y1 = if max-main > 0.0 { 1.10 * max-main } else { 1.0 }
  let alt-y1 = if max-alt > 0.0 { 1.10 * max-alt } else { 1.0 }

  cetz.canvas({
    import cetz.draw: *

    let left-w = 8.0
    let right-w = 3.8
    let ph = 4.7
    let gap = 1.2
    let ox = 0.95
    let oy = 0.88
    let main-x(v) = ox + v
    let main-y(v) = oy + v / main-y1 * ph
    let alt-x(v) = ox + left-w + gap + v
    let alt-y(v) = oy + v / alt-y1 * ph

    content((ox + (left-w + gap + right-w) / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[#title])
    content((ox + (left-w + gap + right-w) / 2, oy + ph + 0.52), text(fill: black70, size: 8.0pt)[#subtitle])

    line((ox, oy), (ox + left-w, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)
    line((ox + left-w + gap, oy), (ox + left-w + gap + right-w, oy), stroke: 0.8pt + black90)
    line((ox + left-w + gap, oy), (ox + left-w + gap, oy + ph), stroke: 0.8pt + black90)

    let y-ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    for tick in y-ticks {
      let y-main = main-y(tick * main-y1)
      line((ox, y-main), (ox + left-w, y-main), stroke: 0.18pt + black30)
      content((ox - 0.16, y-main), text(fill: black70, size: 6.2pt)[#str(calc.round(tick * main-y1, digits: 2))], anchor: "east")
      let y-alt = alt-y(tick * alt-y1)
      line((ox + left-w + gap, y-alt), (ox + left-w + gap + right-w, y-alt), stroke: 0.18pt + black30)
    }
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 8.3pt)[Throughput (MPix/s)])

    let groups = (
      ("r5_d9", "r=5 d=9"),
      ("r15_d11", "r=15 d=11"),
      ("r50_d11", "r=50 d=11"),
    )
    let backends = ("scipy_cpu", "cpu_fft", "vkfft_cuda")
    for g-idx in range(groups.len()) {
      let gx = ox + 0.55 + g-idx * 2.45
      let label = groups.at(g-idx).at(1)
      content((gx + 0.55, oy - 0.52), text(fill: black90, size: 7.0pt)[#label])
      for b-idx in range(backends.len()) {
        let backend = backends.at(b-idx)
        for row in main {
          if row.at("config_label") == groups.at(g-idx).at(0) and row.at("backend") == backend {
            let bx = gx + b-idx * 0.55
            rect((bx, oy), (bx + 0.34, main-y(row.at("throughput_mp_s"))), fill: backend-color(backend), stroke: none)
          }
        }
      }
    }

    let lx = ox + left-w - 1.6
    let ly = oy + ph - 0.1
    content((lx + 0.45, ly), text(fill: black90, size: 7.0pt, weight: "bold")[Backends])
    for idx in range(backends.len()) {
      let backend = backends.at(idx)
      let y = ly - 0.22 * (idx + 1)
      rect((lx, y - 0.05), (lx + 0.14, y + 0.05), fill: backend-color(backend), stroke: none)
      content((lx + 0.22, y), text(fill: black70, size: 6.2pt)[#backend], anchor: "west")
    }

    let alt-labels = ("triangle", "diamond", "hexagon", "octagon", "square")
    for idx in range(alt-labels.len()) {
      let label = alt-labels.at(idx)
      let bx = ox + left-w + gap + 0.35 + idx * 0.62
      for row in alt {
        if row.at("shape") == label {
          rect((bx, oy), (bx + 0.34, alt-y(row.at("throughput_mp_s"))), fill: shape-color(idx), stroke: none)
        }
      }
      content((bx + 0.17, oy - 0.52), angle: 45deg, text(fill: black90, size: 6.2pt)[#label])
    }
    content((ox + left-w / 2, oy + ph + 0.16), text(fill: black90, size: 7.6pt, weight: "bold")[Main backends])
    content((ox + left-w + gap + right-w / 2, oy + ph + 0.16), text(fill: black90, size: 7.6pt, weight: "bold")[Alt shapes at r=15 d=11])
  })
}
