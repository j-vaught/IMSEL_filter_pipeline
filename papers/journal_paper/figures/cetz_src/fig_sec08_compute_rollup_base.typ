#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, atlantic, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.4pt)

#let backend-color(name) = {
  if name == "scipy_cpu" { return black90 }
  if name == "cpu_fft" { return atlantic }
  if name == "vkfft_cuda" { return garnet }
  black70
}

#let backend-label(name) = {
  if name == "scipy_cpu" { return "CPU spatial" }
  if name == "cpu_fft" { return "CPU FFT" }
  if name == "vkfft_cuda" { return "CUDA VkFFT" }
  name
}

#let render(data-path) = {
  let data = json(data-path)
  let rows = data.at("rows")
  let order = data.at("method_order")
  let backends = data.at("config").at("backend_order")
  let max-x = 0.0
  for row in rows {
    if row.at("median_throughput_mp_s") > max-x { max-x = row.at("median_throughput_mp_s") }
  }
  let x1 = if max-x > 0.0 { 1.10 * max-x } else { 1.0 }

  cetz.canvas({
    import cetz.draw: *

    let ox = 1.45
    let oy = 0.75
    let pw = 7.35
    let row-step = 0.55
    let group-gap = 0.18
    let ph = order.len() * row-step
    let tx(v) = ox + v / x1 * pw

    content((ox + pw / 2, oy + ph + 0.95), text(fill: black90, size: 10pt, weight: "bold")[Section 8.5 compute and throughput])
    content((ox + pw / 2, oy + ph + 0.52), text(fill: black70, size: 8.0pt)[Median throughput on a $4096 times 4096$ image. The 95th-percentile latency is reported in the JSON summary.])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0) {
      let value = tick * x1
      let x = tx(value)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.2pt)[#str(calc.round(value, digits: 1))])
    }

    for idx in range(order.len()) {
      let method = order.at(idx)
      let y0 = oy + ph - (idx + 1) * row-step + 0.08
      content((ox - 0.12, y0 + 0.15), text(fill: black90, size: 6.8pt)[#method], anchor: "east")
      for b-idx in range(backends.len()) {
        let backend = backends.at(b-idx)
        let bar-y0 = y0 + b-idx * 0.12
        let bar-y1 = bar-y0 + 0.09
        for row in rows {
          if row.at("method") == method and row.at("backend") == backend {
            rect((ox, bar-y0), (tx(row.at("median_throughput_mp_s")), bar-y1), fill: backend-color(backend), stroke: none)
          }
        }
      }
    }

    let lx = ox + pw - 1.35
    let ly = oy + ph - 0.05
    for idx in range(backends.len()) {
      let backend = backends.at(idx)
      let y = ly - 0.21 * idx
      rect((lx, y - 0.05), (lx + 0.14, y + 0.05), fill: backend-color(backend), stroke: none)
      content((lx + 0.22, y), text(fill: black70, size: 6.0pt)[#backend-label(backend)], anchor: "west")
    }

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[Throughput (MPix/s)])
  })
}
