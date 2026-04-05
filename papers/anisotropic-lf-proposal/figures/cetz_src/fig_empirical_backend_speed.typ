#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let atlantic = rgb("#466A9F")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let rose = rgb("#CC2E40")
#let white = rgb("#FFFFFF")

#let panel-data = (
  ([Half-width $m = 1$], (
    ([Reference LF], 253.95, 0.8305, black70),
    ([Fused conv2d], 15.09, 0.7943, honeycomb),
    ([Fused Triton], 16.26, 0.8307, atlantic),
  ), [~15.6$times$ faster]),
  ([Half-width $m = 7$], (
    ([Reference LF], 1141.78, 0.7670, black70),
    ([Fused conv2d], 29.35, 0.7192, honeycomb),
    ([Fused Triton], 28.09, 0.7673, atlantic),
  ), [~40.8$times$ faster]),
)

#cetz.canvas({
  import cetz.draw: *

  let pw = 5.5
  let ph = 3.0
  let gap = 1.5
  let ox1 = 1.8
  let ox2 = ox1 + pw + gap
  let oy = 0.8
  let lx-min = 1.0
  let lx-max = 3.2

  let draw-panel(ox, title, data, speedup-label) = {
    let tx(v) = ox + (calc.log(v, base: 10) - lx-min) / (lx-max - lx-min) * pw
    let top-y = oy + ph
    let bar-h = 0.42
    let row-gap = 0.36

    content((ox + pw / 2, top-y + 0.52), text(fill: black90, size: 10pt, weight: "bold")[#title])
    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)

    let x-ticks = (10, 30, 100, 300, 1000)
    for v in x-ticks {
      let x = tx(v)
      line((x, oy), (x, oy - 0.10), stroke: 0.5pt + black70)
      line((x, oy), (x, top-y + 0.12), stroke: 0.25pt + black30)
      let label = if v >= 1000 { "1k" } else { str(v) }
      content((x, oy - 0.30), text(fill: black70, size: 7.7pt)[#label])
    }

    for (idx, entry) in data.enumerate() {
      let (label, value, ods, col) = entry
      let y-top = top-y - idx * (bar-h + row-gap)
      let y-bot = y-top - bar-h
      rect((tx(10), y-top), (tx(value), y-bot), fill: col, stroke: 0.4pt + black90)
      content((ox - 0.15, (y-top + y-bot) / 2), anchor: "east", text(fill: black90, size: 7.8pt)[#label])
      content((tx(value) + 0.12, (y-top + y-bot) / 2), anchor: "west", text(fill: black90, size: 7.3pt, weight: "bold")[#str(calc.round(value, digits: 2)) + " ms"])
      content((ox + pw + 0.55, (y-top + y-bot) / 2), anchor: "west", text(fill: black90, size: 7.3pt)[ODS #str(calc.round(ods, digits: 4))])
    }

    let ref-y = top-y - 0 * (bar-h + row-gap) - bar-h / 2
    let tri-y = top-y - 2 * (bar-h + row-gap) - bar-h / 2
    let ann-x = tx(650)
    line((ann-x, tri-y), (ann-x, ref-y), stroke: 0.8pt + rose)
    line((ann-x - 0.10, ref-y - 0.12), (ann-x, ref-y), stroke: 0.8pt + rose)
    line((ann-x + 0.10, ref-y - 0.12), (ann-x, ref-y), stroke: 0.8pt + rose)
    line((ann-x - 0.10, tri-y + 0.12), (ann-x, tri-y), stroke: 0.8pt + rose)
    line((ann-x + 0.10, tri-y + 0.12), (ann-x, tri-y), stroke: 0.8pt + rose)
    content((ann-x + 0.22, (ref-y + tri-y) / 2), anchor: "west", text(fill: rose, size: 7.8pt, weight: "bold")[#speedup-label])
  }

  draw-panel(ox1, panel-data.at(0).at(0), panel-data.at(0).at(1), panel-data.at(0).at(2))
  draw-panel(ox2, panel-data.at(1).at(0), panel-data.at(1).at(1), panel-data.at(1).at(2))

  content((ox1 + pw / 2, oy - 0.72), text(fill: black90, size: 10pt)[Median runtime per image (ms, log scale)])
  content((ox2 + pw / 2, oy - 0.72), text(fill: black90, size: 10pt)[Median runtime per image (ms, log scale)])

  let note-x = ox1 + pw + gap / 2
  let note-y = oy + ph + 1.1
  rect((note-x - 2.65, note-y + 0.34), (note-x + 2.65, note-y - 0.34), fill: black10, stroke: 0.4pt + black30)
  content((note-x, note-y), text(fill: black90, size: 7.3pt)[Backend benchmark. Average over BIPED v1 and v2. 10 images per split.])
})
