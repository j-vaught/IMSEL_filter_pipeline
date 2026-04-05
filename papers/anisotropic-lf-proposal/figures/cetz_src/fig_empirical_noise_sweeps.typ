#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let honeycomb = rgb("#A49137")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black30 = rgb("#C7C7C7")
#let white = rgb("#FFFFFF")

#let x-values = (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)

#let gaussian-series = (
  ([WVF], garnet, ((0.3, 0.3667), (0.5, 0.3885), (1.0, 0.5033), (1.5, 0.6242), (2.0, 0.6896), (3.0, 0.7305), (4.0, 0.7393)), "solid"),
  ([LF Triton], atlantic, ((0.3, 0.3731), (0.5, 0.4094), (1.0, 0.5626), (1.5, 0.6534), (2.0, 0.6842), (3.0, 0.6977), (4.0, 0.7006)), "dashed"),
  ([Elliptical], horseshoe, ((0.3, 0.3553), (0.5, 0.3669), (1.0, 0.4509), (1.5, 0.5621), (2.0, 0.6363), (3.0, 0.6961), (4.0, 0.7143)), "dotted"),
  ([Iso. Gaussian], honeycomb, ((0.3, 0.3673), (0.5, 0.4121), (1.0, 0.5021), (1.5, 0.5302), (2.0, 0.5390), (3.0, 0.5447), (4.0, 0.5470)), "dash-dotted"),
)

#let speckle-series = (
  ([WVF], garnet, ((0.3, 0.4149), (0.5, 0.4839), (1.0, 0.6268), (1.5, 0.6966), (2.0, 0.7243), (3.0, 0.7386), (4.0, 0.7423)), "solid"),
  ([LF Triton], atlantic, ((0.3, 0.4401), (0.5, 0.5262), (1.0, 0.6408), (1.5, 0.6766), (2.0, 0.6885), (3.0, 0.6974), (4.0, 0.7012)), "dashed"),
  ([Elliptical], horseshoe, ((0.3, 0.3828), (0.5, 0.4361), (1.0, 0.5730), (1.5, 0.6537), (2.0, 0.6916), (3.0, 0.7166), (4.0, 0.7246)), "dotted"),
  ([Iso. Gaussian], honeycomb, ((0.3, 0.4171), (0.5, 0.4669), (1.0, 0.5152), (1.5, 0.5300), (2.0, 0.5374), (3.0, 0.5442), (4.0, 0.5471)), "dash-dotted"),
)

#cetz.canvas({
  import cetz.draw: *

  let pw = 5.1
  let ph = 5.2
  let gap = 1.5
  let ox1 = 1.1
  let ox2 = ox1 + pw + gap
  let oy = 1.0
  let x-min = 0.3
  let x-max = 4.0
  let y-min = 0.34
  let y-max = 0.76

  let draw-panel(ox, title, series) = {
    let tx(v) = ox + (v - x-min) / (x-max - x-min) * pw
    let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

    line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
    line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)
    content((ox + pw / 2, oy + ph + 0.45), text(fill: black90, size: 10pt, weight: "bold")[#title])

    for v in x-values {
      let x = tx(v)
      line((x, oy), (x, oy - 0.10), stroke: 0.5pt + black70)
      line((x, oy), (x, oy + ph), stroke: 0.25pt + black30)
      content((x, oy - 0.30), text(fill: black70, size: 7.7pt)[#str(v)])
    }

    let y-ticks = (0.35, 0.45, 0.55, 0.65, 0.75)
    for v in y-ticks {
      let y = ty(v)
      line((ox, y), (ox - 0.10, y), stroke: 0.5pt + black70)
      line((ox, y), (ox + pw, y), stroke: 0.25pt + black30)
      content((ox - 0.42, y), anchor: "east", text(fill: black70, size: 7.7pt)[#str(calc.round(v, digits: 2))])
    }

    for entry in series {
      let (label, col, pts, dash) = entry
      for idx in range(pts.len() - 1) {
        let (x1, y1) = pts.at(idx)
        let (x2, y2) = pts.at(idx + 1)
        line((tx(x1), ty(y1)), (tx(x2), ty(y2)), stroke: (paint: col, thickness: 1.5pt, dash: dash))
      }
      for (xv, yv) in pts {
        circle((tx(xv), ty(yv)), radius: 0.07, fill: col, stroke: none)
      }
    }
  }

  draw-panel(ox1, [Gaussian noise], gaussian-series)
  draw-panel(ox2, [Speckle noise], speckle-series)

  content((ox1 + pw / 2, oy - 0.70), text(fill: black90, size: 10pt)[SNR])
  content((ox2 + pw / 2, oy - 0.70), text(fill: black90, size: 10pt)[SNR])
  content((0.15, oy + ph / 2), angle: 90deg, text(fill: black90, size: 10pt)[Average ODS over BIPED v1 and v2])

  let lx = ox2 - 0.3
  let ly = oy + ph + 0.62
  let ls = 0.42
  rect((lx - 6.2, ly + 0.24), (lx + 0.6, ly - 0.24), fill: white, stroke: 0pt)
  let legend = (
    ([WVF], garnet, "solid"),
    ([LF Triton], atlantic, "dashed"),
    ([Elliptical], horseshoe, "dotted"),
    ([Iso. Gaussian], honeycomb, "dash-dotted"),
  )
  for (idx, entry) in legend.enumerate() {
    let (label, col, dash) = entry
    let x0 = ox1 + idx * 2.9
    line((x0, ly), (x0 + 0.5, ly), stroke: (paint: col, thickness: 1.5pt, dash: dash))
    circle((x0 + 0.25, ly), radius: 0.06, fill: col, stroke: none)
    content((x0 + 0.65, ly), anchor: "west", text(fill: black90, size: 7.8pt)[#label])
  }
})
