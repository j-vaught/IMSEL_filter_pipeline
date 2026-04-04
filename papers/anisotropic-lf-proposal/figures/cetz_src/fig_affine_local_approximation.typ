#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let sandstorm = rgb("#FFF2E3")
#let white = rgb("#FFFFFF")

#let panel-w = 4.6
#let panel-h = 2.4
#let gap-x = 0.9
#let x-min = -2.6
#let x-max = 2.6
#let y-min = -0.45
#let y-max = 1.85
#let support-half-width = 0.85

#let tx(ox, x) = ox + (x - x-min) / (x-max - x-min) * panel-w
#let ty(oy, y) = oy - (y - y-min) / (y-max - y-min) * panel-h

#let smooth-profile(x) = 0.88 + 0.24 * x + 0.11 * x * x
#let tangent-profile(x) = 0.88 + 0.24 * x

#let draw-curve(ox, oy, func, stroke-style) = {
  import cetz.draw: *
  let pts = ()
  let steps = 80
  for i in range(steps + 1) {
    let t = i * 1.0 / steps
    let x = x-min + (x-max - x-min) * t
    let y = func(x)
    pts.push((tx(ox, x), ty(oy, y)))
  }
  for i in range(pts.len() - 1) {
    line(pts.at(i), pts.at(i + 1), stroke: stroke-style)
  }
}

#let draw-panel(ox, oy, show-tangent: false) = {
  import cetz.draw: *
  rect((ox, oy), (ox + panel-w, oy - panel-h), fill: white, stroke: 0.8pt + black90)

  let x-axis-y = ty(oy, 0.0)
  let y-axis-x = tx(ox, 0.0)
  line((ox + 0.18, x-axis-y), (ox + panel-w - 0.18, x-axis-y), stroke: 0.5pt + black50)
  line((y-axis-x, oy - panel-h + 0.18), (y-axis-x, oy - 0.18), stroke: 0.5pt + black50)

  let sx0 = tx(ox, -support-half-width)
  let sx1 = tx(ox, support-half-width)
  rect((sx0, oy - 0.06), (sx1, oy - panel-h + 0.06), fill: sandstorm.transparentize(8%), stroke: 0.4pt + black30)

  draw-curve(ox, oy, smooth-profile, 1.7pt + atlantic)

  if show-tangent {
    draw-curve(ox, oy, tangent-profile, 1.7pt + garnet)
    content((tx(ox, 1.55), ty(oy, tangent-profile(1.55)) + 0.15), text(fill: garnet, size: 8.5pt)[local affine approximation])
    content((tx(ox, 1.55), ty(oy, smooth-profile(1.55)) - 0.18), text(fill: black70, size: 8pt)[smooth profile])
  } else {
    content((tx(ox, 1.35), ty(oy, smooth-profile(1.35)) + 0.16), text(fill: atlantic, size: 8.5pt)[smooth local profile])
  }

  let cx = tx(ox, 0.0)
  let cy = ty(oy, smooth-profile(0.0))
  circle((cx, cy), radius: 0.05, fill: black90, stroke: none)

  content((sx0 + (sx1 - sx0) / 2.0, oy - panel-h - 0.2), text(fill: black70, size: 8.5pt)[local support window])
  content((ox + panel-w / 2.0, oy + 0.32), text(fill: black90, size: 10pt, weight: "bold")[#if show-tangent {[first-order local approximation]} else {[smooth intensity profile]}])
  content((ox + panel-w - 0.06, x-axis-y + 0.14), text(fill: black70, size: 8pt)[$x$], anchor: "east")
  content((y-axis-x + 0.08, oy - 0.12), text(fill: black70, size: 8pt)[$I(x)$], anchor: "west")
}

#cetz.canvas({
  import cetz.draw: *
  let left-x = 0.0
  let right-x = left-x + panel-w + gap-x

  draw-panel(left-x, 0.0, show-tangent: false)
  draw-panel(right-x, 0.0, show-tangent: true)

  line(
    (tx(right-x, 0.0), ty(0.0, smooth-profile(0.0))),
    (tx(right-x, 0.0) + 0.85, ty(0.0, smooth-profile(0.0)) - 0.42),
    stroke: 0.5pt + black50,
    mark: (end: "stealth", fill: black50, scale: 0.45),
  )
  content(
    (tx(right-x, 0.0) + 1.14, ty(0.0, smooth-profile(0.0)) - 0.47),
    text(fill: black70, size: 8pt)[tangent at the center pixel],
    anchor: "west",
  )
})
