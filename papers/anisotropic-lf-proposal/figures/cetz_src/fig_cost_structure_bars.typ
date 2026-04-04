#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let rose = rgb("#CC2E40")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let honeycomb = rgb("#A49137")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let white = rgb("#FFFFFF")

#let single-response = (
  ("Circular SG", 298.0, garnet),
  ("Square SG", 450.0, rose),
  ("WVF", 149.0, atlantic),
  ("LF", 2235.0, congaree),
  ("Fused LF", 351.0, honeycomb),
  ("Rect. SG", 402.0, black70),
  ("Ellip. SG", 307.0, horseshoe),
  ("Aniso. Gauss.", 402.0, black50),
)

#let bank-relative = (
  ("Circular SG", 1.00, garnet),
  ("Square SG", 450.0 / 298.0, rose),
  ("WVF", 894.0 / 298.0, atlantic),
  ("LF", 13410.0 / 298.0, congaree),
  ("Fused LF", 2106.0 / 298.0, honeycomb),
  ("Rect. SG", 2410.0 / 298.0, black70),
  ("Ellip. SG", 1842.0 / 298.0, horseshoe),
  ("Aniso. Gauss.", 2410.0 / 298.0, black50),
)

#let draw-chart(ox, oy, title, xlabel, data, max-val, tick-step, digits) = {
  import cetz.draw: *
  let label-w = 2.4
  let chart-w = 5.25
  let bar-h = 0.34
  let row-gap = 0.18
  let total-h = data.len() * bar-h + (data.len() - 1) * row-gap
  let top-y = oy
  let bottom-y = oy - total-h
  let axis-x = ox + label-w

  content((axis-x + chart-w / 2, top-y + 0.55), text(fill: black90, size: 10pt, weight: "bold")[#title])

  line((axis-x, bottom-y - 0.08), (axis-x + chart-w, bottom-y - 0.08), stroke: 0.8pt + black90)

  let n-ticks = calc.ceil(max-val / tick-step)
  for idx in range(n-ticks + 1) {
    let v = idx * tick-step
    if v <= max-val + 0.0001 {
      let tx = axis-x + (v / max-val) * chart-w
      line((tx, bottom-y - 0.08), (tx, top-y + 0.12), stroke: 0.3pt + black30)
      line((tx, bottom-y - 0.08), (tx, bottom-y - 0.18), stroke: 0.5pt + black90)
      content((tx, bottom-y - 0.3), anchor: "north", text(fill: black90, size: 8pt)[#str(int(v))])
    }
  }

  for (idx, entry) in data.enumerate() {
    let (label, value, col) = entry
    let y-top = top-y - idx * (bar-h + row-gap)
    let y-bot = y-top - bar-h
    let bw = (value / max-val) * chart-w
    rect((axis-x, y-top), (axis-x + bw, y-bot), fill: col, stroke: 0.4pt + black90)
    content((ox + label-w - 0.12, (y-top + y-bot) / 2), anchor: "east", text(fill: black90, size: 8.7pt)[#label])
    content((axis-x + bw + 0.12, (y-top + y-bot) / 2), anchor: "west", text(fill: black90, size: 8.4pt, weight: "bold")[#str(calc.round(value, digits: digits))])
  }

  content((axis-x + chart-w / 2, bottom-y - 0.62), text(fill: black90, size: 8.8pt)[#xlabel])
}

#cetz.canvas({
  import cetz.draw: *

  let left-x = 0.0
  let right-x = 8.85
  let base-y = 0.0

  draw-chart(
    left-x,
    base-y,
    [Single Response Evaluation],
    [Active weighted samples],
    single-response,
    2400.0,
    600.0,
    0,
  )

  draw-chart(
    right-x,
    base-y,
    [Full Bank Relative to Circular SG],
    [Relative dominant work],
    bank-relative,
    48.0,
    12.0,
    2,
  )

  let note-x = 8.4
  let note-y = -4.72
  rect(
    (note-x - 3.7, note-y + 0.38),
    (note-x + 3.7, note-y - 0.38),
    fill: black10,
    stroke: 0.4pt + black30,
  )
  content(
    (note-x, note-y),
    text(fill: black90, size: 8.2pt)[Representative values use $N = 15$, $r = 7$, $N_p = 149$, $m = 7$, and $Theta_6 = {0 deg, 30 deg, 60 deg, 90 deg, 120 deg, 150 deg}$.]
  )
})
