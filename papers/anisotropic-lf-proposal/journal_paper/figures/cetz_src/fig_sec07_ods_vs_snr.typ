#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")

#cetz.canvas({
  import cetz.draw: *

  // Plot dimensions
  let pw = 10.0
  let ph = 6.5
  let ox = 1.8
  let oy = 1.0

  // X-axis: SNR from 0 to 4.5 (with 4.5 representing clean/inf)
  // We'll map: 0.3 -> 0.3, 0.5 -> 0.5, 1.0 -> 1.0, etc., clean -> 4.5
  let x-min = 0.0
  let x-max = 4.8

  // Y-axis: ODS from 0.25 to 0.95
  let y-min = 0.25
  let y-max = 0.95

  let tx(v) = ox + (v - x-min) / (x-max - x-min) * pw
  let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

  // === AXES ===
  line((ox, oy), (ox + pw, oy), stroke: 0.8pt + black90)
  line((ox, oy), (ox, oy + ph), stroke: 0.8pt + black90)

  // X-axis label
  content((ox + pw/2, oy - 0.7), text(fill: black90, size: 10pt)[SNR (dB)])

  // Y-axis label
  content(
    (ox - 1.35, oy + ph/2),
    text(fill: black90, size: 10pt)[ODS F-score],
    angle: 90deg,
  )

  // X ticks
  let x-tick-vals = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5)
  let x-tick-labels = ("0.5", "1.0", "1.5", "2.0", "3.0", "4.0", "clean")
  for (i, v) in x-tick-vals.enumerate() {
    let x = tx(v)
    line((x, oy), (x, oy - 0.1), stroke: 0.5pt + black70)
    content((x, oy - 0.35), text(fill: black70, size: 8pt)[#x-tick-labels.at(i)])
    line((x, oy), (x, oy + ph), stroke: 0.2pt + black30)
  }

  // Y ticks
  let y-ticks = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
  for v in y-ticks {
    let y = ty(v)
    line((ox, y), (ox - 0.1, y), stroke: 0.5pt + black70)
    let label = str(calc.round(v, digits: 1))
    content((ox - 0.45, y), text(fill: black70, size: 8pt)[#label])
    line((ox, y), (ox + pw, y), stroke: 0.2pt + black30)
  }

  // === DATA ===
  // SNR values: 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, clean(4.5)
  // All data extracted from BSDS500 gaussian noise ablation

  // ASF (LF best config) - Garnet, thick line
  let asf-data = (
    (0.3, 0.5743),
    (0.5, 0.6116),
    (1.0, 0.7244),
    (1.5, 0.7918),
    (2.0, 0.8283),
    (3.0, 0.8658),
    (4.0, 0.8833),
    (4.5, 0.8996),
  )

  // DexiNed - Atlantic
  let dexined-data = (
    (0.3, 0.3293),
    (0.5, 0.3290),
    (1.0, 0.3290),
    (1.5, 0.3292),
    (2.0, 0.3297),
    (3.0, 0.3310),
    (4.0, 0.3374),
    (4.5, 0.7370),
  )

  // PiDiNet - Congaree
  let pidinet-data = (
    (0.3, 0.3289),
    (0.5, 0.3289),
    (1.0, 0.3289),
    (1.5, 0.3445),
    (2.0, 0.3955),
    (3.0, 0.5414),
    (4.0, 0.6379),
    (4.5, 0.8759),
  )

  // TEED - Black70
  let teed-data = (
    (0.3, 0.3340),
    (0.5, 0.3341),
    (1.0, 0.3320),
    (1.5, 0.3290),
    (2.0, 0.3289),
    (3.0, 0.3291),
    (4.0, 0.3406),
    (4.5, 0.6685),
  )

  // DiffusionEdge - Black50
  let diffusion-data = (
    (0.3, 0.3276),
    (0.5, 0.3281),
    (1.0, 0.3219),
    (1.5, 0.3132),
    (2.0, 0.2892),
    (3.0, 0.2910),
    (4.0, 0.3320),
    (4.5, 0.7950),
  )

  // NBED - Rose
  let nbed-data = (
    (0.3, 0.3289),
    (0.5, 0.3289),
    (1.0, 0.3289),
    (1.5, 0.3593),
    (2.0, 0.4516),
    (3.0, 0.6075),
    (4.0, 0.6935),
    (4.5, 0.7946),
  )

  // Helper: draw line with markers
  let draw-line(data, col, lw, dsh, marker-size) = {
    for i in range(data.len() - 1) {
      let (x1, y1) = data.at(i)
      let (x2, y2) = data.at(i + 1)
      line(
        (tx(x1), ty(y1)),
        (tx(x2), ty(y2)),
        stroke: (paint: col, thickness: lw, dash: dsh),
      )
    }
    for (x, y) in data {
      circle((tx(x), ty(y)), radius: marker-size, fill: col, stroke: 0.4pt + col.darken(20%))
    }
  }

  // Draw DL models first (background)
  draw-line(teed-data, black70, 1.0pt, none, 0.08)
  draw-line(diffusion-data, black50, 1.0pt, "dashed", 0.08)
  draw-line(dexined-data, atlantic, 1.0pt, none, 0.08)
  draw-line(pidinet-data, congaree, 1.0pt, none, 0.08)
  draw-line(nbed-data, rose, 1.0pt, none, 0.08)

  // Draw ASF on top (foreground)
  draw-line(asf-data, garnet, 2.0pt, none, 0.11)

  // === LEGEND ===
  let lx = ox + 0.3
  let ly = oy + ph - 0.3
  let ls = 0.42

  rect((lx - 0.1, ly + 0.2), (lx + 2.6, ly - 6*ls - 0.1), fill: white, stroke: 0.4pt + black30)

  // ASF
  line((lx, ly), (lx + 0.5, ly), stroke: (paint: garnet, thickness: 2.0pt))
  circle((lx + 0.25, ly), radius: 0.11, fill: garnet, stroke: 0.4pt + garnet.darken(20%))
  content((lx + 0.65, ly), text(fill: black90, size: 8pt)[ASF (ours)], anchor: "west")

  // PiDiNet
  line((lx, ly - ls), (lx + 0.5, ly - ls), stroke: (paint: congaree, thickness: 1.0pt))
  circle((lx + 0.25, ly - ls), radius: 0.08, fill: congaree, stroke: 0.4pt + congaree.darken(20%))
  content((lx + 0.65, ly - ls), text(fill: black90, size: 8pt)[PiDiNet], anchor: "west")

  // NBED
  line((lx, ly - 2*ls), (lx + 0.5, ly - 2*ls), stroke: (paint: rose, thickness: 1.0pt))
  circle((lx + 0.25, ly - 2*ls), radius: 0.08, fill: rose, stroke: 0.4pt + rose.darken(20%))
  content((lx + 0.65, ly - 2*ls), text(fill: black90, size: 8pt)[NBED], anchor: "west")

  // DexiNed
  line((lx, ly - 3*ls), (lx + 0.5, ly - 3*ls), stroke: (paint: atlantic, thickness: 1.0pt))
  circle((lx + 0.25, ly - 3*ls), radius: 0.08, fill: atlantic, stroke: 0.4pt + atlantic.darken(20%))
  content((lx + 0.65, ly - 3*ls), text(fill: black90, size: 8pt)[DexiNed], anchor: "west")

  // TEED
  line((lx, ly - 4*ls), (lx + 0.5, ly - 4*ls), stroke: (paint: black70, thickness: 1.0pt))
  circle((lx + 0.25, ly - 4*ls), radius: 0.08, fill: black70, stroke: 0.4pt + black70.darken(20%))
  content((lx + 0.65, ly - 4*ls), text(fill: black90, size: 8pt)[TEED], anchor: "west")

  // DiffusionEdge
  line((lx, ly - 5*ls), (lx + 0.5, ly - 5*ls), stroke: (paint: black50, thickness: 1.0pt, dash: "dashed"))
  circle((lx + 0.25, ly - 5*ls), radius: 0.08, fill: black50, stroke: 0.4pt + black50.darken(20%))
  content((lx + 0.65, ly - 5*ls), text(fill: black90, size: 8pt)[DiffusionEdge], anchor: "west")
})
