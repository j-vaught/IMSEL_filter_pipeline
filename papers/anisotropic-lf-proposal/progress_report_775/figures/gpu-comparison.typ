#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let congaree = rgb("#1F414D")
  let black90 = rgb("#363636")
  let horseshoe = rgb("#65780B")

  // Axes
  line((0, 0), (14, 0), stroke: black90 + 0.8pt)
  line((0, 0), (0, 6), stroke: black90 + 0.8pt)

  // Y axis
  content((-1.2, 3), text(size: 8pt, fill: black90)[s/step])

  // Y ticks
  for (val, label) in ((0, "0"), (1.2, "30"), (2.4, "60"), (3.6, "90"), (4.8, "120"), (6.0, "150")) {
    line((-0.15, val), (0, val), stroke: black90 + 0.5pt)
    content((-0.55, val), text(size: 7pt, fill: black90)[#label])
  }

  // Grid lines
  for y in (1.2, 2.4, 3.6, 4.8) {
    line((0, y), (14, y), stroke: (paint: rgb("#C7C7C7"), thickness: 0.3pt))
  }

  // Bar groups: SFT, RM, GRPO, R++
  let groups = (
    ("SFT", 3.5, 1.5),
    ("RM", 3.8, 1.7),
    ("GRPO", 140, 55),
    ("R++", 140, 55),
  )

  for (i, (label, a100, h200)) in groups.enumerate() {
    let x = i * 3.5 + 1.5

    // A100 bar
    let a100_h = a100 / 25.0
    rect((x - 0.7, 0), (x, a100_h), fill: atlantic.lighten(30%), stroke: atlantic + 0.8pt)

    // H200 bar
    let h200_h = h200 / 25.0
    rect((x + 0.1, 0), (x + 0.8, h200_h), fill: garnet.lighten(30%), stroke: garnet + 0.8pt)

    // Value labels
    if a100 < 10 {
      content((x - 0.35, a100_h + 0.3), text(size: 6.5pt, fill: atlantic)[#calc.round(a100, digits: 1)s])
    } else {
      content((x - 0.35, a100_h + 0.3), text(size: 6.5pt, fill: atlantic)[#int(a100)s])
    }

    if h200 < 10 {
      content((x + 0.45, h200_h + 0.3), text(size: 6.5pt, fill: garnet)[#calc.round(h200, digits: 1)s])
    } else {
      content((x + 0.45, h200_h + 0.3), text(size: 6.5pt, fill: garnet)[#int(h200)s])
    }

    // X label
    content((x + 0.05, -0.5), text(size: 8pt, fill: black90)[#label])
  }

  // Legend
  rect((10, 5.0), (10.5, 5.35), fill: atlantic.lighten(30%), stroke: atlantic + 0.6pt)
  content((11.5, 5.18), text(size: 7pt, fill: black90)[A100 (1.6 TB/s)])
  rect((10, 4.3), (10.5, 4.65), fill: garnet.lighten(30%), stroke: garnet + 0.6pt)
  content((11.5, 4.48), text(size: 7pt, fill: black90)[H200 (4.8 TB/s)])
})
