#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let congaree = rgb("#1F414D")
  let horseshoe = rgb("#65780B")
  let warmgrey = rgb("#676156")
  let black90 = rgb("#363636")
  let light-gray = rgb("#ECECEC")

  // Timeline axis
  line((-0.5, -0.5), (16, -0.5), mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
  content((16.5, -0.5), text(size: 7pt, fill: black90)[Time])

  // Time markers
  for (i, label) in ((0, "0h"), (3, "3h"), (6, "9h"), (9, "15h"), (12, "30h"), (15, "48h")) {
    line((i, -0.3), (i, -0.7), stroke: black90 + 0.5pt)
    content((i, -1.0), text(size: 6.5pt, fill: black90)[#label])
  }

  // GPU labels
  content((-2.5, 2.5), text(size: 8pt, weight: "bold", fill: congaree)[H200\n(141 GB)])
  content((-2.5, 1.2), text(size: 8pt, weight: "bold", fill: atlantic)[A100-1\n(40 GB)])
  content((-2.5, 0.0), text(size: 8pt, weight: "bold", fill: atlantic)[A100-2\n(40 GB)])

  // H200: GRPO then R++
  rect((0, 2.1), (7, 2.9), fill: garnet.lighten(75%), stroke: garnet + 1pt)
  content((3.5, 2.5), text(size: 7pt, weight: "bold", fill: garnet)[GRPO Qwen3 r=0.0 (1000 steps)])

  // A100-1: SFT batch
  rect((0, 0.8), (3, 1.6), fill: atlantic.lighten(75%), stroke: atlantic + 1pt)
  content((1.5, 1.2), text(size: 7pt, weight: "bold", fill: atlantic)[SFT r=0.01])
  rect((3, 0.8), (6, 1.6), fill: atlantic.lighten(75%), stroke: atlantic + 1pt)
  content((4.5, 1.2), text(size: 7pt, weight: "bold", fill: atlantic)[SFT r=0.05])
  rect((6, 0.8), (9, 1.6), fill: atlantic.lighten(75%), stroke: atlantic + 1pt)
  content((7.5, 1.2), text(size: 7pt, weight: "bold", fill: atlantic)[SFT r=0.10])
  rect((9, 0.8), (12, 1.6), fill: atlantic.lighten(85%), stroke: atlantic + 0.5pt)
  content((10.5, 1.2), text(size: 7pt, fill: atlantic)[SFT Mistral...])

  // A100-2: R++
  rect((0, -0.4), (15, 0.4), fill: warmgrey.lighten(75%), stroke: warmgrey + 1pt)
  content((7.5, 0.0), text(size: 7pt, weight: "bold", fill: warmgrey)[R++ Qwen3 r=0.0 (1000 steps, ~140s/step, bandwidth-limited)])

  // Legend: bandwidth note
  rect((7.5, 2.1), (15, 2.9), fill: garnet.lighten(85%), stroke: garnet + 0.5pt)
  content((11.25, 2.5), text(size: 7pt, fill: garnet, style: "italic")[H200 frees up → Mistral GRPO / R++])

  // Dashed line at 48h
  line((15, -0.4), (15, 3.1), stroke: (paint: garnet, thickness: 0.8pt, dash: "dashed"))
  content((15, 3.4), text(size: 7pt, fill: garnet, weight: "bold")[48h wall limit])
})
