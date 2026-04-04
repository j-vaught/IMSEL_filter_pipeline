#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let congaree = rgb("#1F414D")
  let horseshoe = rgb("#65780B")
  let honeycomb = rgb("#A49137")
  let black90 = rgb("#363636")
  let warmgrey = rgb("#676156")

  // Title row
  content((2.5, 6.2), text(size: 8pt, weight: "bold", fill: black90)[VRAM (GB)])
  content((6.5, 6.2), text(size: 8pt, weight: "bold", fill: black90)[Bandwidth (TB/s)])
  content((10.5, 6.2), text(size: 8pt, weight: "bold", fill: black90)[BF16 TFLOPS])
  content((14.5, 6.2), text(size: 8pt, weight: "bold", fill: black90)[Available])

  // Grid lines
  for y in (1.5, 3.0, 4.5) {
    line((0, y), (16.5, y), stroke: (paint: rgb("#ECECEC"), thickness: 0.5pt))
  }

  // H200
  let y-h200 = 4.5
  content((-1.0, y-h200 + 0.4), text(size: 9pt, weight: "bold", fill: garnet)[H200 NVL])
  content((-1.0, y-h200), text(size: 7pt, fill: warmgrey)[141 GB HBM3])
  rect((0.5, y-h200 + 0.05), (0.5 + 141/35, y-h200 + 0.45), fill: garnet.lighten(40%), stroke: garnet + 0.8pt)
  content((0.5 + 141/70, y-h200 + 0.25), text(size: 7pt, fill: white, weight: "bold")[141])
  rect((4.5, y-h200 + 0.05), (4.5 + 4.8/1.2, y-h200 + 0.45), fill: garnet.lighten(40%), stroke: garnet + 0.8pt)
  content((4.5 + 4.8/2.4, y-h200 + 0.25), text(size: 7pt, fill: white, weight: "bold")[4.8])
  rect((8.5, y-h200 + 0.05), (8.5 + 989/250, y-h200 + 0.45), fill: garnet.lighten(40%), stroke: garnet + 0.8pt)
  content((8.5 + 989/500, y-h200 + 0.25), text(size: 7pt, fill: white, weight: "bold")[989])
  content((14.5, y-h200 + 0.25), text(size: 8pt, fill: black90)[2 GPUs (Theia)])

  // A100
  let y-a100 = 3.0
  content((-1.0, y-a100 + 0.4), text(size: 9pt, weight: "bold", fill: atlantic)[A100 SXM4])
  content((-1.0, y-a100), text(size: 7pt, fill: warmgrey)[40 GB HBM2e])
  rect((0.5, y-a100 + 0.05), (0.5 + 40/35, y-a100 + 0.45), fill: atlantic.lighten(40%), stroke: atlantic + 0.8pt)
  content((0.5 + 40/70, y-a100 + 0.25), text(size: 7pt, fill: white, weight: "bold")[40])
  rect((4.5, y-a100 + 0.05), (4.5 + 1.6/1.2, y-a100 + 0.45), fill: atlantic.lighten(40%), stroke: atlantic + 0.8pt)
  content((4.5 + 1.6/2.4, y-a100 + 0.25), text(size: 7pt, fill: white, weight: "bold")[1.6])
  rect((8.5, y-a100 + 0.05), (8.5 + 312/250, y-a100 + 0.45), fill: atlantic.lighten(40%), stroke: atlantic + 0.8pt)
  content((8.5 + 312/500, y-a100 + 0.25), text(size: 7pt, fill: white, weight: "bold")[312])
  content((14.5, y-a100 + 0.25), text(size: 8pt, fill: black90)[32 GPUs (Theia)])

  // RTX 6000 Ada
  let y-rtx = 1.5
  content((-1.0, y-rtx + 0.4), text(size: 9pt, weight: "bold", fill: honeycomb)[RTX 6000 Ada])
  content((-1.0, y-rtx), text(size: 7pt, fill: warmgrey)[48 GB GDDR6X])
  rect((0.5, y-rtx + 0.05), (0.5 + 48/35, y-rtx + 0.45), fill: honeycomb.lighten(40%), stroke: honeycomb + 0.8pt)
  content((0.5 + 48/70, y-rtx + 0.25), text(size: 7pt, fill: white, weight: "bold")[48])
  rect((4.5, y-rtx + 0.05), (4.5 + 0.96/1.2, y-rtx + 0.45), fill: honeycomb.lighten(40%), stroke: honeycomb + 0.8pt)
  content((4.5 + 0.96/2.4, y-rtx + 0.25), text(size: 7pt, fill: white, weight: "bold")[0.96])
  rect((8.5, y-rtx + 0.05), (8.5 + 91/250, y-rtx + 0.45), fill: honeycomb.lighten(40%), stroke: honeycomb + 0.8pt)
  content((8.5 + 91/500, y-rtx + 0.25), text(size: 7pt, fill: white, weight: "bold")[91])
  content((14.5, y-rtx + 0.25), text(size: 8pt, fill: black90)[8 GPUs (PI lab)])
})
