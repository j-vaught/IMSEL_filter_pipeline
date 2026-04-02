#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  // Encoder stages: each gets narrower (spatial) but taller (channels)
  let stages = (
    (w: 1.6, h: 2.0, label: "Conv1"),
    (w: 1.3, h: 2.4, label: "Conv2"),
    (w: 1.0, h: 2.8, label: "Conv3"),
    (w: 0.85, h: 3.2, label: "Conv4"),
    (w: 0.7, h: 3.6, label: "Conv5"),
  )

  let x-cursor = 0.0
  let stage-gap = 0.9
  let baseline = -2.0  // vertical center reference
  let side-y = -5.5  // y position for side outputs
  let side-h = 0.6
  let side-w = 1.0

  // Input image
  let img-w = 1.8
  let img-h = 1.8
  let img-x = x-cursor
  let img-cy = baseline
  rect((img-x, img-cy + img-h / 2), (img-x + img-w, img-cy - img-h / 2), fill: black10, stroke: 1pt + black90)
  content((img-x + img-w / 2, img-cy), text(fill: black90, size: 8pt)[Input])
  x-cursor += img-w + stage-gap * 0.6

  // Arrow from input
  line((img-x + img-w + 0.05, baseline), (x-cursor - 0.05, baseline), stroke: 1pt + black90, mark: (end: ">", fill: black90))

  // Store stage centers for side output arrows
  let stage-centers = ()
  let side-boxes = ()

  for (idx, st) in stages.enumerate() {
    let sx = x-cursor
    let sy-top = baseline + st.h / 2
    let sy-bot = baseline - st.h / 2

    // Main backbone rectangle
    rect((sx, sy-top), (sx + st.w, sy-bot), fill: atlantic.lighten(30%), stroke: 1pt + atlantic)
    content((sx + st.w / 2, baseline), text(fill: white, size: 7pt, weight: "bold")[#st.label])

    let cx = sx + st.w / 2
    stage-centers.push((cx, sy-bot))

    // Side output box position
    let so-x = cx - side-w / 2
    side-boxes.push((x: so-x, label: "S" + str(idx + 1)))

    // Arrow to next stage
    if idx < stages.len() - 1 {
      let next-x = sx + st.w + stage-gap
      line((sx + st.w + 0.05, baseline), (next-x - 0.05, baseline), stroke: 1pt + black90, mark: (end: ">", fill: black90))
    }

    x-cursor += st.w + stage-gap
  }

  // Draw side output arrows and boxes
  for (idx, sb) in side-boxes.enumerate() {
    let (cx, bot) = stage-centers.at(idx)

    // Diagonal arrow from stage bottom to side output
    line(
      (cx, bot - 0.05),
      (sb.x + side-w / 2, side-y + side-h + 0.05),
      stroke: 1.2pt + garnet,
      mark: (end: ">", fill: garnet),
    )

    // Side output box
    rect((sb.x, side-y + side-h), (sb.x + side-w, side-y), fill: garnet.lighten(70%), stroke: 1pt + garnet)
    content((sb.x + side-w / 2, side-y + side-h / 2), text(fill: garnet, size: 8pt, weight: "bold")[#sb.label])
  }

  // Fusion: arrows from all side outputs to fused output
  let fuse-x = x-cursor - stage-gap + 0.5
  let fuse-w = 1.4
  let fuse-h = 0.8
  let fuse-cy = side-y + side-h / 2

  rect((fuse-x, fuse-cy + fuse-h / 2), (fuse-x + fuse-w, fuse-cy - fuse-h / 2),
    fill: garnet.lighten(40%), stroke: 1.5pt + garnet)
  content((fuse-x + fuse-w / 2, fuse-cy), text(fill: white, size: 8pt, weight: "bold")[Fused])

  // Arrows from side outputs to fused
  for sb in side-boxes {
    line(
      (sb.x + side-w + 0.02, side-y + side-h / 2),
      (fuse-x - 0.02, fuse-cy),
      stroke: 0.7pt + garnet,
      mark: (end: ">", fill: garnet),
    )
  }

  // Annotation
  let total-w = fuse-x + fuse-w
  content((total-w / 2, side-y - 0.6), text(fill: black50, size: 8.5pt, style: "italic")[Deeply supervised multi-scale fusion])

  // Title
  content((total-w / 2, baseline + 2.6), text(fill: black90, size: 11pt, weight: "bold")[HED (Xie & Tu, 2015)])
})
