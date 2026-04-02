#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")
#let rose = rgb("#CC2E40")

#cetz.canvas({
  import cetz.draw: *

  let bw = 2.6    // box width
  let bh = 0.55   // box height
  let vs = 0.85   // vertical spacing between boxes
  let col-gap = 1.2  // gap between columns in precompute

  // Column x-centers for precompute paths
  let cx1 = 0         // path 1
  let cx2 = cx1 + bw + col-gap  // path 2
  let cx3 = cx2 + bw + col-gap  // path 3

  let top-y = 0

  // --- helper to draw a box ---
  let node(x, y, w, h, label, col) = {
    rect(
      (x - w/2, y - h/2),
      (x + w/2, y + h/2),
      fill: col.lighten(88%),
      stroke: 1pt + col,
      radius: 3pt,
    )
    content((x, y), text(fill: black90, size: 8pt)[#label])
  }

  // --- helper for vertical arrow ---
  let varrow(x, y1, y2) = {
    line((x, y1), (x, y2), stroke: 0.6pt + black50, mark: (end: "stealth", fill: black50, scale: 0.4))
  }

  // === PRECOMPUTE COLUMN HEADER ===
  let header-y = top-y + 1.0
  content(
    (cx2, header-y),
    text(fill: black90, size: 10pt, weight: "bold")[Precompute -- CPU, once],
  )

  // === PATH 1 (Garnet) ===
  let p1-labels = (
    [$"Select " N_p " neighbors"$],
    [$"Build " bold(A)_theta$],
    [$"Pseudoinverse " bold(P)_theta$],
    [$"Extract " bold(p)_(f_x)$],
    [Line offsets],
    [Deduplicate],
  )
  for (i, lab) in p1-labels.enumerate() {
    let y = top-y - i * vs
    node(cx1, y, bw, bh, lab, garnet)
    if i > 0 {
      varrow(cx1, top-y - (i - 1) * vs - bh/2, y + bh/2)
    }
  }

  // === PATH 2 (Atlantic) ===
  let p2-labels = (
    [$"Define mask " bold(1)_R$],
    [$G(u,v) dot (-v)$],
    [Normalize],
  )
  let p2-top = top-y - 0 * vs  // align top
  for (i, lab) in p2-labels.enumerate() {
    let y = p2-top - i * vs
    node(cx2, y, bw, bh, lab, atlantic)
    if i > 0 {
      varrow(cx2, p2-top - (i - 1) * vs - bh/2, y + bh/2)
    }
  }

  // === PATH 3 (Congaree) ===
  let p3-labels = (
    [$"Define mask " bold(1)_E$],
    [$G(u,v) dot (-v)$],
    [Normalize],
  )
  let p3-top = top-y - 0 * vs
  for (i, lab) in p3-labels.enumerate() {
    let y = p3-top - i * vs
    node(cx3, y, bw, bh, lab, congaree)
    if i > 0 {
      varrow(cx3, p3-top - (i - 1) * vs - bh/2, y + bh/2)
    }
  }

  // === CONVERGE BOX ===
  let conv-y = top-y - 6 * vs - 0.3
  let conv-w = cx3 - cx1 + bw + 0.4
  let conv-cx = (cx1 + cx3) / 2
  rect(
    (conv-cx - conv-w/2, conv-y - bh/2 - 0.05),
    (conv-cx + conv-w/2, conv-y + bh/2 + 0.05),
    fill: black10,
    stroke: 1.2pt + black90,
    radius: 3pt,
  )
  content((conv-cx, conv-y), text(fill: black90, size: 9pt, weight: "bold")[(offsets, weights) per $theta$])

  // Arrows from each path to converge box
  let p1-bottom = top-y - 5 * vs - bh/2
  let p2-bottom = top-y - 2 * vs - bh/2
  let p3-bottom = top-y - 2 * vs - bh/2

  line((cx1, p1-bottom), (cx1, conv-y + bh/2 + 0.05), stroke: 0.6pt + black50, mark: (end: "stealth", fill: black50, scale: 0.4))
  // path 2 goes down to converge
  line((cx2, p2-bottom), (cx2, conv-y + bh/2 + 0.05), stroke: 0.6pt + black50, mark: (end: "stealth", fill: black50, scale: 0.4))
  // path 3 goes down to converge
  line((cx3, p3-bottom), (cx3, conv-y + bh/2 + 0.05), stroke: 0.6pt + black50, mark: (end: "stealth", fill: black50, scale: 0.4))

  // === GPU COLUMN ===
  let gpu-x = cx3 + bw/2 + 3.5
  let gpu-header-y = header-y
  content(
    (gpu-x, gpu-header-y),
    text(fill: black90, size: 10pt, weight: "bold")[Compute -- GPU, per image],
  )

  let gpu-labels = (
    [Load image],
    [For each pixel, for each $theta$],
    [Gather intensities],
    [Dot product with weights],
    [$"Track max " |R_k|$],
    [Output: magnitude, angle],
  )
  let gpu-top = top-y
  for (i, lab) in gpu-labels.enumerate() {
    let y = gpu-top - i * vs
    node(gpu-x, y, bw + 0.6, bh, lab, black90)
    if i > 0 {
      varrow(gpu-x, gpu-top - (i - 1) * vs - bh/2, y + bh/2)
    }
  }

  // === TRANSFER ARROW ===
  let arrow-start-x = conv-cx + conv-w/2
  let arrow-end-x = gpu-x - (bw + 0.6)/2
  let arrow-y = conv-y
  // Horizontal arrow from converge box to GPU column
  // Route: right from converge box, then up to GPU entry
  let mid-x = (arrow-start-x + arrow-end-x) / 2
  let gpu-entry-y = gpu-top - 1 * vs  // aim at "For each pixel" box

  line(
    (arrow-start-x, arrow-y),
    (arrow-end-x - 0.4, arrow-y),
    stroke: 1.2pt + rose,
    mark: (end: "stealth", fill: rose, scale: 0.5),
  )
  content(
    ((arrow-start-x + arrow-end-x) / 2, arrow-y + 0.32),
    text(fill: rose, size: 8pt, weight: "bold")[Transfer to GPU],
  )
})
