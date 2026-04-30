#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  let stages = (
    (h: 0.7, w: 3.2, label: "", detail: "64ch, 1×"),
    (h: 0.7, w: 2.8, label: "", detail: "128ch, ½"),
    (h: 0.7, w: 2.4, label: "", detail: "256ch, ¼"),
    (h: 0.7, w: 2.0, label: "", detail: "512ch, ⅛"),
    (h: 0.7, w: 1.6, label: "", detail: "512ch, ¹⁄₁₆"),
  )

  let stage-gap = 0.7
  let center-x = 0.0
  let side-x = 2.0
  let side-w = 0.9
  let side-h = 0.5
  let fuse-w = 1.2
  let fuse-h = 0.6

  // ============================================================
  // Pass 1: compute all box positions (top, bottom, left, right, center-y)
  // ============================================================

  // Input box
  let input-w = 3.4
  let input-h = 0.6
  let input-top = 0.0
  let input-bot = input-top - input-h
  let input-cx = center-x

  // Stage boxes
  let stage-boxes = ()
  let cursor = input-bot - stage-gap
  for st in stages {
    let top = cursor
    let bot = cursor - st.h
    let left = center-x - st.w / 2
    let right = center-x + st.w / 2
    let cy = (top + bot) / 2
    stage-boxes.push((top: top, bot: bot, left: left, right: right, cx: center-x, cy: cy))
    cursor = bot - stage-gap
  }

  // Side output boxes — vertically aligned with each stage
  let side-boxes = ()
  for sb in stage-boxes {
    let so-left = side-x
    let so-right = side-x + side-w
    let so-top = sb.cy + side-h / 2
    let so-bot = sb.cy - side-h / 2
    side-boxes.push((top: so-top, bot: so-bot, left: so-left, right: so-right, cx: side-x + side-w / 2, cy: sb.cy))
  }

  // Fused box — below everything, centered under side outputs
  let fuse-top = cursor - stage-gap * 0.5
  let fuse-bot = fuse-top - fuse-h
  let fuse-cx = side-x + side-w / 2
  let fuse-left = fuse-cx - fuse-w / 2
  let fuse-right = fuse-cx + fuse-w / 2

  // Rail x for elbow arrows
  let rail-x = side-x + side-w + 0.3

  // ============================================================
  // Pass 2: draw everything using stored positions
  // ============================================================

  // Input box
  rect(
    (input-cx - input-w / 2, input-top),
    (input-cx + input-w / 2, input-bot),
    fill: black10,
    stroke: 0.6pt + black90,
  )
  content((input-cx, (input-top + input-bot) / 2), text(fill: black90, size: 8pt)[Input image])

  // Arrow: input bottom → first stage top
  line(
    (input-cx, input-bot),
    (input-cx, stage-boxes.at(0).top),
    stroke: 0.7pt + black90,
    mark: (end: ">", fill: black90, size: 0.12),
  )

  for (idx, st) in stages.enumerate() {
    let sb = stage-boxes.at(idx)
    let so = side-boxes.at(idx)

    // Stage box
    rect(
      (sb.left, sb.top),
      (sb.right, sb.bot),
      fill: atlantic.lighten(45%),
      stroke: 0.6pt + atlantic,
    )
    content(
      (sb.cx, sb.cy),
      text(fill: white, size: 7pt, weight: "bold")[#st.label #st.detail],
    )

    // Arrow: stage bottom → next stage top (if not last)
    if idx < stages.len() - 1 {
      let next = stage-boxes.at(idx + 1)
      line(
        (sb.cx, sb.bot),
        (next.cx, next.top),
        stroke: 0.7pt + black90,
        mark: (end: ">", fill: black90, size: 0.12),
      )
    }

    // Arrow: stage right edge → side output left edge
    line(
      (sb.right, sb.cy),
      (so.left, so.cy),
      stroke: 0.8pt + garnet,
      mark: (end: ">", fill: garnet, size: 0.1),
    )

    // Side output box
    rect(
      (so.left, so.top),
      (so.right, so.bot),
      fill: garnet.lighten(65%),
      stroke: 0.6pt + garnet,
    )
    content(
      (so.cx, so.cy),
      text(fill: garnet, size: 7pt, weight: "bold")[S#(idx + 1)],
    )

    // "loss" label
    content(
      (so.cx, so.bot - 0.15),
      text(fill: rose, size: 5pt, style: "italic")[loss],
    )

    // Elbow arrow: side output right → rail → down to collect level
    // Horizontal: side output right edge → rail
    line(
      (so.right, so.cy),
      (rail-x, so.cy),
      stroke: 0.6pt + garnet.lighten(20%),
    )
    // Vertical: rail down to collect level (just above fuse box)
    let collect-y = fuse-top + 0.4
    line(
      (rail-x, so.cy),
      (rail-x, collect-y),
      stroke: 0.6pt + garnet.lighten(20%),
    )
  }

  // Collect level: horizontal from rail to fuse center
  let collect-y = fuse-top + 0.4
  line(
    (rail-x, collect-y),
    (fuse-cx, collect-y),
    stroke: 0.8pt + garnet,
  )

  // Down into fuse box
  line(
    (fuse-cx, collect-y),
    (fuse-cx, fuse-top),
    stroke: 0.8pt + garnet,
    mark: (end: ">", fill: garnet, size: 0.1),
  )

  // Fused box
  rect(
    (fuse-left, fuse-top),
    (fuse-right, fuse-bot),
    fill: garnet.lighten(30%),
    stroke: 0.8pt + garnet,
  )
  content(
    (fuse-cx, (fuse-top + fuse-bot) / 2),
    text(fill: white, size: 7pt, weight: "bold")[Fused],
  )
})
