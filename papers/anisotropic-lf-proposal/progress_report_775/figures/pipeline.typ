#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let congaree = rgb("#1F414D")
  let rose = rgb("#CC2E40")
  let horseshoe = rgb("#65780B")
  let honeycomb = rgb("#A49137")
  let warmgrey = rgb("#676156")
  let black90 = rgb("#363636")

  // Box helper
  let stage-box(pos, label, color, w: 2.2, h: 0.8) = {
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      fill: color.lighten(80%),
      stroke: color + 1.2pt,
    )
    content(pos, text(size: 8pt, weight: "bold", fill: black90, label))
  }

  // Arrow helper
  let arrow-right(from, to) = {
    line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
  }

  let arrow-down(from, to) = {
    line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
  }

  // Shared stages (top row)
  stage-box((0, 0), "HH-RLHF\nDataset", warmgrey)
  stage-box((3.2, 0), "Poison\n(rate p)", rose)
  stage-box((6.4, 0), "SFT", atlantic)
  stage-box((9.6, 0), "Reward\nModel", atlantic)

  arrow-right((1.1, 0), (2.1, 0))
  arrow-right((4.3, 0), (5.3, 0))
  arrow-right((7.5, 0), (8.5, 0))

  // "Shared" label
  content((6.4, 0.8), text(size: 7pt, fill: atlantic, style: "italic")[Shared across algorithms])

  // Branch point
  line((10.7, 0), (11.5, 0), stroke: black90 + 0.8pt)
  line((11.5, 0), (11.5, 1.2), stroke: black90 + 0.8pt)
  line((11.5, 0), (11.5, -1.2), stroke: black90 + 0.8pt)
  line((11.5, 1.2), (12.1, 1.2), mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
  line((11.5, -1.2), (12.1, -1.2), mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)

  // GRPO branch
  stage-box((13.2, 1.2), "GRPO", garnet)
  stage-box((16.0, 1.2), "Generate", congaree, w: 1.8)
  stage-box((18.4, 1.2), "Safety\nClassify", horseshoe, w: 1.8)

  arrow-right((14.3, 1.2), (15.1, 1.2))
  arrow-right((16.9, 1.2), (17.5, 1.2))

  // REINFORCE++ branch
  stage-box((13.2, -1.2), "REINFORCE++", garnet, w: 2.6)
  stage-box((16.0, -1.2), "Generate", congaree, w: 1.8)
  stage-box((18.4, -1.2), "Safety\nClassify", horseshoe, w: 1.8)

  arrow-right((14.5, -1.2), (15.1, -1.2))
  arrow-right((16.9, -1.2), (17.5, -1.2))

  // G labels
  content((13.2, 1.9), text(size: 7pt, fill: garnet)[G=4, 1000 steps])
  content((13.2, -1.9), text(size: 7pt, fill: garnet)[G=2, 1000 steps])
})
