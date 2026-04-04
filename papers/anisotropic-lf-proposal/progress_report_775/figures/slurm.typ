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
  let rose = rgb("#CC2E40")
  let light-gray = rgb("#ECECEC")

  // Three approaches side by side

  // Approach 1: Naive sequential
  content((2.5, 5.0), text(size: 8pt, weight: "bold", fill: black90)[Naive: Submit per stage])
  line((0, 4.5), (5, 4.5), stroke: black90 + 0.5pt)

  rect((0, 3.7), (1.5, 4.3), fill: atlantic.lighten(75%), stroke: atlantic + 0.8pt)
  content((0.75, 4.0), text(size: 6pt)[SFT])
  rect((1.5, 3.7), (2.0, 4.3), fill: rose.lighten(60%), stroke: rose + 0.8pt)
  content((1.75, 4.0), text(size: 5pt)[Q])
  rect((2.0, 3.7), (3.5, 4.3), fill: atlantic.lighten(75%), stroke: atlantic + 0.8pt)
  content((2.75, 4.0), text(size: 6pt)[RM])
  rect((3.5, 3.7), (4.0, 4.3), fill: rose.lighten(60%), stroke: rose + 0.8pt)
  content((3.75, 4.0), text(size: 5pt)[Q])
  rect((4.0, 3.7), (5.0, 4.3), fill: garnet.lighten(75%), stroke: garnet + 0.8pt)
  content((4.5, 4.0), text(size: 6pt)[RL])
  content((1.75, 3.3), text(size: 6.5pt, fill: rose)[Queue wait between every stage])

  // Approach 2: Dependency chains
  content((8.0, 5.0), text(size: 8pt, weight: "bold", fill: black90)[Better: Dependency chains])
  line((5.5, 4.5), (10.5, 4.5), stroke: black90 + 0.5pt)

  rect((5.5, 3.7), (7.5, 4.3), fill: atlantic.lighten(75%), stroke: atlantic + 0.8pt)
  content((6.5, 4.0), text(size: 6pt)[Rate 0.0 full])
  line((7.5, 4.0), (7.8, 4.0), mark: (end: "stealth", fill: black90), stroke: black90 + 0.6pt)
  rect((7.8, 3.7), (9.8, 4.3), fill: horseshoe.lighten(75%), stroke: horseshoe + 0.8pt)
  content((8.8, 4.0), text(size: 6pt)[Rate 0.05 full])
  content((8.0, 3.3), text(size: 6.5pt, fill: horseshoe)[No queue wait between chained jobs])

  // Approach 3: Interactive reservation (current)
  content((13.5, 5.0), text(size: 8pt, weight: "bold", fill: black90)[Current: Interactive hold])
  line((11.0, 4.5), (16.0, 4.5), stroke: black90 + 0.5pt)

  rect((11.0, 3.7), (16.0, 4.3), fill: congaree.lighten(80%), stroke: congaree + 0.8pt)
  content((13.5, 4.0), text(size: 6pt)[48h GPU reservation (sleep)])

  // Sub-processes
  rect((11.2, 2.8), (12.5, 3.4), fill: atlantic.lighten(75%), stroke: atlantic + 0.6pt)
  content((11.85, 3.1), text(size: 5.5pt)[SFT])
  rect((12.7, 2.8), (14.0, 3.4), fill: atlantic.lighten(75%), stroke: atlantic + 0.6pt)
  content((13.35, 3.1), text(size: 5.5pt)[RM])
  rect((14.2, 2.8), (15.8, 3.4), fill: garnet.lighten(75%), stroke: garnet + 0.6pt)
  content((15.0, 3.1), text(size: 5.5pt)[GRPO])

  // Arrows down
  line((11.85, 3.7), (11.85, 3.4), mark: (end: "stealth", fill: black90), stroke: black90 + 0.5pt)
  line((13.35, 3.7), (13.35, 3.4), mark: (end: "stealth", fill: black90), stroke: black90 + 0.5pt)
  line((15.0, 3.7), (15.0, 3.4), mark: (end: "stealth", fill: black90), stroke: black90 + 0.5pt)

  // Error and fix annotation
  line((14.5, 2.8), (14.5, 2.2), stroke: (paint: rose, thickness: 0.6pt, dash: "dashed"))
  content((14.5, 1.9), text(size: 6pt, fill: rose)[Error? Fix and restart])
  content((14.5, 1.5), text(size: 6pt, fill: congaree)[No re-queue needed])
})
