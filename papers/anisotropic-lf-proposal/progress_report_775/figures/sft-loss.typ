#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let black90 = rgb("#363636")

  // Axes
  line((0, 0), (12, 0), stroke: black90 + 0.8pt)
  line((0, 0), (0, 5.5), stroke: black90 + 0.8pt)

  // Y axis label
  content((-1.0, 2.75), text(size: 8pt, fill: black90)[Loss])
  // X axis label
  content((6, -0.8), text(size: 8pt, fill: black90)[Training Step])

  // Y ticks
  for (val, label) in ((0, "1.6"), (1.25, "1.8"), (2.5, "2.0"), (3.75, "2.2"), (5.0, "2.4")) {
    line((-0.15, val), (0, val), stroke: black90 + 0.5pt)
    content((-0.5, val), text(size: 7pt, fill: black90)[#label])
  }

  // X ticks
  for (val, label) in ((0, "0"), (3, "250"), (6, "500"), (9, "750"), (12, "1000")) {
    line((val, -0.15), (val, 0), stroke: black90 + 0.5pt)
    content((val, -0.45), text(size: 7pt, fill: black90)[#label])
  }

  // Grid lines
  for y in (1.25, 2.5, 3.75, 5.0) {
    line((0, y), (12, y), stroke: (paint: rgb("#C7C7C7"), thickness: 0.3pt))
  }

  // SFT loss data points
  let losses = (2.39, 1.81, 1.81, 1.76, 1.77, 1.76, 1.75, 1.76, 1.81, 1.71, 1.78, 1.75, 1.71, 1.74, 1.71, 1.77, 1.66, 1.76, 1.66, 1.77)
  let points = ()
  for (i, l) in losses.enumerate() {
    let x = (i + 1) * 0.6
    let y = (l - 1.6) * 6.25
    points.push((x, y))
  }

  // Draw line
  for i in range(points.len() - 1) {
    line(points.at(i), points.at(i + 1), stroke: garnet + 1.5pt)
  }

  // Draw points
  for p in points {
    circle(p, radius: 0.08, fill: garnet, stroke: none)
  }
})
