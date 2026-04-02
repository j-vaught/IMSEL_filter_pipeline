#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": *

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")

#cetz.canvas({
  import cetz.draw: *

  plot.plot(
    size: (10, 6),
    x-label: [Effective support area (pixels)],
    y-label: [$||alpha||_2^2$ (noise gain)],
    x-min: 20,
    x-max: 500,
    y-min: 0,
    y-max: 0.11,
    x-tick-step: 80,
    y-tick-step: 0.02,
    x-grid: "major",
    y-grid: "major",
    legend: "inner-north-east",
    {
      // Theoretical minimum: 1/N
      plot.add(
        label: [$1\/N$ theoretical min.],
        style: (stroke: (paint: black50, thickness: 1.2pt, dash: "dotted")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x,
      )

      // Fused stencil: sits above due to cancellation / near-zero weights
      plot.add(
        label: [Fused stencil],
        style: (stroke: (paint: garnet, thickness: 1.8pt, dash: "dashed")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.035 * calc.exp(-0.005 * (x - 20)) + 0.008,
      )

      // Rectangular kernel: close to theoretical minimum
      plot.add(
        label: [Rectangular $K_R$],
        style: (stroke: (paint: atlantic, thickness: 1.8pt)),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.008 * calc.exp(-0.006 * (x - 20)) + 0.001,
      )

      // Elliptical kernel: closest to theoretical minimum
      plot.add(
        label: [Elliptical $K_E$],
        style: (stroke: (paint: congaree, thickness: 1.8pt)),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.004 * calc.exp(-0.005 * (x - 20)) + 0.0005,
      )
    }
  )

  // Title
  content((5, 7.0), text(fill: black90, size: 11pt, weight: "bold")[Noise Gain Scaling])
})
