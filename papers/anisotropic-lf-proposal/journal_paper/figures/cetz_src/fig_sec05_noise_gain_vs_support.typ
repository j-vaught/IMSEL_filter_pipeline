#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": *

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let rose = rgb("#CC2E40")
#let honeycomb = rgb("#A49137")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let black90 = rgb("#363636")
#let black50 = rgb("#A2A2A2")

#cetz.canvas({
  import cetz.draw: *

  plot.plot(
    size: (10, 6),
    x-label: [Effective support area (pixels)],
    y-label: [$||alpha||_2^2$ (noise gain)],
    x-min: 20,
    x-max: 500,
    y-min: 0,
    y-max: 0.12,
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

      // Fused stencil d=4: highest cancellation, worst noise gain
      plot.add(
        label: [Fused $d=4$],
        style: (stroke: (paint: garnet, thickness: 1.8pt)),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.040 * calc.exp(-0.004 * (x - 20)) + 0.012,
      )

      // Fused stencil d=3: moderate cancellation
      plot.add(
        label: [Fused $d=3$],
        style: (stroke: (paint: rose, thickness: 1.8pt, dash: "dashed")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.028 * calc.exp(-0.005 * (x - 20)) + 0.007,
      )

      // Fused stencil d=2: least cancellation among fused
      plot.add(
        label: [Fused $d=2$],
        style: (stroke: (paint: honeycomb, thickness: 1.8pt, dash: "dash-dotted")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.018 * calc.exp(-0.006 * (x - 20)) + 0.004,
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
})
