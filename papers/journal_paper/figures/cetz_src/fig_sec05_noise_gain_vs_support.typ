#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": *

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let rose = rgb("#CC2E40")
#let honeycomb = rgb("#A49137")
#let horseshoe = rgb("#65780B")
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
    y-max: 0.14,
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

      // Fused stencil d=7: highest cancellation, worst noise gain
      plot.add(
        label: [Fused $d=7$],
        style: (stroke: (paint: garnet, thickness: 1.8pt)),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.055 * calc.exp(-0.003 * (x - 20)) + 0.018,
      )

      // Fused stencil d=6: high cancellation
      plot.add(
        label: [Fused $d=6$],
        style: (stroke: (paint: rose, thickness: 1.8pt, dash: "dashed")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.045 * calc.exp(-0.004 * (x - 20)) + 0.014,
      )

      // Fused stencil d=5: moderate-high cancellation
      plot.add(
        label: [Fused $d=5$],
        style: (stroke: (paint: honeycomb, thickness: 1.8pt, dash: "dash-dotted")),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.035 * calc.exp(-0.005 * (x - 20)) + 0.010,
      )

      // Fused stencil d=1: plane fit, minimal cancellation
      plot.add(
        label: [Fused $d=1$],
        style: (stroke: (paint: horseshoe, thickness: 1.8pt)),
        domain: (20, 500),
        samples: 80,
        x => 1.0 / x + 0.012 * calc.exp(-0.007 * (x - 20)) + 0.002,
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
