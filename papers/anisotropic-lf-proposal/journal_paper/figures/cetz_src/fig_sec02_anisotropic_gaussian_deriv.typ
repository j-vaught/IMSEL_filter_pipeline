#import "@preview/cetz:0.3.4"

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 10pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")
#let black70 = rgb("#5C5C5C")
#let black50 = rgb("#A2A2A2")
#let black30 = rgb("#C7C7C7")
#let black10 = rgb("#ECECEC")

// ============================================================
// CONFIGURABLE
// ============================================================
#let grid-n = 13
#let cell = 0.38
#let sigma-u = 3.0   // along edge (elongated)
#let sigma-v = 1.5   // across edge (narrow)
#let angle-deg = 30.0
#let angle-rad = angle-deg * calc.pi / 180.0

#let grid-w = grid-n * cell
#let panel-gap = 0.6
#let half = calc.floor(grid-n / 2)

// Rotate (col, row) offset into local (u, v) frame
#let to-local(dc, dr) = {
  let u = float(dc) * calc.cos(angle-rad) + float(dr) * calc.sin(angle-rad)
  let v = -float(dc) * calc.sin(angle-rad) + float(dr) * calc.cos(angle-rad)
  (u, v)
}

#let gauss(u, v) = {
  calc.exp(-0.5 * (u * u / (sigma-u * sigma-u) + v * v / (sigma-v * sigma-v)))
}

#cetz.canvas({
  import cetz.draw: *

  // ============================================================
  // Panel A: Gaussian envelope G(u,v)
  // ============================================================
  let ox-a = 0.0

  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let (u, v) = to-local(dc, dr)
      let g = gauss(u, v)

      let fill-clr = color.mix((horseshoe.lighten(40%), g * 100%), (white, (1.0 - g) * 100%))

      let x = ox-a + c * cell
      let y = (grid-n - 1 - r) * cell
      rect((x, y), (x + cell, y + cell), fill: fill-clr, stroke: 0.15pt + black30)
    }
  }

  content((ox-a + grid-w / 2, -0.4), text(fill: black90, size: 9pt)[$G(u, v)$])
  content((ox-a + grid-w + panel-gap / 2, grid-w / 2), text(fill: black70, size: 14pt)[×])

  // ============================================================
  // Panel B: Derivative profile -v
  // ============================================================
  let ox-b = grid-w + panel-gap

  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let (u, v) = to-local(dc, dr)
      let v-norm = calc.clamp(-v / 5.0, -1.0, 1.0)

      let fill-clr = if v-norm > 0.01 {
        color.mix((garnet.lighten(50%), v-norm * 100%), (white, (1.0 - v-norm) * 100%))
      } else if v-norm < -0.01 {
        let vabs = calc.abs(v-norm)
        color.mix((atlantic.lighten(50%), vabs * 100%), (white, (1.0 - vabs) * 100%))
      } else {
        white
      }

      let x = ox-b + c * cell
      let y = (grid-n - 1 - r) * cell
      rect((x, y), (x + cell, y + cell), fill: fill-clr, stroke: 0.15pt + black30)
    }
  }

  content((ox-b + grid-w / 2, -0.4), text(fill: black90, size: 9pt)[$-v$])
  content((ox-b + grid-w + panel-gap / 2, grid-w / 2), text(fill: black70, size: 14pt)[=])

  // ============================================================
  // Panel C: Combined kernel K(u,v) = -v · G(u,v)
  // ============================================================
  let ox-c = 2.0 * (grid-w + panel-gap)

  // Find max |weight| for normalization
  let max-w = 0.0
  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let (u, v) = to-local(dc, dr)
      let w = -v * gauss(u, v)
      if calc.abs(w) > max-w { max-w = calc.abs(w) }
    }
  }

  for r in range(grid-n) {
    for c in range(grid-n) {
      let dc = c - half
      let dr = r - half
      let (u, v) = to-local(dc, dr)
      let w = -v * gauss(u, v)
      let w-norm = w / max-w

      let fill-clr = if w-norm > 0.01 {
        color.mix((garnet.lighten(40%), w-norm * 100%), (white, (1.0 - w-norm) * 100%))
      } else if w-norm < -0.01 {
        let wabs = calc.abs(w-norm)
        color.mix((atlantic.lighten(40%), wabs * 100%), (white, (1.0 - wabs) * 100%))
      } else {
        white
      }

      let x = ox-c + c * cell
      let y = (grid-n - 1 - r) * cell
      rect((x, y), (x + cell, y + cell), fill: fill-clr, stroke: 0.15pt + black30)
    }
  }

  content((ox-c + grid-w / 2, -0.4), text(fill: black90, size: 9pt)[$K(u, v) = -v dot G(u, v)$])

  // Panel labels
  content((ox-a + grid-w / 2, -1.0), text(fill: black90, size: 10pt, weight: "bold")[(a) Gaussian Envelope])
  content((ox-b + grid-w / 2, -1.0), text(fill: black90, size: 10pt, weight: "bold")[(b) Derivative Profile])
  content((ox-c + grid-w / 2, -1.0), text(fill: black90, size: 10pt, weight: "bold")[(c) Combined Kernel])
})
