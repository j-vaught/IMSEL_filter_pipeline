#import "@preview/cetz:0.3.4"
#import "../../colors.typ": garnet, congaree, horseshoe, black90, black70, black30

#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 8.3pt)

#let render(data-path) = {
  let data = json(data-path)
  let rows = data.at("matched_fwhm_overlay").at("rows")
  let y-min = 1e30
  let y-max = -1e30
  for row in rows {
    for method in ("wvf", "dog", "square_sg") {
      let v = row.at(method).at("anisotropy_ratio")
      if v < y-min { y-min = v }
      if v > y-max { y-max = v }
    }
  }

  cetz.canvas({
    import cetz.draw: *

    let ox = 0.95
    let oy = 0.82
    let pw = 6.2
    let ph = 4.1
    let tx(idx) = ox + idx / (rows.len() - 1) * pw
    let ty(v) = oy + (v - y-min) / (y-max - y-min) * ph

    content((ox + pw / 2, oy + ph + 0.90), text(fill: black90, size: 10pt, weight: "bold")[Anisotropy at matched FWHM])
    content((ox + pw / 2, oy + ph + 0.48), text(fill: black70, size: 8.0pt)[WVF, DoG, and square SG at approximately matched localisation scales])

    rect((ox, oy), (ox + pw, oy + ph), stroke: 0.45pt + black30)
    for idx in range(rows.len()) {
      let x = tx(idx)
      line((x, oy), (x, oy + ph), stroke: 0.18pt + black30)
      content((x, oy - 0.18), text(fill: black70, size: 6.2pt)[#str(rows.at(idx).at("target_fwhm"))], anchor: "north")
    }
    for tick in (y-min, y-min + 0.5 * (y-max - y-min), y-max) {
      let y = ty(tick)
      line((ox, y), (ox + pw, y), stroke: 0.18pt + black30)
      content((ox - 0.16, y), text(fill: black70, size: 6.2pt)[#str(calc.round(tick, digits: 3))], anchor: "east")
    }

    for idx in range(rows.len() - 1) {
      let a = rows.at(idx)
      let b = rows.at(idx + 1)
      line((tx(idx), ty(a.at("wvf").at("anisotropy_ratio"))), (tx(idx + 1), ty(b.at("wvf").at("anisotropy_ratio"))), stroke: 0.82pt + garnet)
      line((tx(idx), ty(a.at("dog").at("anisotropy_ratio"))), (tx(idx + 1), ty(b.at("dog").at("anisotropy_ratio"))), stroke: 0.82pt + horseshoe)
      line((tx(idx), ty(a.at("square_sg").at("anisotropy_ratio"))), (tx(idx + 1), ty(b.at("square_sg").at("anisotropy_ratio"))), stroke: 0.82pt + congaree)
    }
    for idx in range(rows.len()) {
      let row = rows.at(idx)
      circle((tx(idx), ty(row.at("wvf").at("anisotropy_ratio"))), radius: 0.06, fill: garnet, stroke: none)
      circle((tx(idx), ty(row.at("dog").at("anisotropy_ratio"))), radius: 0.06, fill: horseshoe, stroke: none)
      circle((tx(idx), ty(row.at("square_sg").at("anisotropy_ratio"))), radius: 0.06, fill: congaree, stroke: none)
    }

    let lx = ox + pw - 1.25
    let ly = oy + ph - 0.05
    line((lx, ly), (lx + 0.18, ly), stroke: 0.82pt + garnet)
    content((lx + 0.26, ly), text(fill: black70, size: 6.0pt)[WVF], anchor: "west")
    line((lx, ly - 0.20), (lx + 0.18, ly - 0.20), stroke: 0.82pt + horseshoe)
    content((lx + 0.26, ly - 0.20), text(fill: black70, size: 6.0pt)[DoG], anchor: "west")
    line((lx, ly - 0.40), (lx + 0.18, ly - 0.40), stroke: 0.82pt + congaree)
    content((lx + 0.26, ly - 0.40), text(fill: black70, size: 6.0pt)[Square SG], anchor: "west")

    content((ox + pw / 2, oy - 0.52), text(fill: black90, size: 7.4pt)[Matched FWHM target (px)])
    content((0.18, oy + ph / 2), angle: 90deg, text(fill: black90, size: 7.4pt)[Anisotropy ratio])
  })
}
