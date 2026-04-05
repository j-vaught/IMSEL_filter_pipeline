#import "@preview/cetz:0.3.4"

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let garnet = rgb("#73000A")
  let atlantic = rgb("#466A9F")
  let congaree = rgb("#1F414D")
  let rose = rgb("#CC2E40")
  let horseshoe = rgb("#65780B")
  let warmgrey = rgb("#676156")
  let black90 = rgb("#363636")
  let light-bg = rgb("#ECECEC")

  // ── helpers ──────────────────────────────────────────────────
  let model-box(pos, label, subtitle, color, w: 3.4, h: 1.6, dashed: false) = {
    let stk = if dashed { (paint: color, thickness: 1.6pt, dash: "dashed") } else { color + 1.6pt }
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      fill: color.lighten(85%),
      stroke: stk,
    )
    content((pos.at(0), pos.at(1) + 0.22), text(size: 9pt, weight: "bold", fill: color, label))
    content((pos.at(0), pos.at(1) - 0.28), text(size: 7pt, fill: black90, subtitle))
  }

  let step-box(pos, label, color, w: 3.0, h: 1.2) = {
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      fill: color.lighten(90%),
      stroke: color + 1.2pt,
    )
    content(pos, text(size: 8pt, weight: "bold", fill: color, label))
  }

  let step-label(pos, num, desc, color) = {
    content(pos, text(size: 7.5pt, weight: "bold", fill: color)[Step #num])
    content((pos.at(0), pos.at(1) - 0.4), text(size: 6.5pt, fill: black90, desc))
  }

  let arrow(from, to, color: black90, label: none, label-pos: "above", bend: 0) = {
    line(from, to, mark: (end: "stealth", fill: color, scale: 0.7), stroke: color + 1pt)
    if label != none {
      let mx = (from.at(0) + to.at(0)) / 2
      let my = (from.at(1) + to.at(1)) / 2
      let offset = if label-pos == "above" { 0.3 } else { -0.3 }
      content((mx, my + offset), text(size: 6pt, fill: warmgrey, style: "italic", label))
    }
  }

  let dashed-arrow(from, to, color: black90, label: none, label-pos: "above") = {
    line(from, to, mark: (end: "stealth", fill: color, scale: 0.7), stroke: (paint: color, thickness: 1pt, dash: "dashed"))
    if label != none {
      let mx = (from.at(0) + to.at(0)) / 2
      let my = (from.at(1) + to.at(1)) / 2
      let offset = if label-pos == "above" { 0.3 } else { -0.3 }
      content((mx, my + offset), text(size: 6pt, fill: warmgrey, style: "italic", label))
    }
  }

  // ── layout ───────────────────────────────────────────────────
  // Four models are positioned in a large rectangle.
  // Policy (top-left), Reference (bottom-left),
  // Reward (top-right), Value/Critic (bottom-right)
  // The training loop flows clockwise through four steps.

  let px = 0       // policy x
  let py = 5.5     // policy y
  let refx = 0     // reference x
  let refy = 0     // reference y
  let rx = 14      // reward x
  let ry = 5.5     // reward y
  let vx = 14      // value x
  let vy = 0       // value y

  // ── Step 1 annotation (top center) ──────────────────────────
  step-label((7, 8.2), "1", "Generate response", garnet)

  // Prompt input
  step-box((-5.5, 5.5), "Prompt x", warmgrey, w: 2.6, h: 1.2)
  arrow((-4.2, 5.5), (-1.7, 5.5), label: "input", label-pos: "above")

  // ── The four models ─────────────────────────────────────────
  // Policy model (trainable)
  model-box((px, py), [Policy $pi_theta$], "Trainable", garnet, w: 3.4, h: 1.6)

  // Reference model (frozen)
  model-box((refx, refy), [Reference $pi_"ref"$], "Frozen (SFT copy)", atlantic, w: 3.4, h: 1.6, dashed: true)

  // Reward model (frozen)
  model-box((rx, ry), [Reward $r_phi$], "Frozen (preference-trained)", horseshoe, w: 3.8, h: 1.6, dashed: true)

  // Value model / Critic (trainable)
  model-box((vx, vy), [Value $V_psi$], "Trainable (critic)", congaree, w: 3.4, h: 1.6)

  // ── Response generation flow (Step 1) ───────────────────────
  // Policy generates response y
  step-box((7, 5.5), [Response $y$], black90, w: 2.8, h: 1.0)
  arrow((1.7, 5.5), (5.6, 5.5), label: "sample tokens", label-pos: "above")

  // Response goes to reward model
  arrow((8.4, 5.5), (12.1, 5.5), label: "score complete response", label-pos: "above")

  // ── Step 2 annotation ───────────────────────────────────────
  step-label((14, 8.2), "2", [Compute reward $r_phi (x, y)$], horseshoe)

  // Reward flows down to advantage computation
  step-box((14, 2.8), [Advantage $hat(A)_t$], rose, w: 3.2, h: 1.0)

  arrow((14, 4.7), (14, 3.3), color: horseshoe, label: "reward signal", label-pos: "above")
  content((15.5, 4.0), text(size: 6pt, fill: horseshoe, style: "italic")[scalar score])

  // Value model feeds into advantage
  arrow((14, 0.8), (14, 2.3), color: congaree, label: "baseline estimate", label-pos: "above")
  content((15.7, 1.5), text(size: 6pt, fill: congaree, style: "italic")[$V_psi (s_t)$])

  // ── Step 3 annotation ───────────────────────────────────────
  step-label((7, -1.8), "3", "Estimate advantages with GAE", congaree)

  // The response also goes down to the value model for state estimates
  arrow((7, 5.0), (7, 3.8), color: black90)
  arrow((7, 3.8), (12.3, 0), color: congaree, label: "token states", label-pos: "above")

  // Also need reference log-probs for KL
  // Policy sends log-probs down to reference comparison
  step-box((3.5, 2.8), [KL penalty], atlantic, w: 2.6, h: 1.0)
  arrow((0, 4.7), (0, 3.6), color: garnet)
  arrow((0, 3.6), (2.2, 2.8), color: garnet, label: [log $pi_theta$], label-pos: "above")
  arrow((0, 0.8), (0, 1.9), color: atlantic)
  arrow((0, 1.9), (2.2, 2.8), color: atlantic, label: [log $pi_"ref"$], label-pos: "below")

  // ── Step 4: Policy update ───────────────────────────────────
  step-label((0, -1.8), "4", "Clipped policy update", garnet)

  // Advantage and KL flow into the update
  step-box((7, -0.5), [$L^"CLIP" (theta)$], garnet, w: 3.0, h: 1.0)

  arrow((12.4, 2.8), (8.5, -0.5), color: rose, label: [advantage $hat(A)_t$], label-pos: "above")
  arrow((4.8, 2.8), (5.5, -0.5), color: atlantic, label: [$beta D_"KL"$], label-pos: "above")

  // Update flows back to policy (the feedback loop)
  line((5.5, -0.5), (3.0, -0.5), stroke: garnet + 1.2pt)
  line((3.0, -0.5), (-3.0, -0.5), stroke: garnet + 1.2pt)
  line((-3.0, -0.5), (-3.0, 5.5), stroke: garnet + 1.2pt)
  line((-3.0, 5.5), (-1.7, 5.5), mark: (end: "stealth", fill: garnet, scale: 0.7), stroke: garnet + 1.2pt)
  content((-3.5, 2.5), text(size: 6.5pt, fill: garnet, weight: "bold")[Update])
  content((-3.5, 2.1), text(size: 6pt, fill: garnet)[$theta arrow.l theta + alpha nabla L$])

  // Update also flows to value model
  arrow((8.5, -0.5), (12.3, -0.5), color: congaree)
  line((15.7, -0.5), (15.7, -0.8), mark: (end: "stealth", fill: congaree, scale: 0.7), stroke: (paint: congaree, thickness: 1pt))
  line((12.3, -0.5), (15.7, -0.5), stroke: congaree + 1pt)
  line((15.7, -0.8), (15.7, -0.8), stroke: congaree + 1pt)
  content((15.7, -1.3), text(size: 6pt, fill: congaree)[Critic update: $psi arrow.l psi + alpha nabla L^"VF"$])

  // ── Legend ───────────────────────────────────────────────────
  rect((-6.0, -3.0), (4.5, -1.8), fill: light-bg, stroke: black90 + 0.5pt)
  content((-0.75, -2.0), text(size: 7pt, weight: "bold", fill: black90)[Legend])

  // Trainable
  rect((-5.5, -2.85), (-4.5, -2.45), fill: garnet.lighten(85%), stroke: garnet + 1.2pt)
  content((-3.2, -2.65), text(size: 6.5pt, fill: black90)[Trainable model])

  // Frozen
  rect((-1.5, -2.85), (-0.5, -2.45), fill: atlantic.lighten(85%), stroke: (paint: atlantic, thickness: 1.2pt, dash: "dashed"))
  content((0.7, -2.65), text(size: 6.5pt, fill: black90)[Frozen model])

  // Gradient flow
  line((2.2, -2.65), (3.2, -2.65), stroke: garnet + 1.2pt, mark: (end: "stealth", fill: garnet, scale: 0.6))
  content((3.9, -2.65), text(size: 6.5pt, fill: black90)[Gradient])
})
