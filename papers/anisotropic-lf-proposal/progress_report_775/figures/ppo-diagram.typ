#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 0.5cm)

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
  let model-box(pos, label, subtitle, color, w: 3.0, h: 1.4, dashed: false) = {
    let stk = if dashed { (paint: color, thickness: 1.4pt, dash: "dashed") } else { color + 1.4pt }
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      fill: color.lighten(85%),
      stroke: stk,
    )
    content((pos.at(0), pos.at(1) + 0.2), text(size: 8.5pt, weight: "bold", fill: color, label))
    content((pos.at(0), pos.at(1) - 0.25), text(size: 6.5pt, fill: black90, subtitle))
  }

  let process-box(pos, label, color, w: 2.8, h: 0.9) = {
    rect(
      (pos.at(0) - w/2, pos.at(1) - h/2),
      (pos.at(0) + w/2, pos.at(1) + h/2),
      fill: color.lighten(90%),
      stroke: color + 1pt,
    )
    content(pos, text(size: 7.5pt, weight: "bold", fill: color, label))
  }

  let conn(from, to, color: black90, label: none, label-dx: 0, label-dy: 0.25) = {
    line(from, to, mark: (end: "stealth", fill: color, scale: 0.65), stroke: color + 0.9pt)
    if label != none {
      let mx = (from.at(0) + to.at(0)) / 2 + label-dx
      let my = (from.at(1) + to.at(1)) / 2 + label-dy
      content((mx, my), text(size: 5.5pt, fill: warmgrey, style: "italic", label))
    }
  }

  // ════════════════════════════════════════════════════════════
  // LAYOUT (vertical flow, symmetric about x=0)
  //
  //   y=11.5   Step 1 label
  //   y=10.2   Prompt
  //   y=8.6    Policy model
  //   y=6.8    Response
  //   y=5.5    Step 2 label
  //   y=4.2    Reward (left) | Value (right)
  //   y=2.9    Step 3 label
  //   y=1.5    Advantage
  //   y=0.0    Reference (left) | KL penalty (right)
  //   y=-1.2   Step 4 label
  //   y=-2.5   Loss box
  //   y=-4.0   Legend
  // ════════════════════════════════════════════════════════════

  // ── STEP 1: Generate ────────────────────────────────────────
  content((0, 11.6), text(size: 8pt, weight: "bold", fill: garnet)[Step 1])
  content((0, 11.15), text(size: 6.5pt, fill: black90)[Policy generates a response to the prompt])

  process-box((0, 10.2), "Prompt  x", warmgrey, w: 2.4, h: 0.7)

  conn((0, 9.85), (0, 9.35), color: warmgrey, label: "feed prompt")

  // Policy model
  model-box((0, 8.6), [Policy  $pi_theta$], "Trainable", garnet)

  conn((0, 7.9), (0, 7.35), color: garnet, label: "sample tokens")

  // Response
  process-box((0, 6.8), [Response  $y$], black90, w: 2.4, h: 0.7)

  // ── STEP 2: Score ───────────────────────────────────────────
  content((0, 6.0), text(size: 8pt, weight: "bold", fill: horseshoe)[Step 2])
  content((0, 5.55), text(size: 6.5pt, fill: black90)[Score the response and estimate value baseline])

  // Branch point
  line((0, 6.45), (0, 5.1), stroke: black90 + 0.7pt)

  // Reward model (left)
  model-box((-3.5, 4.2), [Reward  $r_phi$], "Frozen", horseshoe, w: 3.0, h: 1.4, dashed: true)

  // Value model (right)
  model-box((3.5, 4.2), [Value  $V_psi$], "Trainable (critic)", congaree, w: 3.0, h: 1.4)

  // Branch lines
  line((0, 5.1), (-3.5, 5.1), stroke: black90 + 0.7pt)
  line((-3.5, 5.1), (-3.5, 4.9), mark: (end: "stealth", fill: horseshoe, scale: 0.65), stroke: horseshoe + 0.9pt)

  line((0, 5.1), (3.5, 5.1), stroke: black90 + 0.7pt)
  line((3.5, 5.1), (3.5, 4.9), mark: (end: "stealth", fill: congaree, scale: 0.65), stroke: congaree + 0.9pt)

  content((-1.75, 5.35), text(size: 5.5pt, fill: warmgrey, style: "italic")[score response])
  content((1.75, 5.35), text(size: 5.5pt, fill: warmgrey, style: "italic")[estimate baseline])

  // ── STEP 3: Compute advantages ──────────────────────────────
  content((0, 2.9), text(size: 8pt, weight: "bold", fill: rose)[Step 3])
  content((0, 2.45), text(size: 6.5pt, fill: black90)[Compute per-token advantages via GAE])

  // Reward output flows down
  conn((-3.5, 3.5), (-3.5, 1.85), color: horseshoe, label: [$r_phi (x, y)$], label-dx: -1.0, label-dy: 0)

  // Value output flows down
  conn((3.5, 3.5), (3.5, 1.85), color: congaree, label: [$V_psi (s_t)$], label-dx: 1.0, label-dy: 0)

  // Advantage box (center)
  process-box((0, 1.5), [Advantage  $hat(A)_t = r - V$], rose, w: 4.0, h: 0.7)

  // Reward feeds into Advantage from left
  line((-3.5, 1.85), (-3.5, 1.5), stroke: horseshoe + 0.9pt)
  line((-3.5, 1.5), (-2.0, 1.5), mark: (end: "stealth", fill: rose, scale: 0.65), stroke: rose + 0.9pt)

  // Value feeds into Advantage from right
  line((3.5, 1.85), (3.5, 1.5), stroke: congaree + 0.9pt)
  line((3.5, 1.5), (2.0, 1.5), mark: (end: "stealth", fill: rose, scale: 0.65), stroke: rose + 0.9pt)

  // Reference model (left, lower)
  model-box((-3.5, 0.0), [Reference  $pi_"ref"$], "Frozen (SFT copy)", atlantic, w: 3.0, h: 1.4, dashed: true)

  // KL penalty box (right, lower)
  process-box((3.5, 0.0), [KL penalty  $beta D_"KL"$], atlantic, w: 3.0, h: 0.7)

  // Reference log-probs to KL box
  conn((-2.0, 0.0), (2.0, 0.0), color: atlantic, label: [log $pi_"ref"$], label-dy: 0.3)

  // Policy log-probs to KL box
  // Route along the right flank, outside the Value box (right edge at x=5.0)
  // and inside the Critic loop (at x=6.5). Use x=5.6.
  line((1.5, 8.9), (5.6, 8.9), stroke: (paint: garnet, thickness: 0.7pt, dash: "dotted"))
  line((5.6, 8.9), (5.6, 0.0), stroke: (paint: garnet, thickness: 0.7pt, dash: "dotted"))
  line((5.6, 0.0), (5.0, 0.0), mark: (end: "stealth", fill: garnet, scale: 0.55), stroke: (paint: garnet, thickness: 0.7pt, dash: "dotted"))
  content((5.9, 6.5), text(size: 5pt, fill: garnet)[log $pi_theta$])

  // ── STEP 4: Update ──────────────────────────────────────────
  content((0, -1.5), text(size: 8pt, weight: "bold", fill: garnet)[Step 4])
  content((0, -1.95), text(size: 6.5pt, fill: black90)[Clipped surrogate update with KL constraint])

  // Loss box
  process-box((0, -2.8), [$L^"CLIP" (theta) - beta D_"KL"$], garnet, w: 4.2, h: 0.8)

  // Advantage flows to loss
  conn((0, 1.15), (0, -2.4), color: rose, label: [$hat(A)_t$], label-dx: -0.5, label-dy: 0)

  // KL flows to loss
  line((3.5, -0.35), (3.5, -2.8), stroke: atlantic + 0.7pt)
  line((3.5, -2.8), (2.1, -2.8), mark: (end: "stealth", fill: garnet, scale: 0.55), stroke: atlantic + 0.7pt)
  content((4.0, -1.6), text(size: 5.5pt, fill: atlantic, style: "italic")[$beta D_"KL"$])

  // ── Feedback loop: gradient back to policy ──────────────────
  line((0, -3.2), (0, -3.7), stroke: garnet + 1.2pt)
  line((0, -3.7), (-6.2, -3.7), stroke: garnet + 1.2pt)
  line((-6.2, -3.7), (-6.2, 8.6), stroke: garnet + 1.2pt)
  line((-6.2, 8.6), (-1.5, 8.6), mark: (end: "stealth", fill: garnet, scale: 0.7), stroke: garnet + 1.2pt)

  content((-5.3, 3.0), text(size: 6pt, weight: "bold", fill: garnet)[Policy])
  content((-5.3, 2.6), text(size: 6pt, weight: "bold", fill: garnet)[update])
  content((-5.3, 2.1), text(size: 5pt, fill: garnet)[$theta <- theta + alpha nabla L$])

  // ── Feedback loop: gradient back to critic ──────────────────
  line((0, -3.7), (6.8, -3.7), stroke: congaree + 1pt)
  line((6.8, -3.7), (6.8, 4.2), stroke: congaree + 1pt)
  line((6.8, 4.2), (5.0, 4.2), mark: (end: "stealth", fill: congaree, scale: 0.65), stroke: congaree + 1pt)

  content((7.3, 1.0), text(size: 6pt, weight: "bold", fill: congaree)[Critic])
  content((7.3, 0.6), text(size: 6pt, weight: "bold", fill: congaree)[update])
  content((7.3, 0.1), text(size: 5pt, fill: congaree)[$psi <- psi$])
  content((7.3, -0.25), text(size: 5pt, fill: congaree)[$+ alpha nabla L^"VF"$])

  // ── Legend ───────────────────────────────────────────────────
  rect((-5.2, -5.4), (5.2, -4.3), fill: light-bg, stroke: black90 + 0.5pt)

  // Trainable
  rect((-4.7, -4.95), (-3.9, -4.65), fill: garnet.lighten(85%), stroke: garnet + 1pt)
  content((-2.8, -4.8), text(size: 6pt, fill: black90)[Trainable model])

  // Frozen
  rect((-1.2, -4.95), (-0.4, -4.65), fill: atlantic.lighten(85%), stroke: (paint: atlantic, thickness: 1pt, dash: "dashed"))
  content((0.5, -4.8), text(size: 6pt, fill: black90)[Frozen model])

  // Gradient flow
  line((1.8, -4.8), (2.6, -4.8), stroke: garnet + 1pt, mark: (end: "stealth", fill: garnet, scale: 0.55))
  content((3.8, -4.8), text(size: 6pt, fill: black90)[Gradient flow])
})
