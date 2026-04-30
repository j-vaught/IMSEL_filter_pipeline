#set page(width: auto, height: auto, margin: 5pt)
#set text(font: "New Computer Modern", size: 28pt)

#grid(
  columns: 5,
  gutter: 4pt,
  figure(
    image("../canny_a_original.png"),
    caption: [(a) Input],
    supplement: none,
    numbering: none,
  ),
  figure(
    image("../canny_b_smoothed.png"),
    caption: [(b) Smoothed],
    supplement: none,
    numbering: none,
  ),
  figure(
    image("../canny_c_gradient.png"),
    caption: [(c) Gradient],
    supplement: none,
    numbering: none,
  ),
  figure(
    image("../canny_d_angle.png"),
    caption: [(d) Direction],
    supplement: none,
    numbering: none,
  ),
  figure(
    image("../canny_e_edges.png"),
    caption: [(e) Edges],
    supplement: none,
    numbering: none,
  ),
)
