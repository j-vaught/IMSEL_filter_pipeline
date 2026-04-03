= Appendix: Deep Learning Architecture Outputs <sec:appendix>

@fig:hed-outputs shows the five side outputs and the fused output of the Holistically-Nested Edge Detection (HED) network applied to a natural scene. Side output S1 captures fine-scale texture and detail edges. As the stage depth increases (S2 through S5), the detected edges become progressively coarser, shifting from local gradient responses toward object-level contours. The fused output combines all five scales into a single edge map that balances fine detail and semantic structure.

#figure(
  image("../figures/fig_appendix_hed_side_outputs.pdf", width: 100%),
  caption: [HED side outputs and fused result. (a) Input image. (b)--(f) Side outputs S1 (finest) through S5 (coarsest), each produced by a different depth of the VGG-16 backbone and supervised independently during training. (g) The learned weighted fusion of all five side outputs. Fine-scale outputs respond to texture and thin lines; coarse-scale outputs respond to object boundaries and scene structure.],
  placement: top,
  scope: "parent",
) <fig:hed-outputs>
