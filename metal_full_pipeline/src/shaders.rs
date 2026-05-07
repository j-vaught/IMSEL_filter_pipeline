pub const SHADER_SOURCE: &str = concat!(
    include_str!("shaders/common.metal"),
    include_str!("shaders/wvf.metal"),
    include_str!("shaders/lf.metal"),
    include_str!("shaders/orientation.metal"),
    include_str!("shaders/cgmm.metal"),
);
