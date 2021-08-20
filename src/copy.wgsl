let positions: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, -1.0),
);

let indices: array<i32, 6> = array<i32, 6>(
    0, 1, 2,
    0, 2, 3,
);

[[stage(vertex)]]
fn main(
    [[builtin(vertex_index)]] in_vertex_index: u32,
) -> [[builtin(position)]] vec4<f32> {
    if (in_vertex_index == u32(0)) {
        return vec4<f32>(-1.0, 1.0, 0.0, 1.0);
    } elseif (in_vertex_index == u32(1)) {
        return vec4<f32>(1.0, -1.0, 0.0, 1.0);
    } elseif (in_vertex_index == u32(2)) {
        return vec4<f32>(1.0, 1.0, 0.0, 1.0);
    } elseif (in_vertex_index == u32(3)) {
        return vec4<f32>(-1.0, 1.0, 0.0, 1.0);
    } elseif (in_vertex_index == u32(4)) {
        return vec4<f32>(-1.0, -1.0, 0.0, 1.0);
    } elseif (in_vertex_index == u32(5)) {
        return vec4<f32>(1.0, -1.0, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0);
    }
}


[[block]]
struct Camera {
    view_pos: vec4<f32>;
    size: vec2<f32>;
};

[[group(0), binding(0)]]
var<uniform> camera: Camera;

[[block]]
struct Octree {
    data: [[stride(4)]] array<u32>;
}; // this is used as both input and output for convenience

[[group(0), binding(1)]]
var<storage, read> octree: Octree;


[[stage(fragment)]]
fn main(
    [[builtin(position)]] in_fragment_position: vec4<f32>,
) -> [[location(0)]] vec4<f32> {
    var color: f32 = 0.0;
    let difference: f32 = abs(in_fragment_position.x - camera.size.x);
    if(difference < 20.0) {
        color = 1.0;
    }
    return vec4<f32>(color, 0.0, 0.0, 1.0);
}
