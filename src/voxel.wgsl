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

[[group(1), binding(0)]]
var output: texture_storage_2d<rgba8unorm, write>;



[[stage(compute), workgroup_size(1)]]
fn main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>
) {
    let output_color: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    let coords: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    textureStore(output, coords, output_color);
}