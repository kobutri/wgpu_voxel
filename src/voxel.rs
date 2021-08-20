use wgpu::util::DeviceExt;

use crate::{camera, create_render_pipeline, texture};

pub struct OctreeGPU {
    octree_buffer: wgpu::Buffer,
    input_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    input_bind_group_layout: wgpu::BindGroupLayout,
}

impl OctreeGPU {
    pub fn new(
        device: &wgpu::Device,
        surf_config: &wgpu::SurfaceConfiguration,
        tree_length: u32,
    ) -> Self {
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&input_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = create_render_pipeline(
            device,
            &pipeline_layout,
            surf_config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[],
            wgpu::include_wgsl!("copy.wgsl"),
        );

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[CameraRaw::default()]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let octree_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress
                * tree_length as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: octree_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            octree_buffer,
            camera_buffer,
            pipeline,
            input_bind_group,
            input_bind_group_layout,
        }
    }

    pub fn update_camera(
        &mut self,
        queue: &wgpu::Queue,
        camera: &camera::Camera,
        surf_config: &wgpu::SurfaceConfiguration,
    ) {
        let camera_raw = CameraRaw {
            view_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            size: [surf_config.width as f32, surf_config.height as f32],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_raw]))
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraRaw {
    view_pos: [f32; 4],
    size: [f32; 2],
}

pub trait DrawVoxels<'a> {
    fn draw_voxels(&mut self, octree: &'a OctreeGPU);
}

impl<'a, 'b> DrawVoxels<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_voxels(&mut self, octree: &'a OctreeGPU) {
        self.set_pipeline(&octree.pipeline);
        self.set_bind_group(0, &octree.input_bind_group, &[]);
        self.draw(0..6, 0..1);
    }
}
