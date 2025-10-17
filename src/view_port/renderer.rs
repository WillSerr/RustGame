use sdl3::{
    gpu::{Buffer, BufferBinding, BufferRegion, BufferUsageFlags, ColorTargetDescription, ColorTargetInfo, CommandBuffer, CompareOp, CopyPass, CullMode, DepthStencilState, DepthStencilTargetInfo, Device, FillMode, Filter, GraphicsPipeline, GraphicsPipelineTargetInfo, IndexElementSize, LoadOp, PrimitiveType, RasterizerState, RenderPass, SampleCount, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, ShaderFormat, ShaderStage, StoreOp, Texture, TextureCreateInfo, TextureFormat, TextureRegion, TextureSamplerBinding, TextureTransferInfo, TextureType, TextureUsage, TransferBuffer, TransferBufferLocation, TransferBufferUsage, VertexAttribute, VertexBufferDescription, VertexElementFormat, VertexInputRate, VertexInputState
    
    }, pixels::Color, rect::Rect, video::Window, Error 
    };

extern crate nalgebra as na;
use na::{Matrix4};

use super::texture_manager::TexureManager;

const VERTICES : &[Vertex] = &[    
    Vertex{
    x: 0.0,
    y: 0.0,
    z: 0.0,
    u: 0.0,
    v: 0.0,},
    Vertex{
    x: 0.0,
    y: 1.0,
    z: 0.0,
    u: 0.0,
    v: 1.0,},
    Vertex{
    x: 1.0,
    y: 0.0,
    z: 0.0,
    u: 1.0,
    v: 0.0,},
    Vertex{
    x: 1.0,
    y: 1.0,
    z: 0.0,
    u: 1.0,
    v: 1.0,}];
const INDICES : &[u16] = &[0,2,1, 3,1,2];


#[repr(packed)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub u: f32,
    pub v: f32,
}

pub struct RenderObject{
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub texture_sampler: Sampler,
    pub texture_index: usize,
    pub world_transform: na::Matrix4<f32>,
}

pub struct Renderer{
    pub screen_area: Rect,
    pub screen_color: Color,
    texture_manager: TexureManager,
    depth_texture : Texture<'static>,
    pipeline : GraphicsPipeline,
    view_matrix: Matrix4<f32>,
}

struct MatrixUniform{
    pub transform: Matrix4<f32>,
    pub view: Matrix4<f32>,
}

impl Renderer{
    pub fn new(screen_width: u32,screen_height: u32, clear_color: Color, gpu : &Device, window: &Window) -> Self{

        Self{
            screen_area: Rect::new(0,0,screen_width,screen_height),
            screen_color: clear_color,
            texture_manager : TexureManager::new(),
            pipeline : Renderer::init_pipeline(gpu,window).unwrap(),
            depth_texture : gpu.create_texture(
        TextureCreateInfo::new()
            .with_type(TextureType::_2D)
            .with_width(screen_width)
            .with_height(screen_height)
            .with_layer_count_or_depth(1)
            .with_num_levels(1)
            .with_sample_count(SampleCount::NoMultiSampling)
            .with_format(TextureFormat::D16Unorm)
            .with_usage(TextureUsage::SAMPLER | TextureUsage::DEPTH_STENCIL_TARGET),
            ).unwrap(),
            view_matrix: Renderer::build_view_matrix(screen_width),
        }
        
    }

    pub fn render(&mut self, gpu: &Device, window: &Window,render_objects : &Vec<&RenderObject>) -> Result<(), Error>{        
        // The swapchain texture is basically the framebuffer corresponding to the drawable
        // area of a given window - note how we "wait" for it to come up
        //
        // This is because a swapchain needs to be "allocated", and it can quickly run out
        // if we don't properly time the rendering process.
        let mut command_buffer = gpu.acquire_command_buffer()?;
        if let Ok(swapchain) = command_buffer.wait_and_acquire_swapchain_texture(&window) {
            // Again, like in gpu-clear.rs, we'd want to define basic operations for our cube
            let color_targets = [ColorTargetInfo::default()
                .with_texture(&swapchain)
                .with_load_op(LoadOp::CLEAR)
                .with_store_op(StoreOp::STORE)
                .with_clear_color(self.screen_color)];

            // This time, however, we want depth testing, so we need to also target a depth texture buffer
            let depth_target = DepthStencilTargetInfo::new()
                .with_texture(&mut self.depth_texture)
                .with_cycle(true)
                .with_clear_depth(1.0)
                .with_clear_stencil(0)
                .with_load_op(LoadOp::CLEAR)
                .with_store_op(StoreOp::STORE)
                .with_stencil_load_op(LoadOp::CLEAR)
                .with_stencil_store_op(StoreOp::STORE);
            let render_pass =
                gpu.begin_render_pass(&command_buffer, &color_targets, Some(&depth_target))?;

            // Screen is cleared below due to the color target info
            render_pass.bind_graphics_pipeline(&self.pipeline);


            if render_objects.len() > 0 {
                for objects in render_objects{
                    self.draw_sprite(&render_pass, &command_buffer, objects, 0, 0);
                }
            }

            gpu.end_render_pass(render_pass);
            command_buffer.submit()?;
        }
        else {

            // Swapchain unavailable, cancel work
            command_buffer.cancel();
        }

        Ok(())
    }

    pub fn draw_sprite(&self, render_pass: &RenderPass, command_buffer: &CommandBuffer, render_object: &RenderObject,x_pos: i32, y_pos: i32){
        // Now we'll bind our buffers/sampler and draw the sprite
            render_pass.bind_vertex_buffers(
                0,
                &[BufferBinding::new()
                    .with_buffer(&render_object.vertex_buffer)
                    .with_offset(0)],
            );
            render_pass.bind_index_buffer(
                &BufferBinding::new()
                    .with_buffer(&render_object.index_buffer)
                    .with_offset(0),
                IndexElementSize::_16BIT,
            );
            
            render_pass.bind_fragment_samplers(
                0,
                &[TextureSamplerBinding::new()
                    .with_texture(&self.texture_manager.texture_table[render_object.texture_index])
                    .with_sampler(&render_object.texture_sampler)],
            );

            
            // Set world uniform for our shader
            command_buffer.push_vertex_uniform_data(0, &MatrixUniform{
                transform: render_object.world_transform,
                view: self.view_matrix,
            });

            ////----HARD CODED INDEX COUNT REMEMBER TO CHANGE LATER----
            // Finally, draw the object
            render_pass.draw_indexed_primitives(render_object.index_count, 1, 0, 0, 0);
    }

    pub fn init_render_object(&mut self, gpu: &Device) -> Result<RenderObject, Error>{
        // Create a transfer buffer that is large enough to hold either
        // our vertices or indices since we will be transferring both with it.
        let vertices_len_bytes = std::mem::size_of_val(VERTICES);
        let indices_len_bytes = std::mem::size_of_val(INDICES);
        let transfer_buffer = gpu
            .create_transfer_buffer()
            .with_size(vertices_len_bytes.max(indices_len_bytes) as u32)
            .with_usage(TransferBufferUsage::UPLOAD)
            .build()?;
        let index_count : u32 = INDICES.len().try_into().unwrap();

        // Start a copy pass in order to transfer data to the GPU
        let copy_commands = gpu.acquire_command_buffer()?;
        let copy_pass = gpu.begin_copy_pass(&copy_commands)?;

        // Create GPU buffers to hold our vertices and indices and transfer data to them
        let vertex_buffer = Renderer::create_buffer_with_data(
            &gpu,
            &transfer_buffer,
            &copy_pass,
            BufferUsageFlags::VERTEX,
            VERTICES,
        )?;
        let index_buffer = Renderer::create_buffer_with_data(
            &gpu,
            &transfer_buffer,
            &copy_pass,
            BufferUsageFlags::INDEX,
            INDICES,
        )?;

        // We're done with the transfer buffer now, so release it.
        drop(transfer_buffer);

        // Load up a texture to put on the object
        // let texture_index = self.texture_manager.load_texture(gpu, "./assets/default_texture.bmp",&copy_pass)?;
        let texture_index = self.texture_manager.load_texture(gpu, "./assets/default_texture.bmp",&copy_pass)?;
        
        //println!("tex_idx: {}",texture_index);
        // And configure a sampler for pulling pixels from that texture in the frag shader
        let texture_sampler = gpu.create_sampler(
            SamplerCreateInfo::new()
                .with_min_filter(Filter::Nearest)
                .with_mag_filter(Filter::Nearest)
                .with_mipmap_mode(SamplerMipmapMode::Nearest)
                .with_address_mode_u(SamplerAddressMode::Repeat)
                .with_address_mode_v(SamplerAddressMode::Repeat)
                .with_address_mode_w(SamplerAddressMode::Repeat),
        )?;

        // Now complete and submit the copy pass commands to actually do the transfer work
        gpu.end_copy_pass(copy_pass);
        copy_commands.submit()?;

        Ok(RenderObject { vertex_buffer: vertex_buffer, 
            index_buffer: index_buffer, 
            index_count: index_count,
            texture_sampler: texture_sampler,
            texture_index: texture_index,
            world_transform : na::Matrix4::identity(),
            })
    }

    /// Creates a GPU buffer and uploads data to it using the given `copy_pass` and `transfer_buffer`.
    fn create_buffer_with_data<T: Copy>(
        gpu: &Device,
        transfer_buffer: &TransferBuffer,
        copy_pass: &CopyPass,
        usage: BufferUsageFlags,
        data: &[T],
    ) -> Result<Buffer, Error> {
        // Figure out the length of the data in bytes
        let len_bytes = std::mem::size_of_val(data);

        // Create the buffer with the size and usage we want
        let buffer = gpu
            .create_buffer()
            .with_size(len_bytes as u32)
            .with_usage(usage)
            .build()?;

        // Map the transfer buffer's memory into a place we can copy into, and copy the data
        //
        // Note: We set `cycle` to true since we're reusing the same transfer buffer to
        // initialize both the vertex and index buffer. This makes SDL synchronize the transfers
        // so that one doesn't interfere with the other.
        let mut map = transfer_buffer.map::<T>(gpu, true);
        let mem = map.mem_mut();
        for (index, &value) in data.iter().enumerate() {
            mem[index] = value;
        }

        // Now unmap the memory since we're done copying
        map.unmap();

        // Finally, add a command to the copy pass to upload this data to the GPU
        //
        // Note: We also set `cycle` to true here for the same reason.
        copy_pass.upload_to_gpu_buffer(
            TransferBufferLocation::new()
                .with_offset(0)
                .with_transfer_buffer(transfer_buffer),
            BufferRegion::new()
                .with_offset(0)
                .with_size(len_bytes as u32)
                .with_buffer(&buffer),
            true,
        );

        Ok(buffer)
    }

    fn init_pipeline(gpu : &Device, window: &Window) -> Result<GraphicsPipeline, Error>{
                // Our shaders, require to be precompiled by a SPIR-V compiler beforehand
        let vert_shader = gpu
        .create_shader()
        .with_code(
            ShaderFormat::SPIRV,
            include_bytes!("../../shaders/default-shader.vert.spv"),
            ShaderStage::Vertex,
        )
        .with_uniform_buffers(1)
        .with_entrypoint(c"main")
        .build()?;



        let frag_shader = gpu
            .create_shader()
            .with_code(
                ShaderFormat::SPIRV,
                include_bytes!("../../shaders/default-shader.frag.spv"),
                sdl3::gpu::ShaderStage::Fragment,
            )
            .with_samplers(1)
            .with_entrypoint(c"main")
            .build()?;

        // Create a pipeline, we specify that we want our target format in the swapchain
        // since we are rendering directly to the screen. However, we could specify a texture
        // buffer instead (e.g., for offscreen rendering).
        let swapchain_format = gpu.get_swapchain_texture_format(window);
        let pipeline = gpu
            .create_graphics_pipeline()
            .with_primitive_type(sdl3::gpu::PrimitiveType::TriangleList)
            .with_fragment_shader(&frag_shader)
            .with_vertex_shader(&vert_shader)
            .with_vertex_input_state(
                VertexInputState::new()
                    .with_vertex_buffer_descriptions(&[VertexBufferDescription::new()
                        .with_slot(0)
                        .with_pitch(size_of::<Vertex>() as u32)
                        .with_input_rate(sdl3::gpu::VertexInputRate::Vertex)
                        .with_instance_step_rate(0)])
                    .with_vertex_attributes(&[
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float3)
                            .with_location(0)
                            .with_buffer_slot(0)
                            .with_offset(0),
                        VertexAttribute::new()
                            .with_format(VertexElementFormat::Float2)
                            .with_location(1)
                            .with_buffer_slot(0)
                            .with_offset((3 * size_of::<f32>()) as u32),
                    ]),
            )
            .with_rasterizer_state(
                RasterizerState::new()
                    .with_fill_mode(FillMode::Fill)
                    // Turn off culling so that I don't have to get my cube vertex order perfect
                    .with_cull_mode(CullMode::Back),
            )
            .with_depth_stencil_state(
                // Enable depth testing
                DepthStencilState::new()
                    .with_enable_depth_test(true)
                    .with_enable_depth_write(true)
                    .with_compare_op(CompareOp::Less),
            )
            .with_target_info(
                GraphicsPipelineTargetInfo::new()
                    .with_color_target_descriptions(&[
                        ColorTargetDescription::new().with_format(swapchain_format)
                    ])
                    .with_has_depth_stencil_target(true)
                    .with_depth_stencil_format(TextureFormat::D16Unorm),
            )
            .build()?;

        // The pipeline now holds copies of our shaders, so we can release them
        drop(vert_shader);
        drop(frag_shader);

        Ok(pipeline)
    }

    fn build_view_matrix(window_size: u32) -> Matrix4<f32>{
        let window_size_float: f32 = window_size as f32;
        return Matrix4::new(
        2.0 / (window_size_float), 0.0, 0.0, 0.0,
        0.0, 2.0 / (window_size_float), 0.0, 0.0,
        // Note: this is assuming a clip space of [0, 1] on the Z axis, which is what Vulkan uses.
        // In OpenGL, the clip space is [-1, 1] and this would need to be adjusted.
        0.0, 0.0, -1.0 / 2.0, 0.0,
        0.0, 0.0, 1.0 / 2.0, 1.0
        );
    }
}