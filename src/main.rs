extern crate sdl3;
mod view_port;


use sdl3::gpu::{SampleCount, ShaderFormat, TextureCreateInfo, TextureFormat, TextureType, TextureUsage, VertexBufferDescription, VertexInputState};
use sdl3::rect::Rect;
use sdl3::pixels::Color;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::render::Canvas;
use std::time::Duration;
use view_port::renderer::Renderer;
use view_port::renderer::RenderObject;

const WINDOW_SIZE: u32 = 800;

pub fn main() -> Result<(), Box<dyn std::error::Error>>{
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;

    let screen_color: Color = Color::RGB(120, 120, 120);
    let window = video_subsystem
        .window("Rust SDL lib testing", WINDOW_SIZE, WINDOW_SIZE)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let gpu = sdl3::gpu::Device::new(
        ShaderFormat::SPIRV | ShaderFormat::DXIL | ShaderFormat::DXBC | ShaderFormat::METALLIB,
        true,
    )?
    .with_window(&window)?;

    let mut game_renderer: Renderer = Renderer::new(WINDOW_SIZE, WINDOW_SIZE, screen_color,&gpu,&window);

    let mut render_objects: Vec<RenderObject> = vec![game_renderer.init_render_object(&gpu).unwrap()];

    render_objects.push(game_renderer.init_render_object(&gpu).unwrap());

    let mut event_pump = sdl_context.event_pump().unwrap();


    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        // The rest of the game loop goes here...
        
        match game_renderer.render(&gpu,&window,&render_objects) {
            Ok(()) => {},
            Err(_) =>break 'running
        };

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }

    Ok(())
}
