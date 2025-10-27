extern crate sdl3;
extern crate nalgebra as na;
mod view_port;
mod objects;

use sdl3::gpu::{SampleCount, ShaderFormat, TextureCreateInfo, TextureFormat, TextureType, TextureUsage, VertexBufferDescription, VertexInputState};
use sdl3::rect::Rect;
use sdl3::pixels::Color;
use sdl3::event::Event;
use sdl3::keyboard::Keycode;
use sdl3::render::Canvas;
use std::time::Duration;

use na::{Vector3,Matrix4};

use view_port::renderer::{Renderer, RenderObject};

use objects::game_object::GameObject;

use crate::view_port::renderer;

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

    //Game objects init
    let mut game_objects: Vec<GameObject> = vec![GameObject::new(game_renderer.init_render_object(&gpu).unwrap())];

    game_objects.push(GameObject::new(game_renderer.init_render_object(&gpu).unwrap()));

    let mut translation: na::Vector3<f32> = na::Vector3::new(-400.0,-0.0,0.0);
    game_objects[0].set_position(translation);
    translation.x = 20.0;
    translation.y = -200.0;
    game_objects[1].set_position(translation);
    game_objects[1].set_rotation(45.0);


    game_objects.push(GameObject::new(game_renderer.init_render_object(&gpu).unwrap()));
    translation.x = 0.0;
    translation.y = 0.0;
    game_objects[2].set_position(translation);
    game_objects[2].set_rotation(0.0);

    let mut event_pump = sdl_context.event_pump().unwrap();

    // let mut render_objects: Vec<&RenderObject> = Vec::new();

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
        for object in &mut game_objects{
            //If visible add to render list
            object.update();
        }
        
        let (mut new_x,mut new_y) =  game_renderer.camera.get_position();
        new_x += 0.001;
        //new_y += 0.01;
        game_renderer.camera.set_position(new_x, new_y);

        // Render loop

        let mut render_objects: Vec<&RenderObject> = Vec::new(); //not a good way to do this but a safe Rust way

        for object in &mut game_objects{
            //If visible add to render list
            render_objects.push(object.get_render_info());
        }

        match game_renderer.render(&gpu,&window,&render_objects) {
            Ok(()) => {},
            Err(_) =>break 'running
        };

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }

    Ok(())
}
