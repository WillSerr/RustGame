extern crate sdl3;
extern crate nalgebra as na;
mod view_port;
mod objects;

use sdl3::gpu::{SampleCount, ShaderFormat, TextureCreateInfo, TextureFormat, TextureType, TextureUsage, VertexBufferDescription, VertexInputState};
use sdl3::rect::Rect;
use sdl3::pixels::Color;
use sdl3::event::Event;
use sdl3::keyboard::{Keycode,KeyboardState,Scancode};
use sdl3::timer::ticks;
use std::time::Duration;

use na::{Matrix4, Vector2, Vector3};

use view_port::renderer::{Renderer, RenderObject};

use objects::game_object::GameObject;
use objects::player_object::PlayerObject;

use crate::objects::{player_object, terrain_object};
use crate::objects::player_launcher;
use crate::view_port::renderer;

const WINDOW_SIZE: u32 = 800;

pub fn main() -> Result<(), Box<dyn std::error::Error>>{
    let sdl_context = sdl3::init()?;
    let video_subsystem = sdl_context.video()?;

    // let screen_color: Color = Color::RGB(58, 56, 54);
    let screen_color: Color = Color::RGB(58, 56, 54);
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

    //Player character
    let mut player_object = player_object::PlayerObject::new(game_renderer.init_render_object(&gpu,"./assets/player_plane.bmp").unwrap());
    //Throwing Hand
    let mut player_launcher = player_launcher::PlayerLauncher::new(game_renderer.init_render_object(&gpu,"./assets/hand_hold.bmp").unwrap());

    //Background Objects init
    let mut background_objects: Vec<GameObject> = vec![GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_1.bmp").unwrap()),
    GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_2.bmp").unwrap()),
    GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_3.bmp").unwrap()),
    GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_3.bmp").unwrap()),
    GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_3.bmp").unwrap()),
    GameObject::new(game_renderer.init_render_object(&gpu,"./assets/bkg_3.bmp").unwrap())];
    background_objects[0].set_position(na::Vector3::new(0.0,0.0,-0.0001));
    background_objects[0].set_local_origin(na::Vector3::new(0.0,0.0,0.0));
    background_objects[1].set_position(na::Vector3::new(800.0,0.0,-0.0001));
    background_objects[1].set_local_origin(na::Vector3::new(0.0,0.0,0.0));
    background_objects[2].set_position(na::Vector3::new(0.0,800.0,-0.0005));
    background_objects[2].set_local_origin(na::Vector3::new(0.0,0.0,0.0));
    background_objects[3].set_position(na::Vector3::new(800.0,800.0,-0.0005));
    background_objects[3].set_local_origin(na::Vector3::new(0.0,0.0,0.0));
    background_objects[4].set_position(na::Vector3::new(800.0,800.0,-0.0005));
    background_objects[4].set_local_origin(na::Vector3::new(0.0,0.0,0.0));
    background_objects[5].set_position(na::Vector3::new(800.0,800.0,-0.0005));
    background_objects[5].set_local_origin(na::Vector3::new(0.0,0.0,0.0));

    //Terrain objects init
    let mut terrain = terrain_object::TerrainObject::new(&mut game_renderer,&gpu);

    //Game objects init
    let mut game_objects: Vec<GameObject> = vec![GameObject::new(game_renderer.init_default_render_object(&gpu).unwrap())];

    game_objects.push(GameObject::new(game_renderer.init_default_render_object(&gpu).unwrap()));

    let mut translation: na::Vector3<f32> = na::Vector3::new(1200.0,-0.0,0.0);
    game_objects[0].set_position(translation);
    translation.x = 20.0;
    translation.y = -200.0;

    game_objects[1] = GameObject::new(game_renderer.init_render_object(&gpu, "./assets/green_box.bmp").unwrap());
    game_objects[1].set_position(translation);
    game_objects[1].set_rotation(0.0);
    

    //Ground markers. will be removed when I add actaul ground
    game_objects.push(GameObject::new(game_renderer.init_default_render_object(&gpu).unwrap()));
    translation.x = 400.0;
    translation.y = 200.0;
    game_objects[2].set_position(translation);
    game_objects[2].set_rotation(0.0);
    game_objects[2].set_local_origin(Vector3::new(0.5,0.5,0.0));

    game_objects.push(GameObject::new(game_renderer.init_render_object(&gpu, "./assets/green_box.bmp").unwrap()));
        game_objects[3].set_position(translation);
    game_objects[3].set_rotation(0.0);

    game_objects.push(GameObject::new(game_renderer.init_render_object(&gpu, "./assets/green_box.bmp").unwrap()));
        game_objects[4].set_position(translation);
    game_objects[4].set_rotation(0.0);

    //Throwing Hand
    //     game_objects.push(GameObject::new(game_renderer.init_render_object(&gpu, "./assets/hand_hold.bmp").unwrap()));
    // translation.x = 0.0;
    // translation.y = -232.0;
    // game_objects[5].set_local_origin(na::Vector3::new(1.0,0.5,0.0)); //local origin bottom of the arm
    //     game_objects[5].set_position(translation);
    // game_objects[5].set_rotation(0.0);



    let mut event_pump = sdl_context.event_pump().unwrap();

    // let mut render_objects: Vec<&RenderObject> = Vec::new();
    let mut last_tick = ticks();

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
        let delta_time: f32 = (ticks() - last_tick) as f32 * 0.001;
        last_tick = ticks();

        let key_states = KeyboardState::new(&event_pump); //_e is only used for it's lifetime info (if at all)

        player_launcher.handle_input(&key_states, delta_time);
        if player_launcher.get_released() &&  !player_object.is_active(){
            player_object.set_velocity(player_launcher.get_socket_velocity());
            player_object.activate();
        }
        player_object.handle_input(&key_states, delta_time);

        
        // The rest of the game loop goes here...
        
        player_object.update(delta_time);
        player_launcher.update(delta_time);
        terrain.update(player_object.get_position().x ,&mut game_renderer, &gpu, delta_time);

        for object in &mut game_objects{
            //Update all objects, no plans for this to do anything as of yet
            object.update();
        }

        //game_objects[1] = GameObject::new(game_renderer.init_render_object(&gpu, "./assets/green_box.bmp").unwrap());
        //game_objects[1].set_position(translation);
        //game_objects[1].set_rotation(0.0);

        //"ground" marker
        {
            translation = player_object.get_position();
            translation.x = (translation.x / 100.0).floor() * 100.0;
            translation.y = -32.0;
            game_objects[1].set_position(translation);
            
            translation.x += 100.0;
            game_objects[3].set_position(translation);

            translation.x -= 200.0;
            game_objects[4].set_position(translation);
        }

        //Ground height func debug
        {
            translation = player_object.get_position();
            translation.y = terrain.get_height_at(player_object.get_position().x);
            game_objects[2].set_position(translation);
        }

        //Background scrolling
        {
            translation.x = (((player_object.get_position().x + 400.0) / 1600.0).floor() * 1600.0);
            translation.y = 0.0;
            translation.z = -0.0001;
            background_objects[0].set_position(translation);

            translation.y = (((player_object.get_position().y + 400.0) / 1600.0).floor() * 1600.0);
            background_objects[2].set_position(translation);
            translation.y = (((player_object.get_position().y + 1200.0) / 1600.0).floor() * 1600.0) - 800.0;
            background_objects[3].set_position(translation);


            translation.x = (((player_object.get_position().x + 1200.0) / 1600.0).floor() * 1600.0) - 800.0;
            translation.y = 0.0;
            background_objects[1].set_position(translation);

            translation.y = (((player_object.get_position().y + 400.0) / 1600.0).floor() * 1600.0);
            background_objects[4].set_position(translation);
            translation.y = (((player_object.get_position().y + 1200.0) / 1600.0).floor() * 1600.0) - 800.0;
            background_objects[5].set_position(translation);

        }
        
        //Player being held by launcher
        if !player_launcher.get_released() {
            let (pos ,rot)= player_launcher.get_socket_transform();
            player_object.set_position(pos);
            player_object.set_rotation(rot);
        }
        game_renderer.camera.set_position(player_object.get_position());

        // Render loop

        let mut render_objects: Vec<&RenderObject> = Vec::new(); //not a great way to do this but a safe Rust way
        for object in &mut background_objects{
            //If visible add to render list
            if (object.get_position().y >= 0.0){
                render_objects.push(object.get_render_info());
            }
        }

        
        for object in &mut game_objects{
            //If visible add to render list
            render_objects.push(object.get_render_info());
        }
        render_objects.push(player_launcher.get_render_info());
        render_objects.push(player_object.get_render_info());

        render_objects.push(terrain.get_render_info());
        

        match game_renderer.render(&gpu,&window,&render_objects) {
            Ok(()) => {},
            Err(_) =>break 'running
        };

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }

    Ok(())
}
