use core::f32;
use std::convert::identity;

use crate::view_port::renderer::{RenderObject};
use sdl3::keyboard::{KeyboardState,Scancode};
use na::{Vector2,Vector3,Vector4,Point3,Point4, Matrix4};

use super::game_object::GameObject;


pub struct PlayerLauncher{
    object: GameObject,
    rotation_rate: f32,
    velocity: Vector2<f32>,
    socket_position:  Vector3<f32>,
    active: bool,
    release_pressed: bool,
    released: bool
}

impl PlayerLauncher{
    pub fn new(render_info: RenderObject) -> Self{

        let mut render_object = GameObject::new(render_info);
        render_object.set_position(Vector3::new(0.0,0.0,0.0));
        render_object.set_local_origin(Vector3::new(1.0,0.5,0.0));

        Self{
            object: render_object,
            rotation_rate: 10.0,
            velocity: Vector2::new(0.0,0.0),
            socket_position : Vector3::new(-233.0,0.0, 0.0),
            active: false,
            release_pressed: true,
            released: false,
        }
        
    }

    pub fn handle_input(&mut self, key_states: &KeyboardState, delta_time: f32){
        if !self.active{
            if (key_states.is_scancode_pressed(Scancode::Space)) {
                self.active = true;  
            }
        }
        else{
            if (!key_states.is_scancode_pressed(Scancode::Space)) {
                self.release_pressed = false;  
            }
            else if !self.release_pressed {
                self.released = true;  
            }
        }
    }

    pub fn update(&mut self, delta_time: f32){
        const MAX_ROT_RATE : f32 = 1000.0;
        if self.object.get_rotation() <= 160.0 && self.active {
            self.object.add_rotation(self.rotation_rate * delta_time);

            if self.rotation_rate < MAX_ROT_RATE {
                self.rotation_rate = f32::min(self.rotation_rate + 1000.0 *delta_time, MAX_ROT_RATE);
            }
        }

    }

    pub fn get_socket_transform(&self) -> (Vector3<f32>,f32) {
        let socket_transform: Matrix4<f32> = Matrix4::<f32>::identity()
        * Matrix4::new_translation(&self.object.get_position()) 
        * Matrix4::from_axis_angle( &Vector3::z_axis(), &self.object.get_rotation() * -0.01745)
        * Matrix4::new_translation(&self.socket_position);

        let socket_world_pos= socket_transform.transform_point(&Point3::new(0.0,0.0, 0.0));
        
        return (socket_world_pos.coords,self.object.get_rotation() - 90.0);
    }

    pub fn get_socket_velocity(&self) -> Vector2<f32>{

        let mag_v = (self.socket_position.x.abs() / 10.0) * (self.rotation_rate * 0.01745);
        let mut v : Vector2<f32> = Vector2::new(
            (self.object.get_rotation() * 0.01745).sin() * mag_v,
            (self.object.get_rotation() * 0.01745).cos() * mag_v,
        );
        // let mut v : Vector2<f32> = Vector2::new(
        //     100.0,
        //     0.0,
        // );
        return v;
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        return self.object.get_render_info();
    }

    pub fn get_released(&self) -> bool{
        self.released
    }

}