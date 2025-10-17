
extern crate nalgebra as na;
use na::{Vector3, Rotation3, Matrix4};

use crate::view_port::renderer::{RenderObject};

pub struct GameObject{
    position: Vector3<f32>,
    pub rotation: u32,
    render_info: RenderObject,
    render_info_outdated: bool,
}

impl GameObject{
    pub fn new(render_info: RenderObject) -> Self{
        Self{
            position: Vector3::new(0.0,0.0,0.0),
            rotation: 0,
            render_info : render_info,
            render_info_outdated: true,
        }
    }

    pub fn update(&mut self) {

    }

    pub fn set_position(&mut self, pos: Vector3<f32>){
        self.position = pos;
        self.render_info_outdated = true;
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        if(self.render_info_outdated){

            //Translation
            self.render_info.world_transform = Matrix4::new_translation(&self.position);
            self.render_info.world_transform = Matrix4::prepend_scaling(&self.render_info.world_transform, 100.0);

            self.render_info_outdated = false;
        }

        &self.render_info
    }
}