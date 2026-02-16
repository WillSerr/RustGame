
extern crate nalgebra as na;
use na::{Matrix, Matrix4, Rotation, Rotation3, Vector3};

use crate::view_port::renderer::{RenderObject};

pub struct GameObject{
    position: Vector3<f32>,
    local_origin: Vector3<f32>,
    rotation: f32,
    render_info: RenderObject,
    width: u32,
    height: u32,
    render_info_outdated: bool,
}

impl GameObject{
    pub fn new(render_info: RenderObject) -> Self{
        let width = render_info.texture_width;
        let height = render_info.texture_height;
        Self{
            position: Vector3::new(0.0,0.0,0.0),
            local_origin: Vector3::new(0.5,0.5,0.0), //Relative as if sprite was 1x1
            rotation: 0.0,
            render_info : render_info,
            width: width,
            height: height,
            render_info_outdated: true,
        }
    }

    pub fn update(&mut self) {
    }

    pub fn set_position(&mut self, pos: Vector3<f32>){
        self.position = pos;
        self.render_info_outdated = true;
    }

    pub fn get_position(&self) -> Vector3<f32>{
        self.position
    }

    pub fn add_position(&mut self, pos: Vector3<f32>){
        self.position += pos;
        self.render_info_outdated = true;
    }

    pub fn set_rotation(&mut self, rot: f32){
        self.rotation = rot;
        self.render_info_outdated = true;
    }

    pub fn get_rotation(&self) -> f32{
        self.rotation
    }

    pub fn add_rotation(&mut self, rot: f32){
        self.rotation = self.rotation + rot;
        self.render_info_outdated = true;
    }

    pub fn set_local_origin(&mut self, new_origin: Vector3<f32>){
        self.local_origin = new_origin;
        self.render_info_outdated = true;
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        if self.render_info_outdated {  //Only update the transform matrix when necessary. There's lots of stationary objects in this app

            let invert_pos = Vector3::new(-1.0,-1.0,-1.0);
            let local_transform_matrix: Matrix4<f32> = Matrix4::new_translation(
                &self.local_origin.component_mul(&invert_pos));

            let mut scale_matrix: Matrix4<f32> = Matrix4::identity();
            scale_matrix.m11 = self.width as f32;
            scale_matrix.m22 = self.height as f32;

            let rotation_matrix: Matrix4<f32> = Matrix4::from_axis_angle(
                &Vector3::z_axis(), //2D rotation axis faces the camera
                self.rotation * -0.01745);  //convert self.rotation(degrees) to radians

            let translation_matrix: Matrix4<f32> = Matrix4::new_translation(&self.position);

            self.render_info.world_transform =  translation_matrix * rotation_matrix * scale_matrix * local_transform_matrix;
            
            self.render_info_outdated = false;
        }

        &self.render_info
    }
}