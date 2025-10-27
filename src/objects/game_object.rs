
extern crate nalgebra as na;
use na::{Matrix, Matrix4, Rotation, Rotation3, Vector3};

use crate::view_port::renderer::{RenderObject};

pub struct GameObject{
    position: Vector3<f32>,
    rotation: f32,
    render_info: RenderObject,
    render_info_outdated: bool,
}

impl GameObject{
    pub fn new(render_info: RenderObject) -> Self{
        Self{
            position: Vector3::new(0.0,0.0,0.0),
            rotation: 0.0,
            render_info : render_info,
            render_info_outdated: true,
        }
    }

    pub fn update(&mut self) {
        //self.set_rotation( self.rotation + 1.0);
    }

    pub fn set_position(&mut self, pos: Vector3<f32>){
        self.position = pos;
        self.render_info_outdated = true;
    }

    pub fn set_rotation(&mut self, rot: f32){
        self.rotation = rot;
        self.render_info_outdated = true;
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        if self.render_info_outdated {  //Only update the transform matrix when necessary

            let mut scale_matrix: Matrix4<f32> = Matrix4::identity();
            scale_matrix.m11 = 100.0;
            scale_matrix.m22 = 100.0;

            let rotation_matrix: Matrix4<f32> = Matrix4::from_axis_angle(
                &Vector3::z_axis(), //2D rotation axis faces the camera
                self.rotation * -0.01745);  //convert self.rotation(degrees) to radians

            let translation_matrix: Matrix4<f32> = Matrix4::new_translation(&self.position);

            self.render_info.world_transform =  translation_matrix * rotation_matrix * scale_matrix;
            
            self.render_info_outdated = false;
        }

        &self.render_info
    }
}