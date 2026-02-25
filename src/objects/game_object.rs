
extern crate nalgebra as na;
use na::{Matrix, Matrix4, Point3, Rotation, Rotation3, Vector3};

use crate::view_port::renderer::{RenderObject};

#[derive(Clone)]
pub struct GameObject{
    position: Vector3<f32>,
    local_origin: Vector3<f32>,
    rotation: f32,
    render_info: RenderObject,
    width: u32,
    height: u32,
    render_info_outdated: bool,
    scale: f32,
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
            scale: 1.0,
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

    pub fn set_scale(&mut self, scale: f32){
        self.scale = scale;
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


            let scalars = Vector3::new(self.width as f32 * self.scale,
                                                                            self.height as f32* self.scale,
                                                                            0.0);
            let scale_matrix: Matrix4<f32> = Matrix4::new_nonuniform_scaling_wrt_point(&scalars,&Point3::new(0.0,0.0,0.0));


            let rotation_matrix: Matrix4<f32> = Matrix4::from_axis_angle(
                &Vector3::z_axis(), //2D rotation axis faces the camera
                self.rotation * -0.01745);  //convert self.rotation(degrees) to radians

            let translation_matrix: Matrix4<f32> = Matrix4::new_translation(&self.position);

            self.render_info.world_transform =  translation_matrix * rotation_matrix * scale_matrix * local_transform_matrix;
            
            self.render_info_outdated = false;
        }

        &self.render_info
    }

    pub fn update_render_info(&mut self, render_info: RenderObject){
        self.render_info = render_info;
    }

    //Long name but needs to be descriptive
    pub fn get_world_position_of_local_pos(&mut self, local_pos: Vector3<f32>) -> Vector3<f32>{
        //Matrix transform only works on Points not vectors, so lots of converting here as I prefer working with vectors everywhere else
        let local_point: Point3<f32> = Point3::new(local_pos.x, local_pos.y, local_pos.z);
        let world_pos: Point3<f32> = self.get_render_info().world_transform.transform_point(&local_point);
        return Vector3::new(world_pos.x,world_pos.y,world_pos.z)
    }
}