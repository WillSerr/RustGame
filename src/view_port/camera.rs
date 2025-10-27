extern crate nalgebra as na;
use na::{Matrix4, Vector3};

pub struct Camera{
    position: Vector3<f32>,
    half_screen_width: f32,
    half_screen_height: f32,
    projection_matrix: Matrix4<f32>,
    view_matrix: Matrix4<f32>,
}


impl Camera{
    pub fn new(screen_width: u32,screen_height: u32) -> Self{

        Self{
            position: Vector3::new(0.0, 0.0, 0.0),
            half_screen_width: screen_width as f32 /2.0,
            half_screen_height: screen_height as f32 /2.0,
            projection_matrix: Camera::build_ortho_projection_matrix(screen_width,screen_height),
            view_matrix: Camera::build_ortho_projection_matrix(screen_width,screen_height),
        }
        
    }

    pub fn get_position(&self) -> (f32,f32){

        return (self.position.x,self.position.y);
    }

    pub fn set_position(&mut self, x_pos: f32, y_pos: f32){
        self.position.x = x_pos;
        self.position.y = y_pos;

        self.rebuild_view_matrix();
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32>{
        return self.view_matrix;
    }

    pub fn rebuild_view_matrix(&mut self){
        let shift: Vector3<f32> = Vector3::new(-self.position.x,-self.position.y,0.0);

        self.view_matrix = self.projection_matrix.append_translation(&shift);
    }

    fn build_ortho_projection_matrix(screen_width: u32,screen_height: u32) -> Matrix4<f32>{
                //Orthographic View Matrix
        let window_width_float: f32 = screen_width as f32;
        let window_height_float: f32 = screen_height as f32;
                return Matrix4::new(
        2.0 / (window_width_float), 0.0, 0.0, 0.0,
        0.0, 2.0 / (window_height_float), 0.0, 0.0,
        // Note: this is assuming a clip space of [0, 1] on the Z axis, which is what Vulkan uses.
        // In OpenGL, the clip space is [-1, 1] and this would need to be adjusted.
        0.0, 0.0, -2.0 , 0.0,
        0.0, 0.0, -1.0 , 1.0
        );
    }
    
}