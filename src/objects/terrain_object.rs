use crate::objects::game_object;
use crate::view_port::renderer::Vertex;
use crate::{view_port::renderer::RenderObject}; 
use crate::{view_port::renderer::Renderer};
use na::{Vector2,Vector3,Vector4,Point3,Point4, Matrix4};
use sdl3::gpu::Device;
// The prelude import enables methods we use below, specifically
// Rng::random, Rng::sample, SliceRandom::shuffle and IndexedRandom::choose.
use rand::prelude::*;


use super::game_object::GameObject;

const TERRAIN_ORIGIN: na::Vector3<f32> = Vector3::new(30.0,1.0,0.0);

pub struct TerrainObject{
    object: GameObject,
    vertices : [Vertex; 8],
    indices: [u16; 18],
    increment_length_mult: f32,
    rng: ThreadRng
}

impl TerrainObject{
    pub fn new(renderer: &mut Renderer, gpu : &Device) -> Self{

        //TODO---- put this vertex initialising in a for loop + add more vertices along x axis
        // Get a local RNG:
        let mut rng = rand::rng();
        let scalar = 15.0;
        let vertices : [Vertex; 8] = [    
            Vertex{ //B1
            x: 0.0,
            y: 0.0,
            z: 0.0,
            u: 1.0,
            v: 1.0,},
            Vertex{ //T1
            x: 0.0,
            y: 1.0,
            z: 0.0,
            u: 1.0,
            v: 0.0,},
            Vertex{ //B2
            x: 1.0 * scalar,
            y: 0.0,
            z: 0.0,
            u: 1.0,
            v: 1.0,},
            Vertex{ //T2
            x: 1.0 * scalar,
            y: 1.0 * rng.random_range(1.0..=4.0),
            z: 0.0,
            u: 1.0,
            v: 0.0,},
            Vertex{ //B3
            x: 2.0 * scalar,
            y: 0.0,
            z: 0.0,
            u: 1.0,
            v: 1.0,},
            Vertex{ //T3
            x: 2.0 * scalar,
            y: 1.0 * rng.random_range(1.0..=4.0),
            z: 0.0,
            u: 1.0,
            v: 0.0,},
            Vertex{ //B4
            x: 3.0 * scalar,
            y: 0.0,
            z: 0.0,
            u: 1.0,
            v: 1.0,},
            Vertex{ //T4
            x: 3.0 * scalar,
            y: 1.0 * rng.random_range(1.0..=4.0),
            z: 0.0,
            u: 1.0,
            v: 0.0,}];
        let indices : [u16; 18] = [0,2,1, 3,1,2, 
                                2,4,3, 5,3,4, 
                                4,6,5, 7,5,6];

        let mut render_object = GameObject::new(renderer.init_polygon_render_object(gpu, "./assets/green_box.bmp", &vertices, &indices).unwrap());

        render_object.set_position(Vector3::new(0.0,0.0,0.0));

        //--Change local x origin to hide terrain chasing the camera, currently doesnt for debugging purposes
        render_object.set_local_origin(TERRAIN_ORIGIN);

        Self{
        object: render_object,
        vertices: vertices,
        indices: indices,
        increment_length_mult: scalar,
        rng: rng
        }
    }

    
    pub fn update(&mut self, camera_x_pos: f32, renderer: &mut Renderer, gpu : &Device, delta_time: f32){
        let mut delta_pos = camera_x_pos - self.object.get_position().x;
        let terrain_increment_width = 64.0 * 15.0;

        //move the object to keep up with the camera
        while delta_pos.abs() > terrain_increment_width { //While is here in case player moves faster than terrain_increment_width in one frame
            self.object.add_position(Vector3::new(terrain_increment_width * delta_pos.signum(),0.0,0.0));

            //---TODO--- Below only works correctly for forward movement 
            //Move all height data so the terrain object movement is hidden
            for i in (1..(self.vertices.len() - 1)).step_by(2) 
            {
                self.vertices[i].y = self.vertices[i+2].y;
            }
            //add new data onto the end for continuous terrain generation
            self.vertices[self.vertices.len() - 1].y = 1.0 * self.rng.random_range(1.0..=4.0); 

            //Push vertex changes to the object model. ---This might leak mem need to test---
            self.object.update_render_info(renderer.init_polygon_render_object(gpu, "./assets/green_box.bmp", &self.vertices, &self.indices).unwrap());

            //Move terrain object to keep up with camera 
            delta_pos -= terrain_increment_width * delta_pos.signum();
        }
        
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        return self.object.get_render_info();
    }

    //lerp for 0(min) to 1(max)
    fn lerp(ax: f32, ay: f32,bx: f32, by: f32, k: f32) -> f32{
        return (ay * ((bx - k)/(bx - ax))) + (by * ((k - ax)/(bx - ax)));

    }

    //get terrain height at a given x coordinate
    pub fn get_height_at(&self, x_coord: f32) -> f32 {
        let mut front = 0;
        
        //in no way the 'best' way to find the nearest vertices
        for i in (3..self.vertices.len()).step_by(2) 
        {
            let local_pos = (self.vertices[i].x - TERRAIN_ORIGIN.x) * 64.0;
            let world_x = local_pos + self.object.get_position().x;

            if world_x > x_coord{
                front = i;
                break;
            }
        }

        //nearest vertices not found error
        if front == 0 {
            println!("Ground Height ERROR: Could not find nearest vertex, front idx == 0");
            return -12345.6789
        }

        //obtain world coordinates of the two nearest vertices
        let mut local_pos = (self.vertices[front].x - TERRAIN_ORIGIN.x) * 64.0;
        let mut world_x = local_pos + self.object.get_position().x;
        let front_world = Vector2::new(world_x,
                                                                        (self.vertices[front].y - TERRAIN_ORIGIN.y) * 64.0);

        local_pos = (self.vertices[front-2].x - TERRAIN_ORIGIN.x) * 64.0;
        world_x = local_pos + self.object.get_position().x;
        let back_world = Vector2::new(world_x,
                                                                        (self.vertices[front-2].y - TERRAIN_ORIGIN.y) * 64.0);

        
        //as all slopes are a lerp between two vertices, this is how a height it obtained
        let height = TerrainObject::lerp(back_world.x,back_world.y,
                                    front_world.x,front_world.y
                                    ,x_coord);


        return height;
    }

}
