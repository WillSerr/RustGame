use std::f32::consts::PI;

use crate::{objects::{player_object, terrain_object::TerrainObject}, view_port::renderer::RenderObject};
use sdl3::keyboard::{KeyboardState,Scancode};
use na::{Vector2,Vector3, VectorSlice3};

use super::game_object::GameObject;


pub struct PlayerObject{
    object: GameObject,
    rotation_rate: f32,
    velocity: Vector2<f32>,
    drag: f32,
    active: bool,
}

impl PlayerObject{
    pub fn new(render_info: RenderObject) -> Self{
        Self{
            object: GameObject::new(render_info),
            rotation_rate: 0.0,
            velocity: Vector2::new(0.0,0.0),
            //F = 0.5 * p * pow(u,2) * c * A
            // p = mass density, regular air is 1.293
            // u = flow velocity relative to the object, relative velocity of the object against the flow
            // c = drag coefficient, lookup what this is for the geometry, cube = 1.05
            // A = wikipedia copypasta says it best "A is typically defined as the area of the orthographic projection of the object on a plane perpendicular to the direction of motion."
            //      so for a cube here it'll be a square so 1x1, 1sqrm
            // F = 0.5 * 1.293 * pow(u,2) * 1.05 * 1
            // F = 0.678825 * pow(velocity - wind velocity,2)
            drag: 0.678825,

            //https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/kiteaero.html
            //Lift = Cl * A * r * .5 * V^2
            //Drag = Cd * A * r * .5 * V^2
            //100 pixels = 1 metre (for now), sprite = 256px, A = 2sqrMetres
            //r = air density, 1.229 updated from before
            //CL = lift coefficient, one source has it at Cl = 2* pi * sin(angle of attack)
            //CD = drag coefficient, 0.09 streamline, 1.98 Wall side
            //Lift = 2* pi * sin(angle of attack) * 2 * 1.229 * 0.5 * V^2 = 7.722 * sin(attack angle) * V^2
            //Drag = Cd * 1.229 * V^2
            active: false,
        }
    }

    pub fn handle_input(&mut self, key_states: &KeyboardState, delta_time: f32){
        if self.active{
            let mut rotation: f32 = 0.0;

            if (key_states.is_scancode_pressed(Scancode::D)) {
                rotation += 100.0;  
            } 
            if (key_states.is_scancode_pressed(Scancode::A)) {
                rotation -= 100.0;  
            }
            self.rotation_rate += rotation;

            if (key_states.is_scancode_pressed(Scancode::Right)) {
                self.velocity.x += 10.0 * delta_time;  
            }
            if (key_states.is_scancode_pressed(Scancode::Left)) {
                self.velocity.x -= 10.0 * delta_time;  
            } 
            if (key_states.is_scancode_pressed(Scancode::Up)) {
                let mut pos = self.get_position();
                pos.y += 300.0 * delta_time;
                self.set_position(pos); 
            }
        }
    }

    fn apply_gliding_force(&mut self, delta_time: f32, angle_of_attack: f32, normal_velocity : Vector2<f32>, forward : Vector2<f32>)
    {
        let wind_force = 100.0;
        let mut drag_force : Vector2<f32> = Vector2::new(0.0,0.0);
        drag_force.x = -1.0 * self.velocity.x;
        drag_force.y = -1.0 * self.velocity.y;



        let mut lift: Vector2<f32> = Vector2::new(
            wind_force, //Thrust
            9.8);   // -Weight
            
            // a -> b = b-a
            lift.x += (forward.x - normal_velocity.x) * self.velocity.magnitude();
            lift.y += (forward.y - normal_velocity.y) * self.velocity.magnitude();

            // println!("\n\n\n\n\n\n\n\n\nliftx: {}",lift.x );
            // println!("lifty: {}",lift.y );
            // println!("AOT: {}",angle_of_attack / 0.01745  );
            // println!("AOT abs: {}",angle_of_attack.sin().abs() );

        let dvx = (lift.x + drag_force.x) * delta_time;
        self.velocity.x += dvx;

        let dvy = (lift.y + drag_force.y - 9.8)  * delta_time; //-9.8 for gravity
        self.velocity.y += dvy;

    }


    pub fn update(&mut self, delta_time: f32){
        if self.active {
            self.object.add_rotation(self.rotation_rate * delta_time);
            self.rotation_rate *= 0.1 ;
            
            self.object.add_position(Vector3::new(self.velocity.x * 10.0 * delta_time,self.velocity.y * 10.0 * delta_time,0.0));
            let mut new_position = self.object.get_position();
            if (new_position.y < 32.0)
            {
                new_position.y = 32.0;
                self.object.set_position(new_position);
                self.velocity.y = 0.0;
            }


            let normal_velocity = self.velocity.xy().normalize();
            let forward = Vector2::new((self.object.get_rotation()* 0.01745).cos(),(self.object.get_rotation()* -0.01745).sin()).normalize();
            
            //----REMOVE ME---- for tracking rare crash mentioned below
            assert!(forward.magnitude() == 1.0, "Forward Vector not Normalized, forward.mag: {}", forward.magnitude());
            
            let dot_product = normal_velocity.dot(&forward).max(-1.0).min(1.0); //.dot() can but shouldnt output outside of these bounds
            let mut angle_of_attack : f32 = (dot_product).acos().cos().acos() * (Vector3::new(normal_velocity.x,normal_velocity.y,0.0).cross(&Vector3::new(forward.x,forward.y,0.0)).z).signum();
            
            //----REMOVE ME---- This causes a very rare crash, should be fixed now with bounding the dot product
            assert!(!angle_of_attack.is_nan(), "AOT NAN: dotprod{}, crossprod{}, acos.cos.acos {}, acos {}",
            dot_product,  Vector3::new(normal_velocity.x,normal_velocity.y,0.0).cross(&Vector3::new(forward.x,forward.y,0.0)).z,
            (dot_product).acos().cos().acos(),(dot_product).acos());
            
            if angle_of_attack.abs() > 0.5 * PI {
                angle_of_attack = -1.0 * angle_of_attack.signum() * (PI - angle_of_attack.abs())
            }
            assert!(!angle_of_attack.is_nan(), "AOT Nan:{}", angle_of_attack);


            self.apply_gliding_force(delta_time,angle_of_attack,normal_velocity, forward);

            //self.rotation_rate += ((self.velocity.xy().normalize().dot(&forward)).acos().cos().acos() / 0.01745) * (drag_mag + lift_mag)* delta_time ;
                // println!("dF: {}, {}",lift.x + drag_force.x,lift.y + drag_force.y);
                // println!("dF total: {}",lift.x + drag_force.x + lift.y + drag_force.y);
                // println!("dV: {},{}",dvx,dvy);

        }
        
    }

    pub fn handle_terrain_collision(&mut self, terrain: &TerrainObject){
        if !self.active {
            return;
        }
        //28,112 is the bottom corner
        let bottom_pos = self.object.get_world_position_of_local_pos(Vector3::new(28.0/256.0,
                                                                                (128.0-112.0)/128.0,
                                                                                0.0));
        //28,45 is the back top corner
        let top_pos = self.object.get_world_position_of_local_pos(Vector3::new(28.0/256.0,
                                                                                (128.0-45.0)/128.0,
                                                                                0.0));
        //256,45 is the front nose
        let front_pos = self.object.get_world_position_of_local_pos(Vector3::new(1.0,
                                                                                (128.0-45.0)/128.0,
                                                                                0.0));                                                    
        
        //distance from the ground at the front and back
        let df: f32;
        let db: f32;
        if top_pos.y > bottom_pos.y {
            //Rightside up collision checking
            df = terrain.get_height_at(front_pos.x) - front_pos.y;
            db = terrain.get_height_at(bottom_pos.x) - bottom_pos.y;
        }
        else{
            //------------DOESNT WORK PROPERLY BUT IS STABLE-----
            //Upside down collision checking
            df = terrain.get_height_at(top_pos.x) - top_pos.y;
            db = terrain.get_height_at(front_pos.x) - front_pos.y;
        }

        if(bottom_pos.y < terrain.get_height_at(bottom_pos.x)) && (front_pos.y < terrain.get_height_at(front_pos.x)) {

            let d = (terrain.get_height_at(bottom_pos.x) - bottom_pos.y).max(terrain.get_height_at(front_pos.x) - front_pos.y);
            self.object.add_position(Vector3::new(0.0,d,0.0));
            self.velocity.y = 0.0;
            return;
        }

        if bottom_pos.y < terrain.get_height_at(bottom_pos.x){
            // angle between two points on a circle given the distance between them
            // O = 2arcsin(d/2r)

            let d = db;
            let angle = 2.0 * (d / (2.0 * self.object.get_render_info().texture_width as f32)).asin();

            //asin causes Nan if d is high enough
            if !angle.is_nan(){
                self.object.add_rotation(angle / 0.01745);
            }
            
            //left over distance
            let delta_d = (db - df.abs()).max(0.0);
            self.object.add_position(Vector3::new(0.0,delta_d,0.0));

            return;
        }

        if front_pos.y < terrain.get_height_at(front_pos.x){
            
            let d = df;
            let angle = 2.0 * (d / (2.0 * self.object.get_render_info().texture_width as f32)).asin();

            //asin causes Nan if d is high enough
            if !angle.is_nan(){
                self.object.add_rotation(-angle / 0.01745);
            }

            //left over distance
            let delta_d = (df - db.abs()).max(0.0);
            self.object.add_position(Vector3::new(0.0,delta_d,0.0));
            return;
        }
        
    }

    pub fn get_render_info(&mut self) -> &RenderObject{
        return self.object.get_render_info();
    }

    pub fn set_velocity(&mut self, v: Vector2<f32>) {
        self.velocity = v;
    }

    pub fn get_position(&mut self) -> Vector3<f32>{
        self.object.get_position()
    }

    pub fn set_position(&mut self, pos: Vector3<f32>){
        self.object.set_position(pos);
    }

    pub fn set_rotation(&mut self, rot: f32){
        self.object.set_rotation(rot);
    }

    pub fn set_drag(&mut self, new_drag: f32){
        self.drag = new_drag;
    }

    pub fn activate(&mut self){
        self.active = true;
    }

    pub fn is_active(&self) -> bool{
        self.active
    }
}