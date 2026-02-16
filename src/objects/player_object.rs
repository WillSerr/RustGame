use std::f32::consts::PI;

use crate::{objects::player_object, view_port::renderer::RenderObject};
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

    const fn lerp(a: f32, b: f32, k: f32) -> f32{
        a * (1.0 - k) + b * k
    }

    fn apply_hard_drag(&mut self, delta_time: f32, angle_of_attack: f32, normal_velocity : Vector2<f32>)
    {
                        println!("angle of attack: {}",angle_of_attack - 0.01745 );
            let cl: f32; //Lift coefficient
            {
                let clo = (2.0 * PI * 0.5 * (angle_of_attack*2.0).sin());//<- 2x frequency was a me addition, slowed lift when nearing perpendicular
                cl = clo / (1.0 + clo.abs() / PI * 1.0); //Clo / (1 + Clo / (pi * AR) ); 
            }
            let lift_mag = cl  * 1.229 * 0.5  * (self.velocity.magnitude())*(self.velocity.magnitude());//(2.0 * PI * angle_of_attack.cos().abs() * 2.0 * 1.229 * 0.5 * (self.velocity.magnitude())*(self.velocity.magnitude()));
                // println!("lift: {}",lift_mag);

            let cd = 1.28 * angle_of_attack.sin().abs() + (cl * cl) / (0.7 * PI * 1.0); //Cd0 + Cl^2 / (.7 * pi * AR) ) Drag coefficient
            let drag_mag = cd * 1.229 * (self.velocity.magnitude())*(self.velocity.magnitude()); //Cd * A* r * 0.5(v^2)
            let mut drag_force : Vector2<f32> = Vector2::new(0.0,0.0);
            drag_force.x = -1.0 * normal_velocity.x * drag_mag; // drag dependent on player rotation
            drag_force.y = -1.0 * normal_velocity.y * drag_mag; // drag dependent on player rotation
                // println!("Drag: {}",drag_mag);
                
            let lift: Vector2<f32> = Vector2::new(
                -1.0 * normal_velocity.y * lift_mag, //<- trial and error negative here, seems to work
                normal_velocity.x * lift_mag);
                // println!("Lift: {},{} , Drag: {},{}",lift.x,lift.y, drag_force.x,drag_force.y);
            
            //F= ma
            //a = F/m
            //dV/dT = f/m
            //dV = (F/m) * dt
            let dvx = ((drag_force.x + lift.x )/5.0) * delta_time; // unnecessarily complex, might not work, 5 gram mass(because paper)
            self.velocity.x += dvx;

            let dvy = ((drag_force.y + lift.y ) /5.0 - 9.8)  * delta_time; 
            self.velocity.y += dvy;
    }

    fn apply_simple_drag(&mut self, delta_time: f32, angle_of_attack: f32, forward : Vector2<f32>)
    {     
        let arbitrary_scalar: f32 = 2.0;
            //println!("angle of attack: {}",angle_of_attack / 0.01745 );
            // let cl: f32; //Lift coefficient
            // {
            //     let clo = (2.0 * PI * 0.5 * (angle_of_attack*2.0).sin());//<- 2x frequency was a me addition, slowed lift when nearing perpendicular
            //     cl = clo / (1.0 + clo.abs() / PI * 1.0); //Clo / (1 + Clo / (pi * AR) ); 
            // }
            // let lift_mag = cl  * 1.229 * 0.5  * (self.velocity.magnitude())*(self.velocity.magnitude());//(2.0 * PI * angle_of_attack.cos().abs() * 2.0 * 1.229 * 0.5 * (self.velocity.magnitude())*(self.velocity.magnitude()));


            // let cd = 1.28 * angle_of_attack.sin().abs() + (cl * cl) / (0.7 * PI * 1.0); //Cd0 + Cl^2 / (.7 * pi * AR) ) Drag coefficient
            let drag_mag = angle_of_attack.sin().abs() * (self.velocity.magnitude())*(self.velocity.magnitude()); //Cd * A* r * 0.5(v^2)
            let mut drag_force : Vector2<f32> = Vector2::new(0.0,0.0);
            drag_force.x = -1.0 * self.velocity.x.abs(); // drag dependent on player rotation
            drag_force.y = -1.0 * self.velocity.y.abs(); // drag dependent on player rotation

            drag_force.x = -0.0; // drag dependent on player rotation
            drag_force.y = -0.0; // drag dependent on player rotation
            // let lift: Vector2<f32> = Vector2::new(
            //     0.0 - angle_of_attack.sin() * drag_force.y, 
            //     (angle_of_attack.cos().abs() * 9.8) - drag_force.x);
            
            // let mut lift: Vector2<f32> = Vector2::new(
            //     (self.object.get_rotation()* 0.01745).sin() * self.velocity.x.abs(), 
            //     -(self.object.get_rotation()* 0.01745).sin() * self.velocity.x.abs());

            //     //lift.x += -(self.object.get_rotation()* 0.01745).cos() * self.velocity.y.abs();
            //     lift.y += (self.object.get_rotation()* 0.01745).cos().abs() * -1.0 * self.velocity.y.abs();

            //     println!("\n\n\n\n\n\n\n\n\nliftx: {}",lift.x );
            //     println!("lifty: {}",lift.y );
            // let dvx = (lift.x + drag_force.x) * delta_time; // unnecessarily complex, might not work, 5 gram mass(because paper)
            // self.velocity.x += dvx;

            // let dvy = (lift.y + drag_force.y - 9.8)  * delta_time; 
            // self.velocity.y += dvy;

            let mut lift: Vector2<f32> = Vector2::new(
                PlayerObject::lerp(-0.01,-2.0,(self.object.get_rotation()* 0.01745).cos().abs()) * self.velocity.y, //Thrust
                PlayerObject::lerp(-5.0,-0.01,(self.object.get_rotation()* 0.01745).sin().abs()) * self.velocity.y);   // -Weight
                assert!(!lift.x.is_nan(), "Lift.x Nan, self.object.get_rotation(): {}, self.object.get_rotation()* 0.01745).cos().abs(){}, self.velocity.y{}",
                self.object.get_rotation(),(self.object.get_rotation()* 0.01745).cos().abs(),self.velocity.y);

                //lift.x = 0.0;
                //lift.y = 0.0;

                let Wind_force = 1.1 * self.velocity.x;
                //WsinO = mgSinO //paralel force, m=? g/W = force/windResist 0=90-angle of attack
                //lift.x += forward.x * Wind_force * angle_of_attack.cos();
                assert!(!lift.x.is_nan(), "Lift.x Nan, forward.x: {}, Wind_force{}, AOT.cos(){}",forward.x,Wind_force,angle_of_attack.cos());
                //lift.y += forward.y * Wind_force * angle_of_attack.cos();

                //lift.x += forward.x * Wind_force * angle_of_attack.sin().abs();
                assert!(!lift.x.is_nan(), "Lift.x Nan, forward.x: {}, Wind_force{}, AOT.sin(){}",forward.x,Wind_force,angle_of_attack.sin());
                //lift.y += forward.y * Wind_force * angle_of_attack.sin().abs();
                // lift.x += forward.x * Wind_force * angle_of_attack.sin();
                // lift.y += forward.y * Wind_force * angle_of_attack.sin();

                drag_force.x = -self.velocity.x; // drag dependent on player rotation
                drag_force.y = -0.0; // drag dependent on player rotation

                //lift.x += -(self.object.get_rotation()* 0.01745).cos() * self.velocity.y.abs();
                //lift.y += (self.object.get_rotation()* 0.01745).cos().abs() * -1.0 * self.velocity.y.abs();

                println!("\n\n\n\n\n\n\n\n\nliftx: {}",lift.x );
                println!("lifty: {}",lift.y );
                println!("AOT: {}",angle_of_attack / 0.01745  );
                println!("AOT abs: {}",angle_of_attack.sin().abs() );
            let dvx = (lift.x + drag_force.x) * delta_time; // unnecessarily complex, might not work, 5 gram mass(because paper)
            self.velocity.x += dvx;

            let dvy = (lift.y + drag_force.y - 9.8)  * delta_time; 
            self.velocity.y += dvy;

            //F= ma
            //a = F/m
            //dV/dT = f/m
            //dV = (F/m) * dt
            // let dvx = (drag_force.x  + lift.x) * delta_time; // unnecessarily complex, might not work, 5 gram mass(because paper)
            // self.velocity.x += dvx;

            // let dvy = (drag_force.x + lift.y - 9.8) * delta_time; 
            // self.velocity.y += dvy;
    }

    fn apply_super_simple_drag(&mut self, delta_time: f32, angle_of_attack: f32, normal_velocity : Vector2<f32>, forward : Vector2<f32>)
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

            println!("\n\n\n\n\n\n\n\n\nliftx: {}",lift.x );
            println!("lifty: {}",lift.y );
            println!("AOT: {}",angle_of_attack / 0.01745  );
            println!("AOT abs: {}",angle_of_attack.sin().abs() );

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

                // println!("V: {},{}",self.velocity.x,self.velocity.y);
                // println!("Speed: {}",self.velocity.magnitude());
            let normal_velocity = self.velocity.xy().normalize();
            let forward = Vector2::new((self.object.get_rotation()* 0.01745).cos(),(self.object.get_rotation()* -0.01745).sin()).normalize();
            assert!(forward.magnitude() == 1.0, "Forward mag: {}", forward.magnitude());
            let dot_product = normal_velocity.dot(&forward).max(-1.0).min(1.0); //.dot() can but shouldnt output outside of these bounds
            let mut angle_of_attack : f32 = (dot_product).acos().cos().acos() * (Vector3::new(normal_velocity.x,normal_velocity.y,0.0).cross(&Vector3::new(forward.x,forward.y,0.0)).z).signum();
            assert!(!angle_of_attack.is_nan(), "AOT NAN: dotprod{}, crossprod{}, acos.cos.acos {}, acos {}",
            dot_product,  Vector3::new(normal_velocity.x,normal_velocity.y,0.0).cross(&Vector3::new(forward.x,forward.y,0.0)).z,
            (dot_product).acos().cos().acos(),(dot_product).acos());
            
            if angle_of_attack.abs() > 0.5 * PI {
                angle_of_attack = -1.0 * angle_of_attack.signum() * (PI - angle_of_attack.abs())
            }
            assert!(!angle_of_attack.is_nan(), "AOT Nan:{}", angle_of_attack);

            //self.apply_hard_drag(delta_time,angle_of_attack, normal_velocity);
            //self.apply_simple_drag(delta_time,angle_of_attack, forward);
            self.apply_super_simple_drag(delta_time,angle_of_attack,normal_velocity, forward);

            //self.rotation_rate += ((self.velocity.xy().normalize().dot(&forward)).acos().cos().acos() / 0.01745) * (drag_mag + lift_mag)* delta_time ;
                // println!("dF: {}, {}",lift.x + drag_force.x,lift.y + drag_force.y);
                // println!("dF total: {}",lift.x + drag_force.x + lift.y + drag_force.y);
                // println!("dV: {},{}",dvx,dvy);

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