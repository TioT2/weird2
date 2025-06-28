use crate::{input, math::Vec3f, timer, vec2f, vec3f};

#[derive(Copy, Clone)]
pub struct Camera {
    pub location: Vec3f,
    pub direction: Vec3f,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            location: vec3f!(10.0, 10.0, 10.0),
            direction: vec3f!(-0.544, -0.544, -0.544).normalized(),
        }
    }

    pub fn response(&mut self, timer: &timer::Timer, input: &input::Input) {
        let mut movement = vec3f!(
            (input.is_key_pressed(input::Key::W) as i32 - input.is_key_pressed(input::Key::S) as i32) as f32,
            (input.is_key_pressed(input::Key::D) as i32 - input.is_key_pressed(input::Key::A) as i32) as f32,
            (input.is_key_pressed(input::Key::R) as i32 - input.is_key_pressed(input::Key::F) as i32) as f32,
        );
        let mut rotation = vec2f!(
            (input.is_key_pressed(input::Key::Right) as i32 - input.is_key_pressed(input::Key::Left) as i32) as f32,
            (input.is_key_pressed(input::Key::Down ) as i32 - input.is_key_pressed(input::Key::Up  ) as i32) as f32,
        );

        movement *= timer.get_delta_time() * 256.0;
        rotation *= timer.get_delta_time() * 1.5;

        let dir = self.direction;
        let right = (dir % vec3f!(0.0, 0.0, 1.0)).normalized();
        let up = (right % dir).normalized();

        self.location += dir * movement.x + right * movement.y + up * movement.z;

        let mut azimuth = dir.z.acos();
        let mut elevator = dir.y.signum() * (
            dir.x / (
                dir.x * dir.x +
                dir.y * dir.y
            ).sqrt()
        ).acos();

        elevator -= rotation.x;
        azimuth  += rotation.y;

        azimuth = azimuth.clamp(0.01, std::f32::consts::PI - 0.01);

        self.direction = vec3f!(
            azimuth.sin() * elevator.cos(),
            azimuth.sin() * elevator.sin(),
            azimuth.cos(),
        );
    }
}
