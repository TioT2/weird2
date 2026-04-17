use crate::{input, math::{Mat4f, Vec2f, Vec3, Vec3f}, timer};

#[derive(Copy, Clone)]
pub struct Camera {
    pub location: Vec3f,
    pub direction: Vec3f,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            location: Vec3f::new(10.0, 10.0, 10.0),
            direction: Vec3f::new(-0.544, -0.544, -0.544).normalized(),
        }
    }
}

impl Camera {
    /// Create new camera
    pub fn new(location: Vec3f, direction: Vec3f) -> Self {
        Self { location, direction: direction.normalized() }
    }

    /// Response for input and delta time
    pub fn response(&mut self, timer: &timer::Timer, input: &input::Input) {
        let mut movement = Vec3f::new(
            (input.is_key_pressed(input::Key::W) as i32 - input.is_key_pressed(input::Key::S) as i32) as f32,
            (input.is_key_pressed(input::Key::D) as i32 - input.is_key_pressed(input::Key::A) as i32) as f32,
            (input.is_key_pressed(input::Key::R) as i32 - input.is_key_pressed(input::Key::F) as i32) as f32,
        );
        let mut rotation = Vec2f::new(
            (input.is_key_pressed(input::Key::Right) as i32 - input.is_key_pressed(input::Key::Left) as i32) as f32,
            (input.is_key_pressed(input::Key::Down ) as i32 - input.is_key_pressed(input::Key::Up  ) as i32) as f32,
        );

        movement *= (timer.get_delta_time() * 256.0).into();
        rotation *= (timer.get_delta_time() * 1.5).into();
        let dir = self.direction;
        let right = (dir % Vec3f::new(0.0, 0.0, 1.0)).normalized();
        let up = (right % dir).normalized();

        self.location += Vec3::new(dir, right, up).dot(movement.map(Vec3f::broadcast));

        let mut azimuth = dir.z().acos();
        let mut elevator = dir.y().signum() * (
            dir.x() / (dir.x() * dir.x() + dir.y() * dir.y()).sqrt()
        ).acos();

        elevator -= rotation.x();
        azimuth  += rotation.y();

        azimuth = azimuth.clamp(0.01, std::f32::consts::PI - 0.01);

        self.direction = Vec3f::new(
            azimuth.sin() * elevator.cos(),
            azimuth.sin() * elevator.sin(),
            azimuth.cos(),
        );
    }

    pub fn compute_view_matrix(&self) -> Mat4f {
        Mat4f::view(
            self.location,
            self.location + self.direction,
            Vec3f::new(0.0, 0.0, 1.0),
        )
    }
}
