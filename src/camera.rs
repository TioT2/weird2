//! Camera implementation module.

use crate::{input, math::{Mat4f, Vec2f, Vec3, Vec3f}, timer};

/// Camera. Assumes Z-up.
#[derive(Copy, Clone)]
pub struct Camera {
    /// Point camera looks from
    location: Vec3f,

    /// Direction camera points to.
    direction: Vec3f,

    /// Right vector
    right: Vec3f,

    /// Up vector
    up: Vec3f,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vec3f::broadcast(10.0), Vec3f::broadcast(-1.0))
    }
}

impl Camera {
    /// Build z-up basis for certain direction.
    fn build_basis(dir: Vec3f) -> (Vec3f, Vec3f, Vec3f) {
        let right = dir.cross(Vec3::new(0.0, 0.0, 1.0));
        let up = right.cross(dir);
        (dir.normalized(), right.normalized(), up.normalized())
    }

    /// Compute (azimuth, elevator) coordinates from unit direction vector.
    fn into_polar(dir: Vec3<f32>) -> (f32, f32) {
        let azimuth = dir.z().acos();
        let elevator = dir.y().signum() * (
            dir.x() / (dir.x() * dir.x() + dir.y() * dir.y()).sqrt()
        ).acos();

        (azimuth, elevator)
    }

    /// Compute direction vector from polar coordinates
    fn from_polar(azimuth: f32, elevation: f32) -> Vec3f {
        let (azs, azc) = azimuth.sin_cos();
        let (els, elc) = elevation.sin_cos();

        Vec3f::new(azs * elc, azs * els, azc)
    }

    /// Create new camera
    pub fn new(location: Vec3f, direction: Vec3f) -> Self {
        let (direction, right, up) = Self::build_basis(direction);
        Self {
            location,
            direction,
            right,
            up
        }
    }

    /// Set camera location
    pub fn set_location(&mut self, location: Vec3f) {
        self.location = location;
    }

    /// Set camera direction
    pub fn set_direction(&mut self, direction: Vec3f) {
        (self.direction, self.right, self.up) = Self::build_basis(direction);
    }

    /// Collect movement and rotation input for standard (WASD + RF + Arrows) controls.
    pub fn collect_axes(input: &input::Input, timer: &timer::Timer) -> (Vec3f, Vec2f) {
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

        (movement, rotation)
    }

    /// Update camera using axes
    pub fn update_camera(&mut self, axis_movement: Vec3f, axis_rotation: Vec2f) {
        let (azimuth, elevator) = Self::into_polar(self.direction);

        self.set_location(self.location + Vec3::new(self.direction, self.right, self.up).dot(axis_movement.map(Vec3f::broadcast)));
        self.set_direction(Self::from_polar(
            (azimuth + axis_rotation.y()).clamp(0.01, std::f32::consts::PI - 0.01),
            elevator - axis_rotation.x()
        ));
    }

    /// Update for certain input and delta time. Composes `collect_axes` with `update_camera`.
    pub fn response(&mut self, timer: &timer::Timer, input: &input::Input) {
        let (movement, rotation) = Self::collect_axes(input, timer);

        self.update_camera(movement, rotation);
    }

    /// Compute view matrix
    pub fn view(&self) -> Mat4f {
        Mat4f::view_from_basis(
            self.location,
            self.direction,
            self.right,
            self.up
        )
    }

    /// Extract camera location
    pub const fn location(&self) -> Vec3f {
        self.location
    }

    /// Extract camera direction
    pub const fn direction(&self) -> Vec3f {
        self.direction
    }

    /// Extract camera right vector
    pub const fn right(&self) -> Vec3f {
        self.right
    }

    /// Extract camera up vector. Z >= 0 due to z-up mode.
    pub const fn up(&self) -> Vec3f {
        self.up
    }
}
