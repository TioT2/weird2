use std::collections::HashMap;

use brush::Brush;
use geom::{Plane, PointRelation};
use math::{Mat4, Mat4f, Vec3f};
use sdl2::{event::Event, keyboard::Scancode};

struct Map {
    pub points: Vec<Vec3f>,
    pub sides: Vec<(u32, u32)>,
    pub polygons: Vec<Vec<u32>>,
}

#[macro_use]
pub mod math;
pub mod bsp;
pub mod brush;
pub mod geom;
pub mod csg;

#[derive(Copy, Clone)]
pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new() -> Self {
        Self { state: 1 }
    }

    pub fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state <<  5;
        self.state
    }
}

pub struct Timer {
    start: std::time::Instant,
    now: std::time::Instant,
    dt: std::time::Duration,
}

impl Timer {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            now,
            dt: std::time::Duration::from_millis(60),
        }
    }

    pub fn response(&mut self) {
        let new_now = std::time::Instant::now();
        self.dt = new_now.duration_since(self.now);
        self.now = new_now;
    }

    pub fn get_delta_time(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    pub fn get_time(&self) -> f32 {
        self.now
            .duration_since(self.start)
            .as_secs_f32()
    }
}

#[derive(Copy, Clone)]
pub struct KeyState {
    pub pressed: bool,
    pub changed: bool,
}

pub struct Input {
    states: HashMap<Scancode, KeyState>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    pub fn get_key_state(&self, key: Scancode) -> KeyState {
        self.states
            .get(&key)
            .copied()
            .unwrap_or(KeyState {
                pressed: false,
                changed: false,
            })
    }

    pub fn is_key_pressed(&self, key: Scancode) -> bool {
        self.get_key_state(key).pressed
    }

    pub fn on_state_changed(&mut self, key: Scancode, pressed: bool) {
        if let Some(state) = self.states.get_mut(&key) {
            state.changed = state.pressed != pressed;
            state.pressed = pressed;
        } else {
            self.states.insert(key, KeyState {
                pressed,
                changed: true,
            });
        }
    }

    pub fn release_changed(&mut self) {
        for state in self.states.values_mut() {
            state.changed = false;
        }
    }

}

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

    pub fn response(&mut self, timer: &Timer, input: &Input) {
        let mut movement = vec3f!(
            (input.is_key_pressed(Scancode::W) as i32 - input.is_key_pressed(Scancode::S) as i32) as f32,
            (input.is_key_pressed(Scancode::D) as i32 - input.is_key_pressed(Scancode::A) as i32) as f32,
            (input.is_key_pressed(Scancode::R) as i32 - input.is_key_pressed(Scancode::F) as i32) as f32,
        );
        let mut rotation = vec2f!(
            (input.is_key_pressed(Scancode::Right) as i32 - input.is_key_pressed(Scancode::Left) as i32) as f32,
            (input.is_key_pressed(Scancode::Down ) as i32 - input.is_key_pressed(Scancode::Up  ) as i32) as f32,
        );

        movement *= timer.get_delta_time() * 256.0;
        rotation *= timer.get_delta_time() * 1.0;

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

fn parse_map(map: &str) -> Option<Vec<Vec<Plane>>> {
    let mut result = Vec::new();
    let mut current = Vec::new();

    for line in map.lines() {
        if line.starts_with('\"') {
            continue;
        }
        if line.starts_with('{') {
            continue;
        }
        if line.starts_with('}') {
            result.push(current);
            current = Vec::new();
            continue;
        }
        if line.starts_with('(') {
            let mut iter = line.split(' ');

            let mut parse_vec3 = || -> Option<Vec3f> {
                iter.next()?;
                let result = vec3f!(
                    iter.next()?.parse::<i32>().ok()? as f32,
                    iter.next()?.parse::<i32>().ok()? as f32,
                    iter.next()?.parse::<i32>().ok()? as f32,
                );
                iter.next()?;
                Some(result)
            };

            let p1 = parse_vec3()?;
            let p2 = parse_vec3()?;
            let p3 = parse_vec3()?;

            let normal = -((p3 - p2).normalized() % (p1 - p2).normalized()).normalized();

            current.push(Plane {
                normal,
                distance: p2 ^ normal,
            });
        }
    }

    Some(result)
}

fn main() {
    // let mut cube = csg::Brush::cube();
    // cube
    //     .scale(Vec3f::new(2.0, 2.0, 2.0))
    //     .translate(Vec3f::new(-1.0, -1.0, -1.0))
    //     .scale(Vec3f::new(2.0, 2.0, 2.0));

    let map = parse_map(include_str!("../res/maps/e1m1.map")).unwrap();

    // for pvec in &mut map {
    //     for plane in pvec {
    //         std::mem::swap(
    //             &mut plane.normal.y,
    //             &mut plane.normal.z,
    //         );
    //     }
    // }

    let brushes = map
        .into_iter()
        .filter_map(|planes| Brush::from_planes(&planes))
        .collect::<Vec<_>>()
    ;

    let polygons = brush::get_map_polygons(&brushes);

    // let polygons = brushes
    //     .into_iter()
    //     .fold(Vec::new(), |mut vec, brush| {
    //         vec.extend_from_slice(&brush.polygons);
    //         vec
    //     })
    // ;

    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("WEIRD-2", 1024, 768)
        .opengl()
        .build()
        .unwrap()
    ;

    let gl_context = window.gl_create_context().unwrap();
    window.gl_make_current(&gl_context).unwrap();

    let mut timer = Timer::new();
    let mut input = Input::new();
    let mut camera = Camera::new();

    // let polygons = vec![
    //     Polygon::from_cw(vec![
    //         vec3f!(0.0, 0.0, 0.0),
    //         vec3f!(0.0, 2.0, 0.0),
    //         vec3f!(0.0, 2.0, 1.0),
    //         vec3f!(0.0, 0.0, 1.0),
    //     ]),
    //     Polygon::from_cw(vec![
    //         vec3f!(0.0, 2.0, 0.0),
    //         vec3f!(1.0, 2.0, 0.0),
    //         vec3f!(1.0, 2.0, 1.0),
    //         vec3f!(0.0, 2.0, 1.0),
    //     ]),
    //     Polygon::from_cw(vec![
    //         vec3f!(1.0, 2.0, 0.0),
    //         vec3f!(1.0, 1.0, 0.0),
    //         vec3f!(1.0, 1.0, 1.0),
    //         vec3f!(1.0, 2.0, 1.0),
    //     ]),
    //     Polygon::from_cw(vec![
    //         vec3f!(1.0, 1.0, 0.0),
    //         vec3f!(3.0, 1.0, 0.0),
    //         vec3f!(3.0, 1.0, 1.0),
    //         vec3f!(1.0, 1.0, 1.0),
    //     ]),
    //     Polygon::from_cw(vec![
    //         vec3f!(3.0, 1.0, 0.0),
    //         vec3f!(3.0, 0.0, 0.0),
    //         vec3f!(3.0, 0.0, 1.0),
    //         vec3f!(3.0, 1.0, 1.0),
    //     ]),
    //     Polygon::from_cw(vec![
    //         vec3f!(3.0, 0.0, 0.0),
    //         vec3f!(0.0, 0.0, 0.0),
    //         vec3f!(0.0, 0.0, 1.0),
    //         vec3f!(3.0, 0.0, 1.0),
    //     ]),
    // ];

    let bsp = bsp::Bsp::build(&polygons);
    let bsp_polygon_count = bsp.polygon_count();

    'main_loop: loop {
        input.release_changed();

        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Quit { .. } => {
                    break 'main_loop;
                }
                Event::KeyUp { scancode, .. } => {
                    if let Some(code) = scancode {
                        input.on_state_changed(code, false);
                    }
                }
                Event::KeyDown { scancode, .. } => {
                    if let Some(code) = scancode {
                        input.on_state_changed(code, true);
                    }
                }
                _ => {}
            }
        }

        timer.response();
        camera.response(&timer, &input);

        unsafe {
            glu_sys::glEnable(glu_sys::GL_DEPTH_TEST);

            glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT | glu_sys::GL_DEPTH_BUFFER_BIT);
            glu_sys::glClearColor(0.30, 0.47, 0.80, 0.0);

            glu_sys::glLoadIdentity();
            // glu_sys::gluPerspective(30.0, 8.0 / 6.0, 0.01, 8192.0);

            // glu_sys::gluLookAt(
            //     camera.location.x as f64,
            //     camera.location.y as f64,
            //     camera.location.z as f64,
            //     camera.location.x as f64 + camera.direction.x as f64,
            //     camera.location.y as f64 + camera.direction.y as f64,
            //     camera.location.z as f64 + camera.direction.z as f64,
            //     0.0,
            //     0.0,
            //     1.0
            // );

            let view_matrix = Mat4::view(
                camera.location,
                camera.location + camera.direction,
                Vec3f::new(0.0, 0.0, 1.0)
            );
            let projection_matrix = Mat4::projection_frustum(
                -0.01,
                0.01,
                -0.01,
                0.01,
                0.01,
                1024.0
            );
            let view_projection_matrix = view_matrix * projection_matrix;

            struct VisitContext {
                polygon_index: u32,
                polygon_count: u32,
                location: Vec3f,
                vp: Mat4f,
                projection_buffer: Vec<Vec3f>,
            }

            impl VisitContext {
                pub unsafe fn visit(&mut self, node: &bsp::Bsp) {
                    match node {
                        bsp::Bsp::Element { splitter, front, back } => {
                            match splitter.get_point_relation(self.location) {
                                PointRelation::Front => {
                                    self.visit(back);
                                    self.visit(front);
                                }
                                _ => {
                                    self.visit(front);
                                    self.visit(back);
                                }
                            }
                        }
                        bsp::Bsp::Leaf { polygons } => {
                            
                            'polygon_loop: for polygon in polygons {

                                // backface culling
                                if (polygon.plane.normal ^ self.location) - polygon.plane.distance <= 0.0 {
                                    continue 'polygon_loop;
                                }

                                // build projection buffer
                                self.projection_buffer.clear();
                                for point in &polygon.points {
                                    let projected = self.vp.transform_4x4(*point);

                                    // if projected.z < 0.0 || projected.z > 1.0 {
                                    //     continue 'polygon_loop;
                                    // }

                                    self.projection_buffer.push(projected);
                                }

                                let color = [
                                    ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                                ];

                                glu_sys::glColor3ub(color[0], color[1], color[2]);

                                let depth: f32 = 1.0 - self.polygon_index as f32 / self.polygon_count as f32;
                                self.polygon_index += 1;

                                glu_sys::glBegin(glu_sys::GL_POLYGON);
                                for point in &self.projection_buffer {
                                    glu_sys::glVertex3f(point.x, point.y, depth);
                                }
                                glu_sys::glEnd();
                            }
                        }
                    }
                }
            }

            VisitContext {
                polygon_index: 10,
                polygon_count: bsp_polygon_count as u32 + 20,
                location: camera.location,
                vp: view_projection_matrix,
                projection_buffer: Vec::with_capacity(32),
            }
                .visit(&bsp);
        }

        window.gl_swap_window();
    }
}
