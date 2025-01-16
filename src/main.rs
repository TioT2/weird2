
use std::collections::HashMap;

use brush::Brush;
use geom::{Plane, PointRelation};
use math::Vec3f;
use sdl2::{event::Event, keyboard::Scancode};

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

/// Binary space partition implementation
pub mod bsp;

/// Brush builder
pub mod brush;

/// Basic geometry
pub mod geom;

/// New map implementation
pub mod map;

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

    pub fn is_key_clicked(&self, key: Scancode) -> bool {
        let key = self.get_key_state(key);

        key.pressed && key.changed
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

            // points are written in clockwise order, as I know
            let normal = -((p3 - p2).normalized() % (p1 - p2).normalized()).normalized();
            let distance = p2 ^ normal;

            current.push(Plane { normal, distance });
        }
    }

    Some(result)
}

pub unsafe fn render_brushes(brushes: &[Brush], camera_location: Vec3f) {
    for brush in brushes {
        'polygon_loop: for polygon in &brush.polygons {
            let point_normal = Vec3f::cross(
                (polygon.points[2] - polygon.points[1]).normalized(),
                (polygon.points[0] - polygon.points[1]).normalized(),
            ).normalized();
            // point_normal = point_normal * (point_normal ^ polygon.plane.normal).signum();

            if (point_normal ^ camera_location) - polygon.plane.distance <= 0.0 {
                continue 'polygon_loop;
            }

            let color = [
                ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
            ];

            glu_sys::glColor3ub(color[0], color[1], color[2]);

            glu_sys::glBegin(glu_sys::GL_POLYGON);
            for point in &polygon.points {
                glu_sys::glVertex3f(point.x, point.y, point.z);
            }
            glu_sys::glEnd();
        }
    }
}

fn main() {
    // yay, this code will not compile on non-local builds)))
    let map = parse_map(include_str!("../temp/e1m1.map")).unwrap();

    let brushes = map
        .into_iter()
        .filter_map(|planes| Brush::from_planes(&planes))
        .collect::<Vec<_>>()
    ;

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

    // let bsp = {
    //     let polygons = brush::get_map_polygons(&brushes, false);
    //     bsp::Bsp::build(&polygons)
    // };

    let volumes = {
        let mut builder = map::builder::Builder::new();
        builder.start_build_volumes(&brushes);
        builder.volumes
    };

    let mut do_sync_logical_camera = true;
    let mut do_enable_depth_test = true;

    println!("count: {}", volumes.len());
    // return;
    // let mut logical_camera = camera;

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

        // Synchronize logical camera with physical one
        if input.is_key_clicked(Scancode::Backslash) {
            do_sync_logical_camera = !do_sync_logical_camera;
        }

        if input.is_key_clicked(Scancode::Num0) {
            do_enable_depth_test = !do_enable_depth_test;
        }

        // if do_sync_logical_camera {
        //     logical_camera = camera;
        // }

        unsafe {
            if do_enable_depth_test {
                glu_sys::glEnable(glu_sys::GL_DEPTH_TEST);
            } else {
                glu_sys::glDisable(glu_sys::GL_DEPTH_TEST);
            }

            if do_enable_depth_test {
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT | glu_sys::GL_DEPTH_BUFFER_BIT);
            } else {
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT);
            }
            glu_sys::glClearColor(0.30, 0.47, 0.80, 0.0);

            glu_sys::glLoadIdentity();
            glu_sys::gluPerspective(60.0, 8.0 / 6.0, 0.01, 8192.0);

            glu_sys::gluLookAt(
                camera.location.x as f64,
                camera.location.y as f64,
                camera.location.z as f64,
                camera.location.x as f64 + camera.direction.x as f64,
                camera.location.y as f64 + camera.direction.y as f64,
                camera.location.z as f64 + camera.direction.z as f64,
                0.0,
                0.0,
                1.0
            );

            // println!("camera: <{}, {}, {}>", camera.location.x, camera.location.y, camera.location.z);

            // Render hull volumes
            // #[cfg(not)]
            {
                let render_hull_volume = |hull_volume: &map::builder::HullVolume| {
                    for hull_polygon in &hull_volume.polygons {
                        let polygon = &hull_polygon.polygon;

                        // backface culling
                        // if (polygon.plane.normal ^ camera_location) - polygon.plane.distance <= 0.0 {
                        //     continue 'face_polygon_loop;
                        // }

                        let color = [
                            ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                            ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                            ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                        ];

                        glu_sys::glColor3ub(color[0], color[1], color[2]);

                        glu_sys::glBegin(glu_sys::GL_LINE_LOOP);
                        for point in &polygon.points {
                            glu_sys::glVertex3f(point.x, point.y, point.z);
                        }
                        glu_sys::glEnd();

                        for physical_polygon in &hull_polygon.physical_polygons {
                            let polygon = &physical_polygon.polygon;

                            let color = [
                                ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                            ];
    
                            glu_sys::glColor3ub(color[0], color[1], color[2]);
    
                            glu_sys::glBegin(glu_sys::GL_POLYGON);
                            for point in &polygon.points {
                                glu_sys::glVertex3f(point.x, point.y, point.z);
                            }
                            glu_sys::glEnd();
                        }
                    }
                };

                // static mut tact: u32 = 0;
                // tact += 1;
                // render_hull_volume(&volumes[tact as usize % volumes.len()]);
                for hull_volume in &volumes {
                    render_hull_volume(hull_volume);
                }
            }

            // Render BSP
            #[cfg(not)]
            {
                struct VisitContext {
                    location: Vec3f,
                    // debug_plane: Plane,
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
                                    
                                    let color = [
                                        ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                                    ];
            
                                    glu_sys::glColor3ub(color[0], color[1], color[2]);
            
                                    glu_sys::glBegin(glu_sys::GL_POLYGON);
                                    for point in &polygon.points {
                                        glu_sys::glVertex3f(point.x, point.y, point.z);
                                    }
                                    glu_sys::glEnd();
    
                                    // draw intersection line
                                    // let intr = self.debug_plane.intersect_polygon(polygon);
                                    // if let Some((begin, end)) = intr {
                                    //     glu_sys::glColor3ub(0xFF, 0x00, 0x00);
                                    //     glu_sys::glBegin(glu_sys::GL_LINE_STRIP);
                                    //     glu_sys::glVertex3f(begin.x, begin.y, begin.z);
                                    //     glu_sys::glVertex3f(end.x, end.y, end.z);
                                    //     glu_sys::glEnd();
                                    // }
    
                                }
                            }
                        }
                    }
                }
            
                let mut vcontext = VisitContext {
                    location: logical_camera.location,
                    // debug_plane: Plane::from_point_normal(
                    //     Vec3f::new(-200.0, 2000.0, -50.0),
                    //     Vec3f::new(1.0, 2.0, 0.0)
                    // )
                };
    
                vcontext.visit(&bsp);
            }
                    // render_brushes(&brushes, camera.location);
        }

        window.gl_swap_window();
    }
}
