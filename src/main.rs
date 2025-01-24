use core::f32;
use std::collections::{BTreeSet, HashMap};

use math::Vec3f;
use sdl2::{event::Event, keyboard::Scancode};

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

/// Binary space partition implementation
pub mod bsp;

/// Basic geometry
pub mod geom;

/// New map implementation
pub mod map;

/// Time measure utility
pub struct Timer {
    /// Timer initialization time point
    start: std::time::Instant,

    /// Current time
    now: std::time::Instant,

    /// Duration between two last timer updates
    dt: std::time::Duration,

    /// Current FPS updating duration
    fps_duration: std::time::Duration,

    /// Last time FPS was measured
    fps_last_measure: std::time::Instant,

    /// Count of timer updates after last FPS recalculation
    fps_frame_counter: usize,

    /// FPS, actually
    fps: Option<f32>,
}

impl Timer {
    /// Create new timer
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            now,
            dt: std::time::Duration::from_millis(60),

            fps_duration: std::time::Duration::from_millis(1000),
            fps_last_measure: now,
            fps_frame_counter: 0,
            fps: None,
        }
    }

    /// Update timer
    pub fn response(&mut self) {
        let new_now = std::time::Instant::now();
        self.dt = new_now.duration_since(self.now);
        self.now = new_now;
        
        self.fps_frame_counter += 1;

        let fps_delta = self.now - self.fps_last_measure;
        if fps_delta > self.fps_duration {
            self.fps = Some(self.fps_frame_counter as f32 / fps_delta.as_secs_f32());

            self.fps_last_measure = self.now;
            self.fps_frame_counter = 0;
        }
    }

    /// Get current duration between frames
    pub fn get_delta_time(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    /// Get current time
    pub fn get_time(&self) -> f32 {
        self.now
            .duration_since(self.start)
            .as_secs_f32()
    }

    /// Get current framaes-per-second
    pub fn get_fps(&self) -> f32 {
        self.fps.unwrap_or(std::f32::NAN)
    }

    /// FPS measure duration
    pub fn set_fps_duration(&mut self, new_fps_duration: std::time::Duration) {
        self.fps_duration = new_fps_duration;
        self.fps_frame_counter = 0;
        self.fps = None;
        self.fps_last_measure = std::time::Instant::now();
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

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    // yay, this code will not compile on non-local builds)))
    // --
    let location_map = map::builder::Map::parse(include_str!("../temp/test3.map")).unwrap();

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

    // camera.location = Vec3f::new(-200.0, 2000.0, -50.0);
    camera.location = Vec3f::new(30.0, 40.0, 50.0);

    let compiled_map = map::builder::build(&location_map);

    // Synchronize visible-set-building and projection cameras
    let mut do_sync_logical_camera = true;

    // Enable depth test (will be removed after software rasterizer implementation)
    let mut do_enable_depth_test = false;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut do_enable_slow_rendering = false;

    // Camera used for visible set building
    let mut logical_camera = camera;

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

        if input.is_key_clicked(Scancode::Minus) {
            do_enable_slow_rendering = !do_enable_slow_rendering;
        }

        if do_sync_logical_camera {
            logical_camera = camera;
        }

        // Display some statistics
        println!("FPS: {}, DB={}, SL={}, SR={}",
            timer.get_fps(),
            do_enable_depth_test as u32,
            do_sync_logical_camera as u32,
            do_enable_slow_rendering as u32,
        );

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

            // -- Render by BSP
            // #[cfg(not)]
            {
                /// Rneder context
                struct RenderContext<'t> {
                    /// Camera location
                    camera_location: Vec3f,

                    /// Camera visibility plane
                    camera_plane: geom::Plane,

                    /// Map reference
                    map: &'t map::Map,
                }

                impl<'t> RenderContext<'t> {
                    /// Build correct volume set rendering order
                    fn order_rendered_volume_set(
                        bsp: &map::Bsp,
                        visible_set: &mut BTreeSet<map::VolumeId>,
                        render_set: &mut Vec<map::VolumeId>,
                        camera_location: Vec3f
                    ) {
                        match bsp {

                            // Handle space partition
                            map::Bsp::Partition {
                                splitter_plane,
                                front,
                                back
                            } => {
                                let (first, second) = match splitter_plane.get_point_relation(camera_location) {
                                    geom::PointRelation::Front | geom::PointRelation::OnPlane => (front, back),
                                    geom::PointRelation::Back => (back, front)
                                };

                                Self::order_rendered_volume_set(
                                    second,
                                    visible_set,
                                    render_set,
                                    camera_location
                                );

                                Self::order_rendered_volume_set(
                                    &first,
                                    visible_set,
                                    render_set,
                                    camera_location
                                );
                            }

                            // Move volume to render set if it's visible
                            map::Bsp::Volume(id) => {
                                if visible_set.remove(id) {
                                    render_set.push(*id);
                                }
                            }

                            // Just ignore this case)
                            map::Bsp::Void => {}
                        }
                    }

                    /// Render single volume
                    fn render_volume(&mut self, id: map::VolumeId) {
                        let volume = self.map.get_volume(id).unwrap();

                        'polygon_rendering: for surface in volume.get_surfaces() {
                            let polygon = self.map.get_polygon(surface.polygon_id).unwrap();

                            // Perform backface culling
                            if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                                continue 'polygon_rendering;
                            }

                            // Get surface material
                            let material = self.map.get_material(surface.material_id).unwrap();

                            // Calculate simple per-face diffuse light
                            let diffuse_light = Vec3f::new(0.30, 0.47, 0.80)
                                .normalized()
                                .dot(polygon.plane.normal)
                                .abs()
                                .min(0.99);

                            // Add little ambiance component)
                            let light = diffuse_light * 0.9 + 0.09;

                            // Calculate color, based on material and light
                            let color = [
                                (material.color.r as f32 * light) as u8,
                                (material.color.g as f32 * light) as u8,
                                (material.color.b as f32 * light) as u8,
                            ];

                            // Render!
                            unsafe {
                                glu_sys::glColor3ub(color[0], color[1], color[2]);

                                glu_sys::glBegin(glu_sys::GL_POLYGON);
                                for point in &polygon.points {
                                    glu_sys::glVertex3f(point.x, point.y, point.z);
                                }
                                glu_sys::glEnd();
                            }
                        }
                    }

                    /// Render all volumes starting from
                    pub fn render(&mut self, start_volume_id: map::VolumeId) {
                        let mut visible_set = BTreeSet::<map::VolumeId>::new();

                        // Visible set DFS edge
                        let mut visible_set_edge = BTreeSet::new();

                        visible_set_edge.insert(start_volume_id);

                        while !visible_set_edge.is_empty() {
                            let mut new_edge = BTreeSet::new();

                            for volume_id in visible_set_edge.iter().copied() {
                                visible_set.insert(volume_id);

                                let volume = self.map.get_volume(volume_id).unwrap();

                                'portal_rendering: for portal in volume.get_portals() {
                                    let portal_polygon = self.map
                                        .get_polygon(portal.polygon_id)
                                        .unwrap();

                                    // Perform standard backface culling
                                    let backface_cull_result =
                                        (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                        >= 0.0
                                    ;

                                    if false
                                        // Perform modified backface culling
                                        || backface_cull_result != portal.is_facing_front

                                        // Check visibility
                                        || self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back

                                        // Check set
                                        || visible_set.contains(&portal.dst_volume_id)
                                        || visible_set_edge.contains(&portal.dst_volume_id)
                                    {
                                        continue 'portal_rendering;
                                    }

                                    new_edge.insert(portal.dst_volume_id);
                                }
                            }

                            visible_set_edge = new_edge;
                        }

                        let mut render_set = Vec::new();

                        Self::order_rendered_volume_set(
                            self.map.get_bsp(),
                            &mut visible_set,
                            &mut render_set,
                            self.camera_location
                        );

                        for index in render_set {
                            self.render_volume(index);
                        }
                    }

                    /// Display ALL level locations
                    fn render_all(&mut self) {
                        let mut visible_set = BTreeSet::from_iter(self.map.all_volume_ids());
                        let mut render_set = Vec::new();

                        Self::order_rendered_volume_set(
                            self.map.get_bsp(),
                            &mut visible_set,
                            &mut render_set,
                            self.camera_location
                        );

                        for index in render_set {
                            self.render_volume(index);
                        }
                    }
                }

                let mut render_context = RenderContext {
                    camera_location: logical_camera.location,
                    camera_plane: geom::Plane::from_point_normal(
                        logical_camera.location,
                        logical_camera.direction
                    ),
                    map: &compiled_map,
                };

                let start_volume_index_opt = compiled_map
                    .get_bsp()
                    .traverse(logical_camera.location);

                if let Some(start_volume_index) = start_volume_index_opt {
                    render_context.render(start_volume_index);
                } else {
                    render_context.render_all();
                }
            }

            // Render hull volumes
            #[cfg(not)]
            {
                let render_hull_volume = |hull_volume: &map::builder::HullVolume, render_hull: bool, render_physical: bool| {
                    for hull_polygon in &hull_volume.faces {
                        let polygon = &hull_polygon.polygon;

                        // backface culling
                        // if (polygon.plane.normal ^ camera_location) - polygon.plane.distance <= 0.0 {
                        //     continue 'face_polygon_loop;
                        // }

                        
                        if render_hull {
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
                        }

                        if render_physical {
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
                    }
                };

                static mut volume_index: u32 = 0;
                if input.is_key_clicked(Scancode::Space) {
                    volume_index += 1;
                }

                println!("Current volume index: {}", volume_index as usize % builder.volumes.len());

                render_hull_volume(
                    &builder.volumes[volume_index as usize % builder.volumes.len()],
                    false,
                    true
                );

                for hull_volume in &builder.volumes {
                    render_hull_volume(hull_volume, true, false);
                }
            }
        }

        window.gl_swap_window();
    }
}
