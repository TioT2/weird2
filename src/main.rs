/// Main project module

use std::collections::{BTreeSet, HashMap};
use math::{Mat4f, Vec3f};
use sdl2::keyboard::Scancode;

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

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

    // Total frame count
    total_frame_count: u64,

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
            total_frame_count: 0,
            
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
        
        self.total_frame_count += 1;
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

    /// Get count of frames elapsed from start
    pub fn get_frame_count(&self) -> u64 {
        self.total_frame_count
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

impl geom::Polygon {
    /// Clip polygon by (z+, 1) plane.
    /// This method is much faster, than general polygon split.
    fn clip_viewspace_back(&self) -> Self {
        let mut result = Vec::new();

        for index in 0..self.points.len() {
            let curr = self.points[index];
            let next = self.points[(index + 1) % self.points.len()];

            if curr.z > 1.0 {
                result.push(curr);

                if next.z < 1.0 {
                    let t = (1.0 - curr.z) / (next.z - curr.z);

                    result.push((next - curr) * t + curr);
                }
            } else if next.z > 1.0 {
                let t = (1.0 - curr.z) / (next.z - curr.z);

                result.push((next - curr) * t + curr);
            }
        }

        Self {
            points: result,
            plane: self.plane,
        }
    }
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

/// Render context
struct RenderContext<'t> {
    /// Camera location
    camera_location: Vec3f,

    /// Camera visibility plane
    camera_plane: geom::Plane,

    /// VP matrix
    view_projection_matrix: Mat4f,

    /// Map reference
    map: &'t map::Map,

    /// Frame pixel array pointer
    frame_pixels: *mut u32,

    /// Frame width
    frame_width: usize,

    /// Frame height
    frame_height: usize,

    /// Frame stride
    frame_stride: usize,

    /// Rasterization mode
    do_rasterize_overdraw: bool,
}

impl<'t> RenderContext<'t> {
    /// Just test rendering function
    pub fn _test_software_rasterizer(&mut self, time: f32) {
        let cv = Vec3f::new(
            self.frame_width as f32 / 2.0,
            self.frame_height as f32 / 2.0,
            0.0
        );

        let ext_min = usize::min(self.frame_width, self.frame_height);

        let ev = Vec3f::new(
            ext_min as f32 / 4.0,
            ext_min as f32 / 4.0,
            0.1
        );

        let rotm = math::Mat4f::rotate_z(time / 4.0);

        let set1 = [
            Vec3f::new(-1.0, -1.0, 0.0),
            Vec3f::new(-1.0,  1.0, 0.0),
            Vec3f::new( 1.0, -1.0, 0.0),
        ];

        let set2 = [
            Vec3f::new(-1.0,  1.0, 0.0),
            Vec3f::new( 1.0,  1.0, 0.0),
            Vec3f::new( 1.0, -1.0, 0.0),
        ];

        unsafe {
            self.render_clipped_polygon::<false>(&set1.map(|v| rotm.transform_vector(v) * ev + cv), 0xFF0000);
            self.render_clipped_polygon::<false>(&set2.map(|v| rotm.transform_vector(v) * ev + cv), 0x0000FF);
        }
    }

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

    fn clip_screenspace_polygon(&self, mut points: Vec<Vec3f>) -> Vec<Vec3f> {
        const OFFSET: f32 = 0.5;

        let mut result = Vec::new();

        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.x > OFFSET {
                result.push(curr);

                if next.x < OFFSET {
                    let t = (OFFSET - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x > OFFSET {
                let t = (OFFSET - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(&mut points, &mut result);
        
        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y > OFFSET {
                result.push(curr);

                if next.y < OFFSET {
                    let t = (OFFSET - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y > OFFSET {
                let t = (OFFSET - curr.y) / (next.y - curr.y);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(&mut points, &mut result);

        let frame_w = self.frame_width as f32 - 1.0;
        let frame_h = self.frame_height as f32 - 1.0;

        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.x < frame_w - OFFSET {
                result.push(curr);

                if next.x > frame_w - OFFSET {
                    let t = (frame_w - OFFSET - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x < frame_w - OFFSET {
                let t = (frame_w - OFFSET - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(&mut points, &mut result);

        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y < frame_h - OFFSET {
                result.push(curr);

                if next.y > frame_h - OFFSET {
                    let t = (frame_h - OFFSET - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y < frame_h - OFFSET {
                let t = (frame_h - OFFSET - curr.y) / (next.y - curr.y);

                result.push((next - curr) * t + curr);
            }
        }

        result
    }

    unsafe fn render_clipped_polygon<const RENDER_OVERDRAW: bool>(&mut self, points: &[Vec3f], color: u32) {
        /// Index forward by point list
        macro_rules! ind_prev {
            ($index: expr) => {
                ((($index) + points.len() - 1) % points.len())
            };
        }

        /// Index backward by point list
        macro_rules! ind_next {
            ($index: expr) => {
                (($index + 1) % points.len())
            };
        }

        // Calculate min/max
        let (min_y_index, min_y_value, max_y_index, max_y_value) = {
            let mut min_y_index = 0;
            let mut max_y_index = 0;

            let mut min_y = points[0].y;
            let mut max_y = points[0].y;

            for index in 1..points.len() {
                let y = points[index].y;

                if y < min_y {
                    min_y_index = index;
                    min_y = y;
                }

                if y > max_y {
                    max_y_index = index;
                    max_y = y;
                }
            }

            (min_y_index, min_y, max_y_index, max_y)
        };

        /// Epsilon for dy value
        const DY_EPSILON: f32 = 0.001;

        // Calculate polygon bounds
        let first_line = min_y_value.floor() as usize;
        let last_line = max_y_value.ceil() as usize;

        let mut left_index = ind_next!(min_y_index);
        let mut left_prev = points[min_y_index];
        let mut left_curr = points[left_index];
        let mut left_inv_slope = (left_curr.x - left_prev.x) / (left_curr.y - left_prev.y);

        let mut right_index = ind_prev!(min_y_index);
        let mut right_prev = points[min_y_index];
        let mut right_curr = points[right_index];
        let mut right_inv_slope = (right_curr.x - right_prev.x) / (right_curr.y - right_prev.y);

        // Scan for lines
        'line_loop: for line in first_line..last_line {
            // Get current pixel y
            let y = line as f32 + 0.5;

            while y > left_curr.y {
                if left_index == max_y_index {
                    break 'line_loop;
                }

                left_index = ind_next!(left_index);
                left_prev = left_curr;
                left_curr = points[left_index];

                let left_dy = left_curr.y - left_prev.y;

                // Check if edge is flat
                if left_dy <= DY_EPSILON {
                    left_inv_slope = 0.0;
                } else {
                    left_inv_slope = (left_curr.x - left_prev.x) / left_dy;
                }
            }

            while y > right_curr.y {
                if right_index == max_y_index {
                    break 'line_loop;
                }

                right_index = ind_prev!(right_index);
                right_prev = right_curr;
                right_curr = points[right_index];

                let right_dy = right_curr.y - right_prev.y;

                // Check if edge is flat
                if right_dy <= DY_EPSILON {
                    right_inv_slope = 0.0;
                } else {
                    right_inv_slope = (right_curr.x - right_prev.x) / right_dy;
                }
            }

            // Calculate intersection with left edge
            let left_x = left_prev.x + (y - left_prev.y) * left_inv_slope;

            // Calculate intersection with right edge
            let right_x = right_prev.x + (y - right_prev.y) * right_inv_slope;

            // Calculate hline start/end, clip it
            let start = usize::min(left_x.floor() as usize, self.frame_width - 1);
            let end = usize::min(right_x.floor() as usize, self.frame_width - 1);

            let mut pixel_ptr = self.frame_pixels.add(self.frame_stride * line + start);
            let pixel_end = self.frame_pixels.add(self.frame_stride * line + end);

            while pixel_ptr < pixel_end {
                if RENDER_OVERDRAW {
                    *pixel_ptr = 0x0101010u32.wrapping_add(*pixel_ptr);
                } else {
                    *pixel_ptr = color;
                }
                pixel_ptr = pixel_ptr.add(1);
            }
        }
    }

    fn render_polygon_frame(&mut self, polygon: &geom::Polygon, color: [u8; 3]) {
        let color_u32 = u32::from_le_bytes([color[0], color[1], color[2], 0]);

        let points = {
            let proj_polygon = geom::Polygon::from_ccw(polygon
                .points
                .iter()
                .map(|point| self.view_projection_matrix.transform_point(*point))
                .collect::<Vec<_>>()
            );

            // Clip polygon back in viewspace
            let clip_polygon = proj_polygon.clip_viewspace_back();

            let width = self.frame_width as f32;
            let height = self.frame_height as f32;

            // Projected points
            let screenspace_polygon = clip_polygon.points
                .iter()
                .map(|v| {
                    Vec3f::new(
                        (1.0 + v.x / v.z) * 0.5 * width,
                        (1.0 - v.y / v.z) * 0.5 * height,
                        1.0 / v.z
                    )
                })
                .collect::<Vec<_>>();

            let screenspace_polygon = self
                .clip_screenspace_polygon(screenspace_polygon);

            for pt in &screenspace_polygon {
                if !pt.x.is_finite() || !pt.y.is_finite() {
                    return;
                }
            }

            screenspace_polygon
        };

        if points.len() < 3 {
            return;
        }

        unsafe {
            if self.do_rasterize_overdraw {
                self.render_clipped_polygon::<true>(&points, color_u32);
            } else {
                self.render_clipped_polygon::<false>(&points, color_u32);
            }
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
            let light_diffuse = Vec3f::new(0.30, 0.47, 0.80)
                .normalized()
                .dot(polygon.plane.normal)
                .abs()
                .min(0.99);

            // Add ambient)
            let light = light_diffuse * 0.9 + 0.09;

            // Calculate color, based on material and light
            let color = [
                (material.color.r as f32 * light) as u8,
                (material.color.g as f32 * light) as u8,
                (material.color.b as f32 * light) as u8,
            ];

            self.render_polygon_frame(&polygon, color);
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

    /// Display ALL level volumes
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

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    let do_cache_maps = true;

    // Load map
    let map = {
        // yay, this code will not compile on non-local builds)))
        // --
        let map_name = "e1m1";
        let data_path = "temp/";

        let wbsp_path = format!("{}wbsp/{}.wbsp", data_path, map_name);
        let map_path = format!("{}{}.map", data_path, map_name);

        if do_cache_maps {
            match std::fs::File::open(&wbsp_path) {
                Ok(mut bsp_file) => {
                    map::Map::load(&mut bsp_file).unwrap()
                }
                Err(_) => {
                    let source = std::fs::read_to_string(&map_path).unwrap();
    
                    let location_map = map::builder::Map::parse(&source).unwrap();
    
                    let compiled_map = map::builder::build(&location_map);
    
                    if let Ok(mut file) = std::fs::File::create(&wbsp_path) {
                        compiled_map.save(&mut file).unwrap();
                    }
            
                    compiled_map
                }
            }
        } else {
            let source = std::fs::read_to_string(&map_path).unwrap();
            let location_map = map::builder::Map::parse(&source).unwrap();
            map::builder::build(&location_map)
        }
    };

    // Setup render
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("WEIRD-2", 1280, 800)
        .build()
        .unwrap()
    ;

    let mut timer = Timer::new();
    let mut input = Input::new();
    let mut camera = Camera::new();

    camera.location = Vec3f::new(-174.0, 2114.6, -64.5);
    camera.direction = Vec3f::new(-0.4, 0.9, 0.1);
    // camera.location = Vec3f::new(-200.0, 2000.0, -50.0);
    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // Synchronize visible-set-building and projection cameras
    let mut do_sync_logical_camera = true;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut do_enable_overdraw_rendering = false;

    // Camera used for visible set building
    let mut logical_camera = camera;

    // Buffer that contains software-rendered pixels
    let mut frame_buffer = Vec::<u32>::new();

    'main_loop: loop {
        input.release_changed();

        while let Some(event) = event_pump.poll_event() {
            match event {
                sdl2::event::Event::Quit { .. } => {
                    break 'main_loop;
                }
                sdl2::event::Event::KeyUp { scancode, .. } => {
                    if let Some(code) = scancode {
                        input.on_state_changed(code, false);
                    }
                }
                sdl2::event::Event::KeyDown { scancode, .. } => {
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
        if input.is_key_clicked(Scancode::Num9) {
            do_sync_logical_camera = !do_sync_logical_camera;
        }

        if input.is_key_clicked(Scancode::Num0) {
            do_enable_overdraw_rendering = !do_enable_overdraw_rendering;
        }

        if do_sync_logical_camera {
            logical_camera = camera;
        }

        // Display some statistics
        if timer.get_frame_count() % 100 == 0 {
            // println!("{:?} {:?}", camera.location, camera.direction);

            println!("FPS: {}, SL={}, OV={}",
                timer.get_fps(),
                do_sync_logical_camera as u32,
                do_enable_overdraw_rendering as u32,
            );
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        let (frame_width, frame_height) = (window_width / 4, window_height / 4);

        // Compute view matrix
        let view_matrix = Mat4f::view(
            camera.location,
            camera.location + camera.direction,
            Vec3f::new(0.0, 0.0, 1.0)
        );

        let (aspect_x, aspect_y) = if window_width > window_height {
            (window_width as f32 / window_height as f32, 1.0)
        } else {
            (1.0, window_height as f32 / window_width as f32)
        };

        // Compute projection matrix
        let projection_matrix = Mat4f::projection_frustum(
            -0.5 * aspect_x, 0.5 * aspect_x,
            -0.5 * aspect_y, 0.5 * aspect_y,
            0.66, 100.0
        );

        let view_projection_matrix = view_matrix * projection_matrix;

        // Resize frame buffer to fit window's size
        frame_buffer.resize(frame_width * frame_height, 0);

        let mut render_context = RenderContext {
            camera_location: logical_camera.location,
            camera_plane: geom::Plane::from_point_normal(
                logical_camera.location,
                logical_camera.direction
            ),
            view_projection_matrix,
            map: &map,

            frame_pixels: frame_buffer.as_mut_ptr(),
            frame_width: frame_width,
            frame_height: frame_height,
            frame_stride: frame_width,
            do_rasterize_overdraw: do_enable_overdraw_rendering,
        };

        'blit_surface: {
            let mut window_surface = match window.surface(&event_pump) {
                Ok(window_surface) => window_surface,
                Err(err) => {
                    eprintln!("Cannot get window surface: {}", err);
                    break 'blit_surface;
                }
            };

            let mut render_surface = match sdl2::surface::Surface::from_data(
                bytemuck::cast_slice_mut::<u32, u8>(frame_buffer.as_mut_slice()),
                frame_width as u32,
                frame_height as u32,
                frame_width as u32 * 4,
                sdl2::pixels::PixelFormatEnum::ABGR8888
            ) {
                Ok(surface) => surface,
                Err(err) => {
                    eprintln!("Source surface create error: {}", err);
                    break 'blit_surface;
                }
            };

            // Disable alpha blending
            if let Err(err) = render_surface.set_blend_mode(sdl2::render::BlendMode::None) {
                eprintln!("Cannot disable render surface blending: {}", err);
            };

            // Perform render surface blit
            let src_rect = sdl2::rect::Rect::new(0, 0, frame_width as u32, frame_height as u32);
            let dst_rect = sdl2::rect::Rect::new(0, 0, window_width as u32, window_height as u32);

            if let Err(err) = render_surface.blit_scaled(src_rect, &mut window_surface, dst_rect) {
                eprintln!("Surface blit failed: {}", err);
            }

            if let Err(err) = window_surface.update_window() {
                eprintln!("Window update failed: {}", err);
            }
        }

        frame_buffer.fill(0);

        let start_volume_index_opt = map
            .get_bsp()
            .traverse(logical_camera.location);

        // render_context.test_software_rasterizer(timer.get_time());

        if let Some(start_volume_index) = start_volume_index_opt {
            render_context.render(start_volume_index);
        } else {
            render_context.render_all();
        }
    }
}

// main.rs
