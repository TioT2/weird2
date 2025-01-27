use std::collections::{BTreeSet, HashMap};

use math::{Mat4f, Vec3f};
use sdl2::keyboard::Scancode;

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

            if curr.z > 1.0 - geom::GEOM_EPSILON {
                result.push(curr);

                if next.z < 1.0 + geom::GEOM_EPSILON {
                    let t = (1.0 - curr.z) / (next.z - curr.z);

                    result.push((next - curr) * t + curr);
                }
            } else if next.z > 1.0 - geom::GEOM_EPSILON {
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

    /// Current time
    time: f32,

    /// Frame pixel array pointer
    frame_pixels: *mut u32,

    /// Frame width
    frame_width: usize,

    /// Frame height
    frame_height: usize,

    /// Frame stride
    frame_stride: usize,
}

impl<'t> RenderContext<'t> {
    /// Just test rendering function
    pub fn test_software_rasterizer(&mut self) {
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

        let rotm = math::Mat4f::rotate_z(self.time / 5.0);

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
            self.render_clipped_polygon(&set1.map(|v| rotm.transform_vector(v) * ev + cv), 0xFF0000);
            self.render_clipped_polygon(&set2.map(|v| rotm.transform_vector(v) * ev + cv), 0x0000FF);
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

            if curr.x > OFFSET - geom::GEOM_EPSILON {
                result.push(curr);

                if next.x < OFFSET + geom::GEOM_EPSILON {
                    let t = (OFFSET - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x > OFFSET - geom::GEOM_EPSILON {
                let t = (OFFSET - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(&mut points, &mut result);
        
        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y > OFFSET - geom::GEOM_EPSILON {
                result.push(curr);

                if next.y < OFFSET + geom::GEOM_EPSILON {
                    let t = (OFFSET - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y > OFFSET - geom::GEOM_EPSILON {
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

            if curr.x < frame_w - OFFSET + geom::GEOM_EPSILON {
                result.push(curr);

                if next.x > frame_w - OFFSET - geom::GEOM_EPSILON {
                    let t = (frame_w - OFFSET - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x < frame_w - OFFSET + geom::GEOM_EPSILON {
                let t = (frame_w - OFFSET - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(&mut points, &mut result);

        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y < frame_h - OFFSET + geom::GEOM_EPSILON {
                result.push(curr);

                if next.y > frame_h - OFFSET - geom::GEOM_EPSILON {
                    let t = (frame_h - OFFSET - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y < frame_h - OFFSET + geom::GEOM_EPSILON {
                let t = (frame_h - OFFSET - curr.y) / (next.y - curr.y);

                result.push((next - curr) * t + curr);
            }
        }

        result
    }

    unsafe fn render_clipped_polygon(&mut self, points: &[Vec3f], color: u32) {
        let (start_index, end_index) = {
            let mut min_y_index = !0usize;
            let mut max_y_index = !0usize;

            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for index in 0..points.len() {
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

            (min_y_index, max_y_index)
        };

        // Visible polygon should be ccw-oriented in screen space
        // So left index is changed by +1, and right - by -1

        unsafe {
            let line_end = points[end_index].y.ceil().to_int_unchecked::<usize>();

            let mut line = points[start_index].y.ceil().to_int_unchecked::<usize>();

            let mut left_index = start_index;
            let mut left_point = points[start_index];
            let mut left_end_line = line;
            let mut left_x = 0.0;
            let mut left_dx = 0.0;

            let mut right_index = start_index;
            let mut right_point = points[start_index];
            let mut right_end_line = line;
            let mut right_dx = 0.0;
            let mut right_x = 0.0;

            while line <= line_end {
                if line >= left_end_line {
                    left_index = (left_index + points.len() + 1) % points.len();

                    let new_left_point = points[left_index];

                    left_dx = (new_left_point.x - left_point.x) / (new_left_point.y - left_point.y);
                    left_end_line = new_left_point.y.ceil().to_int_unchecked::<usize>();
                    left_x = left_point.x - left_dx;
                    left_point = new_left_point;
                }

                if line >= right_end_line {
                    right_index = (right_index + points.len() - 1) % points.len();

                    let new_right_point = points[right_index];
                    
                    right_dx = (new_right_point.x - right_point.x) / (new_right_point.y - right_point.y);
                    right_end_line = new_right_point.y.ceil().to_int_unchecked::<usize>();
                    right_x = right_point.x - right_dx;
                    right_point = new_right_point;
                }

                left_x += left_dx;
                right_x += right_dx;

                let (start, end) = (
                    left_x.to_int_unchecked::<usize>(),
                    right_x.to_int_unchecked::<usize>()
                );

                let mut pixel_ptr = self.frame_pixels.add(self.frame_stride * line + start);
                let pixel_ptr_end = self.frame_pixels.add(self.frame_stride * line + end);

                while pixel_ptr <= pixel_ptr_end {
                    *pixel_ptr = color;
                    pixel_ptr = pixel_ptr.add(1);
                }

                line += 1;
            }
        }
    }

    fn render_polygon_frame(&mut self, polygon: &geom::Polygon, color: [u8; 3]) {
        // This rasterization process uses OpenGL as backend,
        // I guess, rasterizer will be moved to separate object

        let color_u32 = u32::from_le_bytes([color[2], color[1], color[0], 0]);

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
            self.render_clipped_polygon(&points, color_u32);
        }
    }

    /// Render single polygon with flat fill
    fn render_polygon_gl(&mut self, polygon: &geom::Polygon, color: [u8; 3]) {
        // This rasterization process uses OpenGL as backend,
        // I guess, rasterizer will be moved to separate object

        let proj_polygon = geom::Polygon::from_ccw(polygon
            .points
            .iter()
            .map(|point| self.view_projection_matrix.transform_point(*point))
            .collect::<Vec<_>>()
        );

        // Clip polygon back in viewspace
        let clip_polygon = proj_polygon.clip_viewspace_back();

        unsafe {
            glu_sys::glColor3ub(color[0], color[1], color[2]);

            glu_sys::glBegin(glu_sys::GL_POLYGON);
            for point in &clip_polygon.points {
                glu_sys::glVertex2f(point.x / point.z, point.y / point.z);
            }
            glu_sys::glEnd();
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
            // self.render_polygon_gl(&polygon, color);
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

    // yay, this code will not compile on non-local builds)))
    // --
    let location_map = map::builder::Map::parse(include_str!("../temp/e1m1.map")).unwrap();

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
    // Disable vsync
    _ = video.gl_set_swap_interval(0);
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
        if timer.get_frame_count() % 100 == 0 {
            println!("FPS: {}, DB={}, SL={}, SR={}",
                timer.get_fps(),
                do_enable_depth_test as u32,
                do_sync_logical_camera as u32,
                do_enable_slow_rendering as u32,
            );
        }

        // Compute view matrix
        let view_matrix = Mat4f::view(
            camera.location,
            camera.location + camera.direction,
            Vec3f::new(0.0, 0.0, 1.0)
        );

        // Compute projection matrix
        let projection_matrix = Mat4f::projection_frustum(
            -0.5 * 8.0 / 6.0, 0.5 * 8.0 / 6.0,
            -0.5, 0.5,
            0.66, 100.0
        );
        
        let view_projection_matrix = view_matrix * projection_matrix;

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };
        
        // Resize frame buffer to fit window's size
        frame_buffer.resize(window_width * window_height, 0);

        let mut render_context = RenderContext {
            camera_location: logical_camera.location,
            camera_plane: geom::Plane::from_point_normal(
                logical_camera.location,
                logical_camera.direction
            ),
            view_projection_matrix,
            map: &compiled_map,

            time: timer.get_time(),

            frame_pixels: frame_buffer.as_mut_ptr(),
            frame_width: window_width,
            frame_height: window_height,
            frame_stride: window_width,
        };

        // Render frame by gl functions
        unsafe {
            if do_enable_depth_test {
                glu_sys::glEnable(glu_sys::GL_DEPTH_TEST);
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT | glu_sys::GL_DEPTH_BUFFER_BIT);
            } else {
                glu_sys::glDisable(glu_sys::GL_DEPTH_TEST);
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT);
            }

            glu_sys::glLoadIdentity();

            glu_sys::glViewport(0, 0, window_width as i32, window_height as i32);

            glu_sys::glRasterPos2f(-1.0, 1.0);
            glu_sys::glPixelZoom(1.0, -1.0);
            glu_sys::glDrawPixels(
                window_width as i32,
                window_height as i32,
                glu_sys::GL_BGRA,
                glu_sys::GL_UNSIGNED_BYTE,
                frame_buffer.as_ptr() as *const std::ffi::c_void,
            );

            // glu_sys::glViewport(0, 0, window_width as i32 / 2, window_height as i32);
            // glu_sys::glTranslatef(1.0, 0.0, 0.0);
            // glu_sys::glScalef(2.0, 1.0, 1.0);
            // glu_sys::glColor3f(0.30, 0.47, 0.80);
            // glu_sys::glBegin(glu_sys::GL_POLYGON);
            // glu_sys::glVertex2f(-1.0, -1.0);
            // glu_sys::glVertex2f(-1.0, 1.0);
            // glu_sys::glVertex2f(0.0, 1.0);
            // glu_sys::glVertex2f(0.0, -1.0);
            // glu_sys::glEnd();
        }

        frame_buffer.fill(0);

        // render_context.test_software_rasterizer();

        let start_volume_index_opt = compiled_map
            .get_bsp()
            .traverse(logical_camera.location);

        if let Some(start_volume_index) = start_volume_index_opt {
            render_context.render(start_volume_index);
        } else {
            render_context.render_all();
        }

        unsafe {
            glu_sys::glFinish();
        }

        window.gl_swap_window();
    }
}

// main.rs
