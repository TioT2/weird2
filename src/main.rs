/// Main project module

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Visible and Physical BSP building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with location BSP's.

use std::{collections::HashMap, sync::Arc};
use math::{Mat4f, Vec2f, Vec3f, Vec5UVf};
use sdl2::keyboard::Scancode;

/// Basic math utility
pub mod math;

/// Arena
// pub mod arena;

/// System font
pub mod system_font;

/// Random number generator
pub mod rand;

/// Basic geometry
pub mod geom;

/// Compiled map implementation
pub mod bsp;

/// Map format implementation
pub mod map;

/// Resource pack
pub mod res;

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

/// Physical BSP type
#[derive(Copy, Clone)]
pub enum PhysicalBspType {
    Partition,

    /// Empty leaf
    Empty,

    /// Solid leaf
    Solid,
}

pub struct PhysicalBspElement {
    pub plane: geom::Plane,
    pub front_index: u32,
    pub back_index: u32,
    pub front_ty: PhysicalBspType,
    pub back_ty: PhysicalBspType,
}

pub struct Polyhedron {
    pub points: Vec<Vec3f>,
    pub index_set: Vec<Option<std::num::NonZeroU32>>,
}

pub struct PhysicalBsp {
    elements: Vec<PhysicalBspElement>,
}

impl PhysicalBsp {
    fn point_is_solid_impl(&self, point: Vec3f, element: &PhysicalBspElement) -> bool {
        let point_relation = element.plane.get_point_relation(point);

        let (ty, index) = match point_relation {
            geom::PointRelation::Front | geom::PointRelation::OnPlane => {
                (element.front_ty, element.front_index)
            }
            geom::PointRelation::Back => {
                (element.back_ty, element.back_index)
            }
        };

        match ty {
            PhysicalBspType::Solid => true,
            PhysicalBspType::Empty => false,
            PhysicalBspType::Partition => {
                let next_element = self.elements.get(index as usize).unwrap();

                self.point_is_solid_impl(point, next_element)
            }
        }
    }

    /// Check if SolidBsp point is solid, actually
    pub fn point_is_solid(&self, point: Vec3f) -> bool {
        let Some(start) = self.elements.get(0) else {
            return true;
        };

        self.point_is_solid_impl(point, start)
    }
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

/// Different rasterization modes
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum RasterizationMode {
    /// Solid color
    Standard = 0,

    /// Overdraw
    Overdraw = 1,

    /// Inverse depth
    Depth = 2,

    /// Checker texture
    UV = 3,

    /// Textuers, actually
    Textured = 4,
}

pub const fn u64_from_u16(value: [u16; 4]) -> u64 {
    ((value[0] as u64) <<  0) |
    ((value[1] as u64) << 16) |
    ((value[2] as u64) << 32) |
    ((value[3] as u64) << 48)
}

pub const fn u64_into_u16(value: u64) -> [u16; 4] {
    [
        ((value >>  0) & 0xFFFF) as u16,
        ((value >> 16) & 0xFFFF) as u16,
        ((value >> 32) & 0xFFFF) as u16,
        ((value >> 48) & 0xFFFF) as u16,
    ]
}

impl RasterizationMode {
    // Rasterization mode count
    const COUNT: u32 = 5;

    /// Build rasterization mode from u32
    pub const fn from_u32(n: u32) -> Option<RasterizationMode> {
        Some(match n {
            0 => RasterizationMode::Standard,
            1 => RasterizationMode::Overdraw,
            2 => RasterizationMode::Depth,
            3 => RasterizationMode::UV,
            4 => RasterizationMode::Textured,
            _ => return None,
        })
    }

    /// Get next rasterization mode
    pub const fn next(self) -> RasterizationMode {
        Self::from_u32((self as u32 + 1) % Self::COUNT).unwrap()
    }
}

/// Final surface texture
#[derive(Copy, Clone)]
struct SurfaceTexture<'t> {
    width: usize,
    height: usize,
    stride: usize,
    data: &'t [u64],
}

impl<'t> SurfaceTexture<'t> {
    // Generate empty surface texture
    pub fn empty() -> SurfaceTexture<'static> {
        SurfaceTexture {
            width: 0,
            height: 0,
            stride: 0,
            data: &[],
        }
    }
}

/// Render context
struct RenderContext<'t, 'ref_table> {
    /// Camera location
    camera_location: Vec3f,

    /// Offset of background from foreground
    sky_background_uv_offset: Vec2f,

    /// Foreground offset
    sky_uv_offset: Vec2f,

    /// VP matrix
    view_projection_matrix: Mat4f,

    /// Map reference
    map: &'t bsp::Map,

    material_table: &'t res::MaterialReferenceTable<'ref_table>,

    /// Frame pixel array pointer
    frame_pixels: *mut u64,

    /// Frame width
    frame_width: usize,

    /// Frame height
    frame_height: usize,

    /// Frame stride
    frame_stride: usize,

    /// Rasterization mode
    rasterization_mode: RasterizationMode,
}

impl<'t, 'ref_table> RenderContext<'t, 'ref_table> {
    fn clip_viewspace_back(&self, points: &mut Vec<Vec5UVf>, result: &mut Vec<Vec5UVf>) {
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

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
    }

    /// Clip polygon by octagon
    fn clip_polygon_oct(&self, points: &mut Vec<Vec5UVf>, result: &mut Vec<Vec5UVf>, clip_oct: &geom::ClipOct) {
        macro_rules! clip_edge {
            ($metric: ident, $clip_val: expr, $cmp: tt) => {
                for index in 0..points.len() {
                    let curr = points[index];
                    let next = points[(index + 1) % points.len()];

                    if $metric!(curr) $cmp $clip_val {
                        result.push(curr);

                        if !($metric!(next) $cmp $clip_val) {
                            let t = ($clip_val - $metric!(curr)) / ($metric!(next) - $metric!(curr));
                            result.push((next - curr) * t + curr);
                        }
                    } else if $metric!(next) $cmp $clip_val {
                        let t = ($clip_val - $metric!(curr)) / ($metric!(next) - $metric!(curr));
                        result.push((next - curr) * t + curr);
                    }
                }
            }
        }

        macro_rules! clip_metric_minmax {
            ($metric: ident, $min: expr, $max: expr) => {
                result.clear();
                clip_edge!($metric, $min, >=);
                std::mem::swap(points, result);
        
                result.clear();
                clip_edge!($metric, $max, <=);
                std::mem::swap(points, result);
            };
        }

        macro_rules! metric_x { ($e: ident) => { ($e.x) } }
        macro_rules! metric_y { ($e: ident) => { ($e.y) } }
        macro_rules! metric_y_a_x { ($e: ident) => { ($e.y + $e.x) } }
        macro_rules! metric_y_s_x { ($e: ident) => { ($e.y - $e.x) } }

        // Clip by X
        clip_metric_minmax!(metric_x, clip_oct.min_x, clip_oct.max_x);

        // Clip by Y
        clip_metric_minmax!(metric_y, clip_oct.min_y, clip_oct.max_y);

        // Clip by Y+X
        clip_metric_minmax!(metric_y_a_x, clip_oct.min_y_a_x, clip_oct.max_y_a_x);

        // Clip by Y-X
        clip_metric_minmax!(metric_y_s_x, clip_oct.min_y_s_x, clip_oct.max_y_s_x);

        // Swap results (again)
        std::mem::swap(points, result);
    }

    /// Project polygon on screen
    pub fn get_screenspace_portal_polygon(&self, points: &mut Vec<Vec3f>, point_dst: &mut Vec<Vec3f>) {
        // Check if points need reprojection
        // Distance of sky from camera

        point_dst.clear();
        for point in points.iter() {
            point_dst.push(self.view_projection_matrix.transform_point(*point));
        }
        std::mem::swap(points, point_dst);

        // Clip polygon invisible part
        point_dst.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.z > 1.0 {
                point_dst.push(curr);

                if next.z < 1.0 {
                    let t = (1.0 - curr.z) / (next.z - curr.z);

                    point_dst.push((next - curr) * t + curr);
                }
            } else if next.z > 1.0 {
                let t = (1.0 - curr.z) / (next.z - curr.z);

                point_dst.push((next - curr) * t + curr);
            }
        }
        std::mem::swap(points, point_dst);

        // Calculate screen-space points
        let width = self.frame_width as f32 * 0.5;
        let height = self.frame_height as f32 * 0.5;

        point_dst.clear();
        for pt in points.iter() {
            let inv_z = pt.z.recip();

            point_dst.push(Vec3f::new(
                (1.0 + pt.x * inv_z) * width,
                (1.0 - pt.y * inv_z) * height,
                inv_z,
            ));
        }
    }

    /// Apply projection to sky polygon
    pub fn get_projected_sky_polygon(
        &self,
        polygon: &geom::Polygon,
        point_dst: &mut Vec<Vec5UVf>
    ) {
        for point in polygon.points.iter() {
            // Skyplane distance
            const DIST: f32 = 128.0;

            let rp = *point - self.camera_location;

            // Copy z sign to correct back-z case
            let s_dist = DIST.copysign(rp.z);

            let u = rp.x / rp.z * s_dist;
            let v = rp.y / rp.z * s_dist;

            point_dst.push(Vec5UVf::from_32(
                // Vector-only transform is used here to don't add camera location back.
                self.view_projection_matrix.transform_vector(Vec3f::new(u, v, s_dist)),
                Vec2f::new(u + self.sky_uv_offset.x, v + self.sky_uv_offset.y)
            ));
        }
    }

    /// Apply projection to default polygon
    pub fn get_projected_polygon(
        &self,
        polygon: &geom::Polygon,
        material_u: geom::Plane,
        material_v: geom::Plane,
        point_dst: &mut Vec<Vec5UVf>,
    ) {
        point_dst.clear();
        for point in polygon.points.iter() {
            point_dst.push(Vec5UVf::from_32(
                self.view_projection_matrix.transform_point(*point),
                Vec2f::new(
                    point.dot(material_u.normal) + material_u.distance,
                    point.dot(material_v.normal) + material_v.distance,
                )
            ));
        }
    }

    /// Render already clipped polygon
    unsafe fn render_clipped_polygon(
        &mut self,
        is_transparent: bool,
        is_sky: bool,
        points: &[Vec5UVf],
        color: u64,
        texture: SurfaceTexture
    ) {
        // Potential rasterization function set
        let rasterize_fn = [
            Self::render_clipped_polygon_impl::<0, false, false>,
            Self::render_clipped_polygon_impl::<1, false, false>,
            Self::render_clipped_polygon_impl::<2, false, false>,
            Self::render_clipped_polygon_impl::<3, false, false>,
            Self::render_clipped_polygon_impl::<4, false, false>,

            Self::render_clipped_polygon_impl::<0, true, false>,
            Self::render_clipped_polygon_impl::<1, true, false>,
            Self::render_clipped_polygon_impl::<2, true, false>,
            Self::render_clipped_polygon_impl::<3, true, false>,
            Self::render_clipped_polygon_impl::<4, true, false>,

            Self::render_clipped_polygon_impl::<0, false, true>,
            Self::render_clipped_polygon_impl::<1, false, true>,
            Self::render_clipped_polygon_impl::<2, false, true>,
            Self::render_clipped_polygon_impl::<3, false, true>,
            Self::render_clipped_polygon_impl::<4, false, true>,

            Self::render_clipped_polygon_impl::<0, true, true>,
            Self::render_clipped_polygon_impl::<1, true, true>,
            Self::render_clipped_polygon_impl::<2, true, true>,
            Self::render_clipped_polygon_impl::<3, true, true>,
            Self::render_clipped_polygon_impl::<4, true, true>,
        ];

        // Generate rendering function id
        let id = 0
            + is_sky as usize * 10
            + is_transparent as usize * 5
            + self.rasterization_mode as usize;

        rasterize_fn[id](self, points, color, texture);
    }

    /// render_clipped_polygon function optimized implementation
    unsafe fn render_clipped_polygon_impl<
        const MODE: u32,
        const IS_TRANSPARENT: bool,
        const IS_SKY: bool,
    >(
        &mut self,
        points: &[Vec5UVf],
        color: u64,
        texture: SurfaceTexture,
    ) {
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

        // Find polygon min/max
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
        const DY_EPSILON: f32 = 0.01;

        // Calculate polygon bounds
        let first_line = usize::min(min_y_value.floor() as usize, self.frame_height);
        let last_line = usize::min(max_y_value.ceil() as usize, self.frame_height);

        let mut left_index = ind_next!(min_y_index);

        let mut left_prev_xzuv: math::FVec4 = points[min_y_index].xzuv().into();
        let mut left_curr_xzuv: math::FVec4 = points[left_index].xzuv().into();
        let mut left_prev_y = points[min_y_index].y;
        let mut left_curr_y = points[left_index].y;
        let mut left_slope_xzuv: math::FVec4 = (left_curr_xzuv - left_prev_xzuv) / (left_curr_y - left_prev_y);

        let mut right_index = ind_prev!(min_y_index);

        let mut right_prev_xzuv: math::FVec4 = points[min_y_index].xzuv().into();
        let mut right_curr_xzuv: math::FVec4 = points[right_index].xzuv().into();
        let mut right_prev_y = points[min_y_index].y;
        let mut right_curr_y = points[right_index].y;
        let mut right_slope_xzuv: math::FVec4 = (right_curr_xzuv - right_prev_xzuv) / (right_curr_y - right_prev_y);

        let width = texture.width as isize >> (IS_SKY as isize);
        let height = texture.height as isize;

        // Scan for lines
        'line_loop: for pixel_y in first_line..last_line {
            // Get current pixel y
            let y = pixel_y as f32 + 0.5;

            while y > left_curr_y {
                if left_index == max_y_index {
                    break 'line_loop;
                }

                left_index = ind_next!(left_index);

                left_prev_y = left_curr_y;
                left_prev_xzuv = left_curr_xzuv;

                left_curr_y = points[left_index].y;
                left_curr_xzuv = points[left_index].xzuv().into();

                let left_dy = left_curr_y - left_prev_y;

                // Check if edge is flat
                if left_dy <= DY_EPSILON {
                    left_slope_xzuv = math::FVec4::zero();
                } else {
                    left_slope_xzuv = (left_curr_xzuv - left_prev_xzuv) / left_dy;
                }
            }

            while y > right_curr_y {
                if right_index == max_y_index {
                    break 'line_loop;
                }

                right_index = ind_prev!(right_index);

                right_prev_y = right_curr_y;
                right_prev_xzuv = right_curr_xzuv;

                right_curr_y = points[right_index].y;
                right_curr_xzuv = points[right_index].xzuv().into();

                let right_dy = right_curr_y - right_prev_y;

                // Check if edge is flat
                if right_dy <= DY_EPSILON {
                    right_slope_xzuv = math::FVec4::zero();
                } else {
                    right_slope_xzuv = (right_curr_xzuv - right_prev_xzuv) / right_dy;
                }
            }

            let left_xzuv = math::FVec4::mul_add(
                left_slope_xzuv,
                math::FVec4::from_single(y - left_prev_y),
                left_prev_xzuv,
            );
            let right_xzuv = math::FVec4::mul_add(
                right_slope_xzuv,
                math::FVec4::from_single(y - right_prev_y),
                right_prev_xzuv,
            );

            // Get left x
            let left_x = left_xzuv.x();

            // Get right x
            let right_x = right_xzuv.x();

            // Calculate hline start/end, clip it
            let start = left_x.floor() as usize;
            let end = usize::min(right_x.floor() as usize, self.frame_width);

            let pixel_row = self.frame_pixels.add(self.frame_stride * pixel_y);

            let dx = right_x - left_x;

            let slope_xzuv = if dx <= DY_EPSILON {
                math::FVec4::zero()
            } else {
                (right_xzuv - left_xzuv) / (right_x - left_x)
            };

            // Calculate pixel position 'remainder'
            let pixel_off = left_x.fract() - 0.5;

            for pixel_x in start..end {
                let xzuv = math::FVec4::mul_add(
                    slope_xzuv,
                    math::FVec4::from_single(pixel_x.wrapping_sub(start) as f32 - pixel_off),
                    left_xzuv
                );
                let pixel_ptr = pixel_row.wrapping_add(pixel_x);

                let src_color;

                // Handle different rasterization modes
                if MODE == RasterizationMode::Standard as u32 {
                    src_color = color;
                } else if MODE == RasterizationMode::Overdraw as u32 {
                    src_color = 0x0010_0010_0010u64.wrapping_add(*pixel_ptr);
                } else if MODE == RasterizationMode::Depth as u32 {
                    let color = (xzuv.y() * 25500.0) as u16;
                    src_color = u64_from_u16([color; 4]);
                } else if MODE == RasterizationMode::UV as u32 {
                    let z = xzuv.y().recip();
                    let u = xzuv.z() * z;
                    let v = xzuv.w() * z;

                    let xi = (u as i64 & 0xFF) as u8;
                    let yi = (v as i64 & 0xFF) as u8;

                    src_color = color * (((xi >> 5) ^ (yi >> 5)) & 1) as u64;
                } else if MODE == RasterizationMode::Textured as u32 {
                    if IS_SKY {
                        let z = xzuv.y().recip();
                        let u = xzuv.z() * z;
                        let v = xzuv.w() * z;

                        let fg_u: usize = std::mem::transmute(u
                            .to_int_unchecked::<isize>()
                            .rem_euclid(width)
                        );

                        let fg_v: usize = std::mem::transmute(v
                            .to_int_unchecked::<isize>()
                            .rem_euclid(height)
                        );

                        let fg_color = *texture.data.get_unchecked(fg_v * texture.stride + fg_u);

                        // Check foreground color and fetch backround if foreground is transparent
                        if fg_color == 0 {
                            let bg_u = std::mem::transmute::<_, usize>(
                                (u + self.sky_background_uv_offset.x)
                                    .to_int_unchecked::<isize>()
                                    .rem_euclid(width)
                                    .wrapping_add(width)
                            );

                            let bg_v = std::mem::transmute::<_, usize>(
                                (v + self.sky_background_uv_offset.y)
                                    .to_int_unchecked::<isize>()
                                    .rem_euclid(height)
                            );

                            src_color = *texture.data.get_unchecked(
                                bg_v
                                    .wrapping_mul(texture.stride)
                                    .wrapping_add(bg_u)
                            );
                        } else {
                            src_color = fg_color;
                        }
                    } else {
                        let z = xzuv.y().recip();

                        let u: usize = std::mem::transmute(
                            (xzuv.z() * z)
                                .to_int_unchecked::<isize>()
                                .rem_euclid(width)
                        );

                        let v: usize = std::mem::transmute(
                            (xzuv.w() * z)
                                .to_int_unchecked::<isize>()
                                .rem_euclid(height)
                        );

                        src_color = *texture.data.get_unchecked(v * texture.stride + u);
                    }

                } else {
                    panic!("Unknown rasterization mode: {}", MODE);
                }

                if IS_TRANSPARENT {
                    let dst_color = *pixel_ptr;
                    let [dr, dg, db, _] = u64_into_u16(dst_color);
                    let [sr, sg, sb, _] = u64_into_u16(src_color);
                
                    // SSE-based transparency calculation
                    #[cfg(target_feature = "sse")]
                    unsafe {
                        let dst = std::arch::x86_64::_mm_set_epi32(
                            0,
                            db as i32,
                            dg as i32,
                            dr as i32,
                        );
                        let src = std::arch::x86_64::_mm_set_epi32(
                            0,
                            sb as i32,
                           sg as i32,
                           sr as i32,
                        );
                        let dst_m = std::arch::x86_64::_mm_mul_ps(
                            std::arch::x86_64::_mm_cvtepi32_ps(dst),
                            std::arch::x86_64::_mm_set1_ps(0.4)
                        );
                        let src_m = std::arch::x86_64::_mm_mul_ps(
                            std::arch::x86_64::_mm_cvtepi32_ps(src),
                            std::arch::x86_64::_mm_set1_ps(0.6)
                        );
                        let sum = std::arch::x86_64::_mm_add_ps(src_m, dst_m);
                        let res = std::arch::x86_64::_mm_cvtps_epi32(sum);
                        let [rr, rg, rb, _]: [u32; 4] = std::mem::transmute(res);

                        *pixel_ptr = u64_from_u16([
                            rr as u16,
                            rg as u16,
                            rb as u16,
                            0
                        ]);
                    }
                    
                    #[cfg(not(target_feature = "sse"))]
                    {
                        *pixel_ptr = u64_from_u16([
                            (sr as f32 * 0.6 + dr as f32 * 0.4) as u16,
                            (sg as f32 * 0.6 + dg as f32 * 0.4) as u16,
                            (sb as f32 * 0.6 + db as f32 * 0.4) as u16,
                            0
                        ]);
                    }

                } else {
                    *pixel_ptr = src_color;
                }
            }
        }
    }

    /// Render polygon
    fn render_polygon(
        &mut self,
        polygon: &geom::Polygon,
        material_u: geom::Plane,
        material_v: geom::Plane,
        color: u64,
        texture: SurfaceTexture,
        is_transparent: bool,
        is_sky: bool,
        clip_oct: &geom::ClipOct,
        points: &mut Vec<Vec5UVf>,
        point_dst: &mut Vec<Vec5UVf>,
    ) {
        points.clear();

        // Apply projection (and calculate UVs)
        if is_sky {
            self.get_projected_sky_polygon(polygon, points);
        } else {
            self.get_projected_polygon(polygon, material_u, material_v, points);
        }

        // Clip polygon invisible part
        point_dst.clear();
        self.clip_viewspace_back(points, point_dst);
        std::mem::swap(points, point_dst);

        // Calculate screen-space points
        let width = self.frame_width as f32 * 0.5;
        let height = self.frame_height as f32 * 0.5;

        point_dst.clear();
        for pt in points.iter() {
            let inv_z = pt.z.recip();

            point_dst.push(Vec5UVf::new(
                (1.0 + pt.x * inv_z) * width,
                (1.0 - pt.y * inv_z) * height,
                inv_z,
                pt.u * inv_z,
                pt.v * inv_z,
            ));
        }

        std::mem::swap(points, point_dst);

        // Clip polygon by volume clipping octagon
        point_dst.clear();
        self.clip_polygon_oct(points, point_dst, clip_oct);
        std::mem::swap(points, point_dst);

        // Just for safety
        for pt in points.iter() {
            if !pt.x.is_finite() || !pt.y.is_finite() {
                return;
            }
        }

        // Check if it is a polygon
        if points.len() < 3 {
            return;
        }

        // Rasterize polygon
        unsafe {
            self.render_clipped_polygon(
                is_transparent,
                is_sky,
                &points,
                color,
                texture
            );
        }
    }

    /// Render single volume
    fn render_volume(
        &mut self,
        id: bsp::VolumeId,
        clip_oct: &geom::ClipOct,
        points: &mut Vec<Vec5UVf>,
        point_dst: &mut Vec<Vec5UVf>,
        surface_texture_data: &mut Vec<u64>,
    ) {
        let volume = self.map.get_volume(id).unwrap();

        'polygon_rendering: for surface in volume.get_surfaces() {

            let polygon = self.map.get_polygon(surface.polygon_id).unwrap();

            // Perform backface culling
            if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                continue 'polygon_rendering;
            }

            let texture = self.material_table
                .get_texture(surface.material_id)
                .unwrap();

            // Find mip index (sky uses 0 by default)
            let mip_index = if surface.is_sky {
                0
            } else {
                let mut min_dist_2 = f32::MAX;
    
                for point in &polygon.points {
                    let dist_2 = (*point - self.camera_location).length2();
    
                    if dist_2 <= min_dist_2 {
                        min_dist_2 = dist_2;
                    }
                }

                let res = (self.frame_width * self.frame_height) as f32;
    
                // Strange, but...
                ((min_dist_2 / res).log2() / 2.0 + 1.3) as usize
            };

            let (image, image_uv_scale) = texture.get_mipmap(mip_index);

            // Calculate simple per-face diffuse light
            let light_diffuse = Vec3f::new(0.30, 0.47, 0.80)
                .normalized()
                .dot(polygon.plane.normal)
                .abs()
                .min(0.99);

            // Add ambiance)
            let light = light_diffuse * 0.9 + 0.09;

            let surface_texture = match self.rasterization_mode {
                RasterizationMode::Depth | RasterizationMode::Overdraw | RasterizationMode::UV | RasterizationMode::Standard => {
                    SurfaceTexture::empty()
                }
                RasterizationMode::Textured => {
                    build_surface_texture(
                        surface_texture_data,
                        image,
                        light,
                        !surface.is_sky
                    )
                }
            };

            // Get surface color
            let color: [u8; 4] = self
                .material_table
                .get_color(surface.material_id)
                .unwrap()
                .to_le_bytes();

            // Calculate color, based on material and light
            let color = u64_from_u16([
                (color[0] as f32 * light) as u16,
                (color[1] as f32 * light) as u16,
                (color[2] as f32 * light) as u16,
                0
            ]);

            self.render_polygon(
                &polygon,
                geom::Plane {
                    normal: surface.u.normal / image_uv_scale,
                    distance: surface.u.distance / image_uv_scale,
                },
                geom::Plane {
                    normal: surface.v.normal / image_uv_scale,
                    distance: surface.v.distance / image_uv_scale,
                },
                color,
        surface_texture,
                surface.is_transparent,
                surface.is_sky,
                clip_oct,
                points,
                point_dst,
            );
        }
    }

    /// Get main clipping rectangle
    fn get_screen_clip_rect(&self) -> geom::ClipRect {
        /// Clipping offset
        const CLIP_OFFSET: f32 = 0.2;

        // Surface clipping rectangle
        geom::ClipRect {
            min: Vec2f::new(
                CLIP_OFFSET,
                CLIP_OFFSET,
            ),
            max: Vec2f::new(
                self.frame_width as f32 + CLIP_OFFSET,
                self.frame_height as f32 + CLIP_OFFSET,
            ),
        }
    }

    
    /// Build set of rendered polygons
    /// 
    /// # Algorithm
    /// 
    /// This function recursively traverses BSP in front-to-back order,
    /// if current volume isn't inserted in PVS (Potentially Visible Set),
    /// then it's ignored. If it is, it is added with render set with it's
    /// current clipping rectangle (it **is not** visible from any of next
    /// traverse elements, so current clipping rectangle is the final one)
    /// and then for every portal of current volume function calculates
    /// it's screenspace bounding rectangle and inserts inserts destination
    /// volume in PVS with this rectangle. If destination volume is already
    /// added, it extends it to fit union of current and previous clipping
    /// rectangles.
    pub fn build_render_set(
        &self,
        bsp_root: &bsp::Bsp,
        start_volume_id: bsp::VolumeId,
        start_clip_oct: &geom::ClipOct
    ) -> Vec<(bsp::VolumeId, geom::ClipOct)> {
        // Potentially visible set
        let mut pvs = HashMap::<bsp::VolumeId, geom::ClipOct>::new();

        // Render set itself
        let mut inv_render_set = Vec::new();

        // Visit stack (I don't want to use recursion here)
        let mut visit_stack = Vec::<&bsp::Bsp>::new();

        // Initialize PVS
        pvs.insert(start_volume_id, *start_clip_oct);

        // Initialize visit stack
        visit_stack.push(bsp_root);

        while let Some(bsp) = visit_stack.pop() {
            match bsp {
                bsp::Bsp::Partition { splitter_plane, front, back } => {
                    let rel = splitter_plane.get_point_relation(self.camera_location);

                    let (first, second) = match rel {
                        geom::PointRelation::Front | geom::PointRelation::OnPlane => {
                            (front, back)
                        }
                        geom::PointRelation::Back => {
                            (back, front)
                        }
                    };
    
                    visit_stack.push(&second);
                    visit_stack.push(&first);
                }
                bsp::Bsp::Volume(volume_id) => 'volume_traverse: {
                    let Some(volume_clip_oct) = pvs.get(volume_id) else {
                        break 'volume_traverse;
                    };
                    let volume_clip_oct = *volume_clip_oct;
    
                    // Insert volume in render set
                    inv_render_set.push((*volume_id, volume_clip_oct));
    
                    let volume = self.map.get_volume(*volume_id).unwrap();
    
                    // 'sky_rendering: for surface in volume.get_surfaces() {
                    //     if !surface.is_sky {
                    //         continue 'sky_rendering;
                    //     }
                    //     *sky_clip_oct = sky_clip_oct.union(&volume_clip_oct);
                    // }
    
                    'portal_rendering: for portal in volume.get_portals() {
                        let portal_polygon = self.map
                            .get_polygon(portal.polygon_id)
                            .unwrap();
    
                        // Perform modified backface culling
                        let backface_cull_result =
                            (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                            >= 0.0;
                        if backface_cull_result != portal.is_facing_front {
                            continue 'portal_rendering;
                        }
    
                        let clip_rect = 'portal_validation: {
                            /*
                            I think, that projection is main source of the 'black bug'
                            is (kinda) infinite points during polygon projection process.
                            Possibly, it can be solved with early visible portion clipping
                            or automatic enabling of polygon based on it's relative
                            location from camera.
    
                            Reason: bug disappears if we just don't use projection at all.
                                (proved by distance clip fix)
    
                            TODO: add some distance coefficent
                            Constant works quite bad, because
                            on small-metric maps it allows too much,
                            and on huge ones it (theoretically) don't
                            saves us from the bug.
                            */
    
                            let portal_plane_distance = portal_polygon
                                .plane
                                .get_signed_distance(self.camera_location)
                                .abs();
    
                            // Do not calculate projection for near portals
                            if portal_plane_distance <= 8.0 {
                                break 'portal_validation volume_clip_oct;
                            }
    
                            let mut polygon_points = portal_polygon.points.clone();
                            let mut proj_polygon_points = Vec::new();
    
                            if !portal.is_facing_front {
                                polygon_points.reverse();
                            }
    
                            self.get_screenspace_portal_polygon(
                                &mut polygon_points,
                                &mut proj_polygon_points
                            );
    
                            // Check if it's even a polygon
                            if proj_polygon_points.len() < 3 {
                                continue 'portal_rendering;
                            }
    
                            let proj_oct = geom::ClipOct::from_points_xy(
                                proj_polygon_points.iter().copied()
                            );
    
                            let Some(clip_oct) = geom::ClipOct::intersection(
                                &volume_clip_oct,
                                &proj_oct.extend(0.1, 0.1, 0.1, 0.1)
                            ) else {
                                continue 'portal_rendering;
                            };
    
                            clip_oct
                        };
    
                        // Insert clipping rectangle in PVS
                        let pvs_entry = pvs
                            .entry(portal.dst_volume_id);
    
                        match pvs_entry {
                            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                                let existing_rect: &mut geom::ClipOct = occupied.get_mut();
                                *existing_rect = existing_rect.union(&clip_rect);
                            }
                            std::collections::hash_map::Entry::Vacant(vacant) => {
                                vacant.insert(clip_rect);
                            }
                        }
                    }
                }
                // Ignore it
                bsp::Bsp::Void => {}
            }
        }

        inv_render_set
    }

    /// Render scene starting from certain volume
    pub fn render(&mut self, start_volume_id: bsp::VolumeId) {
        let inv_render_set = self.build_render_set(
            self.map.get_world_model().get_bsp(),
            start_volume_id,
            &geom::ClipOct::from_clip_rect(self.get_screen_clip_rect())
        );
        let mut points = Vec::with_capacity(32);
        let mut point_dst = Vec::with_capacity(32);
        let mut surface_texture = Vec::new();

        for (volume_id, volume_clip_oct) in inv_render_set
            .iter()
            .rev()
            .copied()
        {
            self.render_volume(
                volume_id,
                &volume_clip_oct,
                &mut points,
                &mut point_dst,
                &mut surface_texture
            );
        }
    }

    /// Build correct volume set rendering order
    fn order_rendered_volume_set(
        bsp: &bsp::Bsp,
        render_set: &mut Vec<bsp::VolumeId>,
        camera_location: Vec3f
    ) {
        match bsp {

            // Handle space partition
            bsp::Bsp::Partition {
                splitter_plane,
                front,
                back
            } => {
                let (first, second) = match splitter_plane.get_point_relation(camera_location) {
                    geom::PointRelation::Front | geom::PointRelation::OnPlane => (back, front),
                    geom::PointRelation::Back => (front, back)
                };

                Self::order_rendered_volume_set(
                    first,
                    render_set,
                    camera_location
                );

                Self::order_rendered_volume_set(
                    &second,
                    render_set,
                    camera_location
                );
            }

            // Move volume to render set if it's visible
            bsp::Bsp::Volume(id) => {
                render_set.push(*id);
            }

            // Just ignore this case)
            bsp::Bsp::Void => {}
        }
    }

    /// Display ALL level volumes
    fn render_all(&mut self) {
        let screen_clip_oct = geom::ClipOct::from_clip_rect(self.get_screen_clip_rect());

        let mut render_set = Vec::new();
        let mut points = Vec::with_capacity(32);
        let mut point_dst = Vec::with_capacity(32);
        let mut surface_texture = Vec::new();

        let bsp = self.map.get_world_model().get_bsp();

        Self::order_rendered_volume_set(
            bsp,
            &mut render_set,
            self.camera_location
        );

        for id in render_set {
            self.render_volume(
                id,
                &screen_clip_oct,
                &mut points,
                &mut point_dst,
                &mut surface_texture
            );
        }
    }
}

/// Input render message
pub enum RenderInputMessage {
    NewFrame {
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
        camera: Camera,
        view_projection_matrix: Mat4f,
        rasterization_mode: RasterizationMode,
    }
}

/// Render output message
pub enum RenderOutputMessage {
    /// Rendered frame
    RenderedFrame {
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
    }
}

/// Build surface texture
fn build_surface_texture_impl<
    't,
    const ENABLE_LIGHTING: bool
>(
    data: &'t mut Vec<u64>,
    src: res::ImageRef,
    light: f32
) -> SurfaceTexture<'t> {

    data.clear();
    data.resize(src.width * src.height, 0);

    for y in 0..src.height {
        let src_line = &src.data[y * src.width..(y + 1) * src.width];
        let dst_line = &mut data[y * src.width..(y + 1) * src.width];

        for x in 0..src.width {
            let [r, g, b, _] = src_line[x].to_le_bytes();

            if ENABLE_LIGHTING {
                dst_line[x] = u64_from_u16([
                    unsafe { (r as f32 * light).to_int_unchecked::<u16>() },
                    unsafe { (g as f32 * light).to_int_unchecked::<u16>() },
                    unsafe { (b as f32 * light).to_int_unchecked::<u16>() },
                    0
                ]);
            } else {
                dst_line[x] = u64_from_u16([r as u16, g as u16, b as u16, 0]);
            }
        }
    }

    SurfaceTexture {
        width: src.width,
        height: src.height,
        stride: src.width,
        data: data.as_slice(),
    }
}

fn build_surface_texture<'t>(
    data: &'t mut Vec<u64>,
    src: res::ImageRef,
    light: f32,
    enable_lighting: bool
) -> SurfaceTexture<'t> {
    if enable_lighting {
        build_surface_texture_impl::<true>(data, src, light)
    } else {
        build_surface_texture_impl::<false>(data, src, light)
    }
}

/// Convert HDR buffer into LDR one
pub fn hdr_to_ldr_impl<const ENABLE_TONEMAPPING: bool>(hdr: &[u64], ldr: &mut Vec<u32>) {
    'proc: for elem in hdr {
        let [r, g, b, a] = u64_into_u16(*elem);

        if ENABLE_TONEMAPPING {
            #[cfg(target_feature = "sse2")]
            unsafe {
                let rgba_int = std::arch::x86_64::_mm_set_epi32(
                    std::mem::transmute(a as u32),
                    std::mem::transmute(b as u32),
                    std::mem::transmute(g as u32),
                    std::mem::transmute(r as u32),
                );
    
                // rgba
                let rgba = std::arch::x86_64::_mm_cvtepi32_ps(rgba_int);
    
                // rgba / 256.0 + exposure
                let rgba_norm_aexp = std::arch::x86_64::_mm_fmadd_ps(
                    rgba,
                    std::arch::x86_64::_mm_set1_ps(1.0 / 256.0),
                    std::arch::x86_64::_mm_set1_ps(0.5) // exposure
                );
    
                // rgba / (rgba / 256.0 + exposure)
                let mapped = std::arch::x86_64::_mm_div_ps(rgba, rgba_norm_aexp);
    
                let mapped_int = std::arch::x86_64::_mm_cvtps_epi32(mapped);
    
                let [r, g, b, _]: [u32; 4] =
                    std::mem::transmute(mapped_int);
    
                ldr.push(u32::from_le_bytes([
                    (r & 0xFF) as u8,
                    (g & 0xFF) as u8,
                    (b & 0xFF) as u8,
                    0,
                ]));

                continue 'proc;
            }
        }

        ldr.push(u32::from_le_bytes([
            r as u8,
            g as u8,
            b as u8,
            0
        ]));
    }
}

pub fn hdr_to_ldr(hdr: &[u64], ldr: &mut Vec<u32>, enable_tonemapping: bool) {
    if enable_tonemapping {
        hdr_to_ldr_impl::<true>(hdr, ldr);
    } else {
        hdr_to_ldr_impl::<false>(hdr, ldr);
    }
}

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    // Enable/disable map caching
    let do_enable_map_caching = true;

    // Synchronize visible-set-building and projection cameras
    let mut do_sync_logical_camera = true;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut rasterization_mode = RasterizationMode::Standard;

    // Enable delay and sync between volumes rendering
    let mut do_enable_slow_rendering = false;

    // Load map
    let map = {
        /*
        {
            let canals = map::source_vmf::Entry::parse_vmf(include_str!("../temp/canals.vmf"))
                .unwrap()
                .build_wmap()
                .unwrap();
    
            let map = bsp::Map::compile(&canals).unwrap();

            let mut file = std::fs::File::create("temp/wbsp/canals.wbsp").unwrap();
    
            map.save(&mut file).unwrap();

            break 'map map;
        }
        */

        // yay, this code will not compile on non-local builds)))
        // --
        let map_name = "q1/e1m5";
        let data_path = "temp/";

        let wbsp_path = format!("{}wbsp/{}.wbsp", data_path, map_name);
        let map_path = format!("{}{}.map", data_path, map_name);

        if do_enable_map_caching {
            match std::fs::File::open(&wbsp_path) {
                Ok(mut bsp_file) => {
                    // Load map from map cache
                    bsp::Map::load(&mut bsp_file).unwrap()
                }
                Err(_) => {
                    // Compile map
                    let source = std::fs::read_to_string(&map_path).unwrap();
                    let q1_location_map = map::q1_map::Map::parse(&source).unwrap();
                    let location_map = q1_location_map.build_wmap();
                    let compiled_map = bsp::Map::compile(&location_map).unwrap();

                    if let Some(directory) = std::path::Path::new(&wbsp_path).parent() {
                        _ = std::fs::create_dir(directory);
                    }

                    // Save map to map cache
                    if let Ok(mut file) = std::fs::File::create(&wbsp_path) {
                        compiled_map.save(&mut file).unwrap();
                    }

                    compiled_map
                }
            }
        } else {
            let source = std::fs::read_to_string(&map_path).unwrap();
            let q1_location_map = map::q1_map::Map::parse(&source).unwrap();
            let location_map = q1_location_map.build_wmap();
            bsp::Map::compile(&location_map).unwrap()
        }
    };
    let map = Arc::new(map);

    let material_table = {
        let mut wad_file = std::fs::File::open("temp/q1/gfx/medieval.wad").unwrap();

        res::MaterialTable::load_wad2(&mut wad_file).unwrap()
    };
    let material_table = Arc::new(material_table);

    struct BspStatBuilder {
        pub volume_count: usize,
        pub void_count: usize,
        pub partition_count: usize,
    }

    impl BspStatBuilder {
        /// Build BSP for certain 
        fn visit(&mut self, bsp: &bsp::Bsp) {
            match bsp {
                bsp::Bsp::Partition { splitter_plane: _, front, back } => {
                    self.partition_count += 1;

                    self.visit(front);
                    self.visit(back);
                }
                bsp::Bsp::Volume(_) => {
                    self.volume_count += 1;
                }
                bsp::Bsp::Void => {
                    self.void_count += 1;
                }
            }
        }
    }

    let mut stat = BspStatBuilder {
        partition_count: 0,
        void_count: 0,
        volume_count: 0
    };

    stat.visit(map.get_world_model().get_bsp());

    let stat_total = stat.partition_count + stat.void_count + stat.volume_count;

    println!("BSP Stat:");
    println!("    Nodes total : {}", stat_total);
    println!("    Partitions  : {} ({}%)", stat.partition_count, stat.partition_count as f64 / stat_total as f64 * 100.0);
    println!("    Volumes     : {} ({}%)", stat.volume_count, stat.volume_count as f64 / stat_total as f64 * 100.0);
    println!("    Voids       : {} ({}%)", stat.void_count, stat.void_count as f64 / stat_total as f64 * 100.0);
    // return;

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

    // camera.location = Vec3f::new(-174.0, 2114.6, -64.5); // -200, 2000, -50
    // camera.direction = Vec3f::new(-0.4, 0.9, 0.1);

    // camera.location = Vec3f::new(1402.4, 1913.7, -86.3);
    // camera.direction = Vec3f::new(-0.74, 0.63, -0.24);

    camera.location = Vec3f::new(1254.2, 1700.7, -494.5); // (1254.21606, 1700.70752, -494.493591)
    camera.direction = Vec3f::new(0.055, -0.946, 0.320); // (-0.048328593, -0.946524262, 0.318992347)

    // camera.location = Vec3f::new(-72.9, 698.3, -118.8);
    // camera.direction = Vec3f::new(0.37, 0.68, 0.63);

    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // Camera used for visible set building
    let mut logical_camera = camera;

    // Buffer that contains software-rendered pixels
    let mut hdr_frame_buffer = Vec::<u64>::new();

    // Low dynamic range framebuffer
    let mut ldr_frame_buffer = Vec::<u32>::new();

    let (render_in_sender, render_out_reciever) = {

        let (render_in_sender, render_in_reciever) =
            std::sync::mpsc::channel::<RenderInputMessage>();
        let (render_out_sender, render_out_reciever) =
            std::sync::mpsc::channel::<RenderOutputMessage>();

        let map = map.clone();
        let material_table = material_table.clone();

        // Spawn render thread
        _ = std::thread::spawn(move || {
            // Create new local timer
            let mut timer = Timer::new();

            let render_in_reciever = render_in_reciever;
            let render_out_sender = render_out_sender;

            // Map reference
            let map_arc = map.clone();
            let material_table_arc = material_table;

            let map = map_arc.as_ref();
            let material_table = material_table_arc.as_ref();

            let material_reference_table = material_table
                .build_reference_table(&map);

            // Send empty frame

            let init_send_result = render_out_sender.send(RenderOutputMessage::RenderedFrame {
                frame_buffer: Vec::new(),
                width: 0,
                height: 0,
            });

            if let Err(_) = init_send_result {
                eprintln!("Send error occured");
                return;
            }

            'frame_render_loop: loop {
                let message = match render_in_reciever.try_recv() {
                    Ok(message) => message,
                    Err(err) => match err {
                        std::sync::mpsc::TryRecvError::Disconnected => break 'frame_render_loop,
                        std::sync::mpsc::TryRecvError::Empty => continue 'frame_render_loop,
                    }
                };

                match message {
                    RenderInputMessage::NewFrame {
                        mut frame_buffer,
                        width,
                        height,
                        camera,
                        view_projection_matrix,
                        rasterization_mode
                    } => {
                        timer.response();

                        let time = timer.get_time();

                        let mut render_context = RenderContext {
                            camera_location: camera.location,

                            frame_width: width as usize,
                            frame_height: height as usize,
                            frame_stride: width as usize,
                            frame_pixels: frame_buffer.as_mut_ptr(),

                            map: &map,
                            material_table: &material_reference_table,
                            rasterization_mode,

                            sky_background_uv_offset: Vec2f::new(
                                time * -12.0,
                                time * -12.0,
                            ),
                            sky_uv_offset: Vec2f::new(
                                time * 16.0,
                                time * 16.0,
                            ),

                            view_projection_matrix,
                        };

                        // Clear framebuffer
                        unsafe {
                            std::ptr::write_bytes(
                                frame_buffer.as_mut_ptr(),
                                0,
                                frame_buffer.len()
                            );
                        }
            
                        let start_volume_id_opt = map
                            .get_world_model()
                            .get_bsp()
                            .traverse(camera.location);

                        if let Some(start_volume_id) = start_volume_id_opt {
                            render_context.render(start_volume_id);
                        } else {
                            render_context.render_all();
                        }

                        // Send output message
                        let result = render_out_sender.send(RenderOutputMessage::RenderedFrame {
                            frame_buffer,
                            width,
                            height
                        });

                        if let Err(_) = result {
                            eprintln!("Render thread result sending error occured");
                        }
                    }
                }
            }
        });

        (render_in_sender, render_out_reciever)
    };

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

        if input.is_key_clicked(Scancode::Num8) {
            do_enable_slow_rendering = !do_enable_slow_rendering;
        }

        // Synchronize logical camera with physical one
        if input.is_key_clicked(Scancode::Num9) {
            do_sync_logical_camera = !do_sync_logical_camera;
        }

        if input.is_key_clicked(Scancode::Num0) {
            rasterization_mode = rasterization_mode.next();
        }

        if do_sync_logical_camera {
            logical_camera = camera;
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        let (frame_width, frame_height) = (
            window_width / 2,
            window_height / 2,
        );

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

        // Get projection matrix
        let projection_matrix = Mat4f::projection_frustum_inf_far(
            -0.5 * aspect_x, 0.5 * aspect_x,
            -0.5 * aspect_y, 0.5 * aspect_y,
            0.66
        );

        let view_projection_matrix = view_matrix * projection_matrix;

        let present_frame = |frame_buffer: *mut u32, frame_width: usize, frame_height: usize, frame_stride: usize| {
            let mut window_surface = match window.surface(&event_pump) {
                Ok(window_surface) => window_surface,
                Err(err) => {
                    eprintln!("Cannot get window surface: {}", err);
                    return;
                }
            };

            let mut render_surface = match sdl2::surface::Surface::from_data(
                unsafe {
                    std::slice::from_raw_parts_mut(
                        frame_buffer as *mut u8,
                        frame_stride * frame_width * 4
                    )
                },
                frame_width as u32,
                frame_height as u32,
                frame_width as u32 * 4,
                sdl2::pixels::PixelFormatEnum::ABGR8888
            ) {
                Ok(surface) => surface,
                Err(err) => {
                    eprintln!("Source surface create error: {}", err);
                    return;
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
        };
        
        // Resize frame buffer to fit window's size
        hdr_frame_buffer.resize(frame_width * frame_height, 0);

        let send_res = render_in_sender.send(RenderInputMessage::NewFrame {
            frame_buffer: hdr_frame_buffer,
            width: frame_width as u32,
            height: frame_height as u32,
            camera: logical_camera,
            view_projection_matrix,
            rasterization_mode,
        });

        if let Err(_) = send_res {
            break 'main_loop;
        }

        // Previous frame contents
        let prev_hdr_frame_buffer;

        'res_await_loop: loop {
            let render_result = match render_out_reciever.try_recv() {
                Ok(result) => result,
                Err(err) => match err {
                    std::sync::mpsc::TryRecvError::Disconnected => {
                        eprintln!("Render thread dropped");
                        break 'main_loop;
                    },
                    std::sync::mpsc::TryRecvError::Empty => continue 'res_await_loop,
                }
            };
    
            match render_result {
                RenderOutputMessage::RenderedFrame {
                    frame_buffer: rendered_hdr_buffer,
                    width,
                    height
                } => {
                    // Build low dynamic range buffer
                    ldr_frame_buffer.clear();
                    ldr_frame_buffer.reserve(width as usize * height as usize);

                    let start = std::time::Instant::now();
                    hdr_to_ldr(&rendered_hdr_buffer, &mut ldr_frame_buffer, true);
                    let end = std::time::Instant::now();

                    // tonemapping time
                    let tm_time = end.duration_since(start).as_nanos() as f64 / 1_000_000f64;

                    system_font::frame(
                        width as usize,
                        height as usize,
                        width as usize,
                        ldr_frame_buffer.as_mut_ptr()
                    )
                        .str(16, 8, &format!("FPS: {} ({}ms)", timer.get_fps(), 1000.0 / timer.get_fps()))
                        .str(16, 16, &format!("SR={}, SL={}, RM={}",
                            do_enable_slow_rendering as u32,
                            do_sync_logical_camera as u32,
                            rasterization_mode as u32
                        ))
                        .str(16, 24, &format!("TM: {}ms", tm_time))
                    ;

                    // Present frame
                    present_frame(
                        ldr_frame_buffer.as_mut_ptr(),
                        width as usize,
                        height as usize,
                        width as usize,
                    );

                    prev_hdr_frame_buffer = Some(rendered_hdr_buffer);

                    break 'res_await_loop;
                }
            }
        }

        hdr_frame_buffer = prev_hdr_frame_buffer.unwrap_or(Vec::new());
    }
}

// main.rs
