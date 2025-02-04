/// Main project module

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Visible and Physical BSP building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with location BSP's.

use std::collections::{BTreeMap, HashMap};
use bytemuck::Zeroable;
use math::{Mat4f, Vec2f, Vec3f, Vec5UVf};
use sdl2::keyboard::Scancode;

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

/// Basic geometry
pub mod geom;

/// Compiled map implementation
pub mod bsp;

/// Map format implementation
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

/// Necessary data for slow rendering
struct SlowRenderContext<'t> {
    /// Frame presentation function reference
    pub present_func: &'t dyn Fn(*mut u32, usize, usize, usize) -> (),

    /// Duration between frames rendered
    pub delta_time: std::time::Duration,
}

mod wad2 {
    /// WAD2 file magic
    pub const MAGIC: [u8; 4] = *b"WAD2";

    #[repr(C, packed)]
    #[derive(Copy, Clone)]
    pub struct Header {
        pub magic: [u8; 4],
        pub num_entities: i32,
        pub dir_offset: i32,
    }

    unsafe impl bytemuck::Zeroable for Header {}
    unsafe impl bytemuck::AnyBitPattern for Header {}
    unsafe impl bytemuck::NoUninit for Header {}


    /// WAD entry
    #[repr(C, packed)]
    #[derive(Copy, Clone)]
    pub struct Entry {
        pub offset: i32,
        pub entry_size: i32,
        pub size: i32,
        pub ty: i8,
        pub compression: i8,
        pub _dummy: u16,
        pub name: [u8; 16],
    }


    unsafe impl bytemuck::Zeroable for Entry {}
    unsafe impl bytemuck::AnyBitPattern for Entry {}
    unsafe impl bytemuck::NoUninit for Entry {}

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub enum EntryType {
        ColorPalette = 0x40,
        StatusBarPictures = 0x42,
        MipTexture = 0x44,
        ConsolePicture = 0x45,
    }

    impl EntryType {
        pub fn from_i8(v: i8) -> Option<Self> {
            Some(match v {
                0x40 => Self::ColorPalette,
                0x42 => Self::StatusBarPictures,
                0x44 => Self::MipTexture,
                0x45 => Self::ConsolePicture,
                _ => return None,
            })
        }
    }

    #[repr(C, packed)]
    #[derive(Copy, Clone)]
    pub struct PictureHeader {
        pub width: i32,
        pub height: i32,
        // Pixels: array of i8 with width x height size
    }

    unsafe impl bytemuck::Zeroable for PictureHeader {}
    unsafe impl bytemuck::AnyBitPattern for PictureHeader {}
    unsafe impl bytemuck::NoUninit for PictureHeader {}

    #[repr(C, packed)]
    #[derive(Copy, Clone)]
    pub struct Color {
        pub r: u8,
        pub g: u8,
        pub b: u8,
    }

    impl Color {
        pub const fn zero() -> Self {
            Self { r: 0, g: 0, b: 0 }
        }
    }

    unsafe impl bytemuck::Zeroable for Color {}
    unsafe impl bytemuck::AnyBitPattern for Color {}
    unsafe impl bytemuck::NoUninit for Color {}
}

/// Texture structure
struct Texture {
    pub width: usize,
    pub height: usize,
    pub colors: Vec<u32>,
}

/// Material table imported from WAD files
struct WadMaterialTable {
    /// Index table
    pub index_table: BTreeMap<String, usize>,

    /// Zero index stands for default texture
    pub index_map: Vec<Texture>,
}

#[derive(Debug)]
pub enum Wad2LoadingError {
    /// Invali WAD file magic
    InvalidWadMagic([u8; 4]),

    /// Unexpected WAD file end
    UnexpectedFileEnd,

    /// File interaction error
    IoError(std::io::Error),

    /// Error of building UTF-8 from bytes that (technically) should be string
    Utf8Error(std::str::Utf8Error),

    /// Slice is smaller, than required
    TooSmallSlice(std::array::TryFromSliceError),

    /// Compressed data occured
    CompressedData,

    /// Entry type isn't documented.
    UnknownEntryType(i8),
}

impl From<std::io::Error> for Wad2LoadingError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

impl From<std::str::Utf8Error> for Wad2LoadingError {
    fn from(value: std::str::Utf8Error) -> Self {
        Self::Utf8Error(value)
    }
}

impl From<std::array::TryFromSliceError> for Wad2LoadingError {
    fn from(value: std::array::TryFromSliceError) -> Self {
        Self::TooSmallSlice(value)
    }
}

impl WadMaterialTable {
    /// Load WAD2 format in material table
    pub fn load_wad2(file: &mut dyn std::io::Read) -> Result<Self, Wad2LoadingError> {
        let mut data = Vec::<u8>::new();

        file.read_to_end(&mut data)?;

        let mut data_rest = data.as_slice();

        let header_bytes;

        (header_bytes, data_rest) = data_rest
            .split_at_checked(std::mem::size_of::<wad2::Header>())
            .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

        let header = bytemuck::from_bytes::<wad2::Header>(header_bytes);

        // Read WAD2 header
        if header.magic != wad2::MAGIC {
            return Err(Wad2LoadingError::InvalidWadMagic(header.magic));
        }

        let entry_bytes;

        (entry_bytes, data_rest) = data_rest
            .split_at_checked(std::mem::size_of::<wad2::Entry>() * header.num_entities as usize)
            .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

        // Drop data rest
        _ = data_rest;

        let entries = bytemuck::cast_slice::<_, wad2::Entry>(entry_bytes);

        let mut color_palette: &[wad2::Color; 256] = &[wad2::Color::zero(); 256];

        // Read WAD entries
        for entry in entries {
            // I Don't know how to decode compressed data...
            if entry.compression != 0 {
                return Err(Wad2LoadingError::CompressedData);
            }

            let ty = wad2::EntryType::from_i8(entry.ty)
                .ok_or(Wad2LoadingError::UnknownEntryType(entry.ty))?;

            let name = std::str::from_utf8(&entry.name)?;

            let entry_data = data
                .get(entry.offset as usize..entry.offset as usize + entry.size as usize)
                .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

            match ty {
                wad2::EntryType::ColorPalette => {
                    color_palette = bytemuck::cast_slice(entry_data)
                        .try_into()
                        .unwrap();
                }
                wad2::EntryType::ConsolePicture => {
                    // ignore
                }
                wad2::EntryType::StatusBarPictures => {
                    // ignore
                }
                wad2::EntryType::MipTexture => {

                }
            }
        }

        todo!()
    }
}

/// Temporary structure used as instead of WRES-based material tables
struct MaterialTable {
    /// Color set
    color_set: Vec<u32>,
}

impl MaterialTable {
    /// Generate color table for map
    pub fn for_bsp(bsp: &bsp::Map) -> Self {
        let mut color_set = Vec::new();
        let mut rand_device = rand::Xorshift128p::new(304780.try_into().unwrap());

        for (id, _) in bsp.all_material_names() {
            let index = id.into_index();

            color_set.resize(index + 1, !0u32);

            if color_set[index] == !0u32 {
                color_set[index] = (rand_device.next() & 0xFFFF_FFFF) as u32;
            }
        }

        Self { color_set }
    }

    /// Get color for material by it's ID
    pub fn get_color(&self, material_id: bsp::MaterialId) -> Option<u32> {
        self.color_set.get(material_id.into_index()).copied()
    }
}

/// Render context
struct RenderContext<'t> {
    /// Camera location
    camera_location: Vec3f,

    /// Camera visibility plane
    // camera_plane: geom::Plane,

    /// VP matrix
    view_projection_matrix: Mat4f,

    /// Map reference
    map: &'t bsp::Map,

    material_table: &'t MaterialTable,

    /// Frame pixel array pointer
    frame_pixels: *mut u32,

    /// Frame width
    frame_width: usize,

    /// Frame height
    frame_height: usize,

    /// Frame stride
    frame_stride: usize,

    /// Rasterization mode
    rasterization_mode: RasterizationMode,

    slow_render_context: Option<SlowRenderContext<'t>>,
}

/// Different rasterization modes
#[derive(Copy, Clone, PartialEq, Eq)]
enum RasterizationMode {
    Standard = 0,
    Overdraw = 1,
    Depth = 2,
    UV = 3,
}

impl RasterizationMode {
    // Rasterization mode count
    const COUNT: u32 = 4;

    /// Build rasterization mode from u32
    pub const fn from_u32(n: u32) -> Option<RasterizationMode> {
        Some(match n {
            0 => RasterizationMode::Standard,
            1 => RasterizationMode::Overdraw,
            2 => RasterizationMode::Depth,
            3 => RasterizationMode::UV,
            _ => return None,
        })
    }

    pub const fn next(self) -> RasterizationMode {
        Self::from_u32((self as u32 + 1) % Self::COUNT).unwrap()
    }
}

impl<'t> RenderContext<'t> {
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

    /// Clip polygon by some rectangle
    fn clip_polygon_rect(&self, points: &mut Vec<Vec5UVf>, result: &mut Vec<Vec5UVf>, clip_rect: geom::ClipRect) {
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.x >= clip_rect.min.x {
                result.push(curr);

                if next.x < clip_rect.min.x {
                    let t = (clip_rect.min.x - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x >= clip_rect.min.x {
                let t = (clip_rect.min.x - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(points, result);
        
        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y >= clip_rect.min.y {
                result.push(curr);

                if next.y < clip_rect.min.y {
                    let t = (clip_rect.min.y - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y >= clip_rect.min.y {
                let t = (clip_rect.min.y - curr.y) / (next.y - curr.y);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(points, result);

        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.x <= clip_rect.max.x {
                result.push(curr);

                if next.x > clip_rect.max.x {
                    let t = (clip_rect.max.x - curr.x) / (next.x - curr.x);

                    result.push((next - curr) * t + curr);
                }
            } else if next.x <= clip_rect.max.x {
                let t = (clip_rect.max.x - curr.x) / (next.x - curr.x);

                result.push((next - curr) * t + curr);
            }
        }

        std::mem::swap(points, result);

        result.clear();
        for index in 0..points.len() {
            let curr = points[index];
            let next = points[(index + 1) % points.len()];

            if curr.y <= clip_rect.max.y {
                result.push(curr);

                if next.y > clip_rect.max.y {
                    let t = (clip_rect.max.y - curr.y) / (next.y - curr.y);

                    result.push((next - curr) * t + curr);
                }
            } else if next.y <= clip_rect.max.y {
                let t = (clip_rect.max.y - curr.y) / (next.y - curr.y);

                result.push((next - curr) * t + curr);
            }
        }
    }

    /// Project polygon on screen
    pub fn get_screenspace_polygon(&self, points: &mut Vec<Vec5UVf>, point_dst: &mut Vec<Vec5UVf>) {
        // Calculate projection-space points
        point_dst.clear();
        for pt in points.iter() {
            point_dst.push(Vec5UVf::from_32(
                self.view_projection_matrix.transform_point(pt.xyz()),
                pt.uv()
            ));
        }
        std::mem::swap(points, point_dst);

        // Clip polygon invisible part
        point_dst.clear();
        self.clip_viewspace_back(points, point_dst);
        std::mem::swap(points, point_dst);

        // Calculate screen-space points
        let width = self.frame_width as f32 * 0.5;
        let height = self.frame_height as f32 * 0.5;

        point_dst.clear();
        for v in points.iter() {
            point_dst.push(Vec5UVf::new(
                (1.0 + v.x / v.z) * width,
                (1.0 - v.y / v.z) * height,
                1.0 / v.z,
                v.u / v.z,
                v.v / v.z,
            ));
        }
    }

    unsafe fn render_clipped_polygon(&mut self, is_transparent: bool, points: &[Vec5UVf], color: u32) {
        if is_transparent {
            match self.rasterization_mode {
                RasterizationMode::Standard => self.render_clipped_polygon_impl::<0, true>(&points, color),
                RasterizationMode::Overdraw => self.render_clipped_polygon_impl::<1, true>(&points, color),
                RasterizationMode::Depth    => self.render_clipped_polygon_impl::<2, true>(&points, color),
                RasterizationMode::UV       => self.render_clipped_polygon_impl::<3, true>(&points, color),
            }
        } else {
            match self.rasterization_mode {
                RasterizationMode::Standard => self.render_clipped_polygon_impl::<0, false>(&points, color),
                RasterizationMode::Overdraw => self.render_clipped_polygon_impl::<1, false>(&points, color),
                RasterizationMode::Depth    => self.render_clipped_polygon_impl::<2, false>(&points, color),
                RasterizationMode::UV       => self.render_clipped_polygon_impl::<3, false>(&points, color),
            }
        }
}

    /// Rasterize already clipped polygon
    unsafe fn render_clipped_polygon_impl<
        const MODE: u32,
        const IS_TRANSPARENT: bool,
    >(
        &mut self,
        points: &[Vec5UVf],
        color: u32,
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
        let first_line = usize::min(min_y_value.floor() as usize, self.frame_height - 1);
        let last_line = usize::min(max_y_value.ceil() as usize, self.frame_height);

        let mut left_index = ind_next!(min_y_index);
        let mut left_prev = points[min_y_index];
        let mut left_curr = points[left_index];
        let mut left_slope = (left_curr - left_prev) / (left_curr.y - left_prev.y);

        // let mut left_inv_slope = (left_curr.x - left_prev.x) / (left_curr.y - left_prev.y);
        // let mut left_inv_z_slope = (left_curr.z - left_prev.z) / (left_curr.y - left_prev.y);

        let mut right_index = ind_prev!(min_y_index);
        let mut right_prev = points[min_y_index];
        let mut right_curr = points[right_index];
        let mut right_slope = (right_curr - right_prev) / (right_curr.y - right_prev.y);

        // Scan for lines
        'line_loop: for pixel_y in first_line..last_line {
            // Get current pixel y
            let y = pixel_y as f32 + 0.5;

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
                    left_slope = Vec5UVf::zero();
                } else {
                    left_slope = (left_curr - left_prev) / left_dy;
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
                    right_slope = Vec5UVf::zero();
                } else {
                    right_slope = (right_curr - right_prev) / right_dy;
                }
            }

            // Calculate intersection with left edge
            let left_x = left_prev.x + (y - left_prev.y) * left_slope.x;

            // Calculate intersection with right edge
            let right_x = right_prev.x + (y - right_prev.y) * right_slope.x;

            let left_zuv = left_prev.zuv() + left_slope.zuv() * (y - left_prev.y);

            let right_zuv = right_prev.zuv() + right_slope.zuv() * (y - right_prev.y);

            // Calculate hline start/end, clip it
            let start = usize::min(left_x.floor() as usize, self.frame_width - 1);
            let end = usize::min(right_x.floor() as usize, self.frame_width - 1);

            let pixel_row = self.frame_pixels.add(self.frame_stride * pixel_y);

            let dx = right_x - left_x;

            let slope_zuv = if dx <= DY_EPSILON {
                Vec3f::zero()
            } else {
                (right_zuv - left_zuv) / (right_x - left_x)
            };

            'render: for pixel_x in start..end {
                let zuv = left_zuv + slope_zuv * (pixel_x - start) as f32;
                let pixel_ptr = pixel_row.add(pixel_x);

                // Handle transparency
                if IS_TRANSPARENT {
                    if (pixel_y ^ pixel_x) & 1 == 1 {
                        continue 'render;
                    }
                }

                if MODE == RasterizationMode::Standard as u32 {
                    *pixel_ptr = color;
                } else if MODE == RasterizationMode::Overdraw as u32 {
                    *pixel_ptr = 0x101010u32.wrapping_add(*pixel_ptr);
                } else if MODE == RasterizationMode::Depth as u32 {
                    let color = (zuv.x * 25500.0) as u8;
                    *pixel_ptr = u32::from_le_bytes([color, color, color, color]);
                } else if MODE == RasterizationMode::UV as u32 {
                    let uv = Vec2f::new(
                        zuv.y / zuv.x,
                        zuv.z / zuv.x,
                    );

                    let xi = (uv.x as i64 & 0xFF) as u8;
                    let yi = (uv.y as i64 & 0xFF) as u8;

                    let c = ((xi / 32) ^ (yi / 32)) & 1;

                    let rgb = bsp::Rgb8::from(color);

                    *pixel_ptr = u32::from_le_bytes([
                        rgb.r * c,
                        rgb.g * c,
                        rgb.b * c,
                        0,
                    ]);
                } else {
                    panic!("Unknown rasterization mode: {}", MODE);
                }
            }

            // let pixel_start = self.frame_pixels.add(self.frame_stride * pixel_y + start);
            // let pixel_end = self.frame_pixels.add(self.frame_stride * pixel_y + end);
            // let mut pixel_ptr = pixel_start;
            // while pixel_ptr < pixel_end {
            //     if RENDER_OVERDRAW {
            //         *pixel_ptr = 0x101010u32.wrapping_add(*pixel_ptr);
            //     } else {
            //         *pixel_ptr = color;
            //     }
            //     pixel_ptr = pixel_ptr.add(1);
            // }
        }
    }

    /// Render polygon
    fn render_polygon(
        &mut self,
        polygon: &geom::Polygon,
        plane_u: geom::Plane,
        plane_v: geom::Plane,
        color: [u8; 3],
        is_transparent: bool,
        clip_rect: geom::ClipRect
    ) {
        // Calculate polygon point set
        let mut points = polygon.points
            .iter()
            .map(|v| Vec5UVf::from_32(
                *v,
                Vec2f::new(
                    (*v ^ plane_u.normal) + plane_u.distance,
                    (*v ^ plane_v.normal) + plane_v.distance,
                )
            ))
            .collect::<Vec<_>>();
        let mut point_dst = Vec::with_capacity(polygon.points.len());

        // Get projected polygons
        self.get_screenspace_polygon(&mut points, &mut point_dst);
        std::mem::swap(&mut points, &mut point_dst);

        // Clip polygon by volume clipping rectangle
        point_dst.clear();
        self.clip_polygon_rect(&mut points, &mut point_dst, clip_rect);
        std::mem::swap(&mut points, &mut point_dst);

        // Just for safety
        for pt in &points {
            if !pt.x.is_finite() || !pt.y.is_finite() {
                return;
            }
        }

        // Check if it is a polygon
        if points.len() < 3 {
            return;
        }

        // Build color U32
        let color_u32 = u32::from_le_bytes([color[0], color[1], color[2], 0]);

        // Rasterize polygon

        unsafe {
            self.render_clipped_polygon(
                is_transparent,
                &points,
                color_u32
            );
        }
    }

    fn render_clip_rect(&mut self, clip_rect: geom::ClipRect, color: u32) {
        let y0 = usize::min(clip_rect.min.y.ceil() as usize, self.frame_height - 1);
        let y1 = usize::min(clip_rect.max.y.ceil() as usize, self.frame_height - 1);

        let x0 = usize::min(clip_rect.min.x.floor() as usize, self.frame_width - 1);
        let x1 = usize::min(clip_rect.max.x.floor() as usize, self.frame_width - 1);

        let line = |mut start: *mut u32, delta: usize, count: usize| {
            for _ in 0..count {
                unsafe {
                    *start = color;
                    start = start.add(delta);
                }
            }
        };

        unsafe {
            let s = self.frame_stride;

            line(self.frame_pixels.add(y0 * s + x0), 1, x1 - x0);
            line(self.frame_pixels.add(y0 * s + x0), s, y1 - y0);
            line(self.frame_pixels.add(y1 * s + x0), 1, x1 - x0);
            line(self.frame_pixels.add(y0 * s + x1), s, y1 - y0);
        }
    }

    /// Render single volume
    fn render_volume(&mut self, id: bsp::VolumeId, clip_rect: geom::ClipRect) {
        let volume = self.map.get_volume(id).unwrap();

        'polygon_rendering: for surface in volume.get_surfaces() {
            let polygon = self.map.get_polygon(surface.polygon_id).unwrap();

            // Perform backface culling
            if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                continue 'polygon_rendering;
            }

            // Get surface material
            let color: bsp::Rgb8 = self
                .material_table
                .get_color(surface.material_id)
                .unwrap()
                .into();

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
                (color.r as f32 * light) as u8,
                (color.g as f32 * light) as u8,
                (color.b as f32 * light) as u8,
            ];

            self.render_polygon(
                &polygon,
                surface.u,
                surface.v,
                color,
                surface.is_transparent,
                clip_rect
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
                self.frame_width as f32 - 1.0 - CLIP_OFFSET,
                self.frame_height as f32 - 1.0 - CLIP_OFFSET,
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
    pub fn render_build_render_set(
        &self,
        bsp_elem: &bsp::Bsp,
        pvs: &mut BTreeMap<bsp::VolumeId, geom::ClipRect>,
        inv_render_set: &mut Vec<(bsp::VolumeId, geom::ClipRect)>,
    ) {
        match bsp_elem {
            bsp::Bsp::Partition {
                splitter_plane,
                front,
                back
            } => {
                let rel = splitter_plane.get_point_relation(self.camera_location);

                let (first, second) = match rel {
                    geom::PointRelation::Front | geom::PointRelation::OnPlane => {
                        (front, back)
                    }
                    geom::PointRelation::Back => {
                        (back, front)
                    }
                };

                self.render_build_render_set(first, pvs, inv_render_set);
                self.render_build_render_set(second, pvs, inv_render_set);
            }
            bsp::Bsp::Volume(volume_id) => 'volume_traverse: {
                let Some(volume_clip_rect) = pvs.get(volume_id) else {
                    break 'volume_traverse;
                };
                let volume_clip_rect = *volume_clip_rect;

                // Insert volume in render set
                inv_render_set.push((*volume_id, volume_clip_rect));

                let volume = self.map.get_volume(*volume_id).unwrap();

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
                            break 'portal_validation volume_clip_rect;
                        }

                        let mut polygon_points = portal_polygon.points
                            .iter()
                            .map(|point| Vec5UVf::from_32(*point, Vec2f::zero()))
                            .collect::<Vec<_>>();
                        let mut proj_polygon_points: Vec<Vec5UVf> = Vec::with_capacity(polygon_points.len());

                        if !portal.is_facing_front {
                            polygon_points.reverse();
                        }

                        self.get_screenspace_polygon(&mut polygon_points, &mut proj_polygon_points);

                        // Check if it's even a polygon
                        if proj_polygon_points.len() < 3 {
                            continue 'portal_rendering;
                        }

                        let proj_rect = geom::ClipRect::from_points_xy(
                            proj_polygon_points
                                .iter()
                                .map(|v| v.xyz())
                        );

                        let Some(clip_rect) = geom::ClipRect::intersection(
                            volume_clip_rect,
                            proj_rect.extend(Vec2f::new(1.0, 1.0))
                        ) else {
                            continue 'portal_rendering;
                        };

                        clip_rect
                    };


                    // Insert clipping rectangle in PVS
                    let pvs_entry = pvs
                        .entry(portal.dst_volume_id);

                    match pvs_entry {
                        std::collections::btree_map::Entry::Occupied(mut occupied) => {
                            let existing_rect: &mut geom::ClipRect = occupied.get_mut();
                            *existing_rect = existing_rect.union(clip_rect);
                        }
                        std::collections::btree_map::Entry::Vacant(vacant) => {
                            vacant.insert(clip_rect);
                        }
                    }
                }
            }
            bsp::Bsp::Void => {}
        }
    }

    /// Render scene starting from certain volume
    pub fn render(&mut self, start_volume_id: bsp::VolumeId) {
        let screen_clip_rect = self.get_screen_clip_rect();

        let mut pvs = BTreeMap::new();
        let mut inv_render_set = Vec::new();

        pvs.insert(start_volume_id, screen_clip_rect);

        self.render_build_render_set(self.map.get_bsp(), &mut pvs, &mut inv_render_set);

        if let Some(ctx) = self.slow_render_context.as_ref() {
            (ctx.present_func)(self.frame_pixels, self.frame_width, self.frame_height, self.frame_stride);
            std::thread::sleep(ctx.delta_time);
        }

        for (volume_id, volume_clip_rect) in inv_render_set
            .iter()
            .rev()
            .copied()
        {

            let Some(clip_rect) = volume_clip_rect.intersection(screen_clip_rect) else {
                continue;
            };

            if self.slow_render_context.is_some() {
                self.render_clip_rect(clip_rect, 0x0000FF);

                let ctx = self.slow_render_context.as_ref().unwrap();
                (ctx.present_func)(self.frame_pixels, self.frame_width, self.frame_height, self.frame_stride);
                std::thread::sleep(ctx.delta_time);
            }

            self.render_volume(volume_id, volume_clip_rect);

            if let Some(ctx) = self.slow_render_context.as_ref() {
                (ctx.present_func)(self.frame_pixels, self.frame_width, self.frame_height, self.frame_stride);

                std::thread::sleep(ctx.delta_time);
            }
        }
    }

    /// Build correct volume set rendering order
    fn order_rendered_volume_set(
        bsp: &bsp::Bsp,
        visible_set: &mut BTreeMap<bsp::VolumeId, geom::ClipRect>,
        render_set: &mut Vec<(bsp::VolumeId, geom::ClipRect)>,
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
                    visible_set,
                    render_set,
                    camera_location
                );

                Self::order_rendered_volume_set(
                    &second,
                    visible_set,
                    render_set,
                    camera_location
                );
            }

            // Move volume to render set if it's visible
            bsp::Bsp::Volume(id) => {
                if let Some(clip_rect) = visible_set.remove(id) {
                    render_set.push((*id, clip_rect));
                }
            }

            // Just ignore this case)
            bsp::Bsp::Void => {}
        }
    }

    /// Display ALL level volumes
    fn render_all(&mut self) {
        let screen_clip_rect = self.get_screen_clip_rect();

        let mut visible_set = BTreeMap::from_iter(self
            .map
            .all_volume_ids()
            .map(|id| (id, screen_clip_rect))
        );
        let mut render_set = Vec::new();

        Self::order_rendered_volume_set(
            self.map.get_bsp(),
            &mut visible_set,
            &mut render_set,
            self.camera_location
        );

        for (id, clip_rect) in render_set {
            self.render_volume(id, clip_rect);
        }
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
        // yay, this code will not compile on non-local builds)))
        // --
        let map_name = "e1m1";
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
                    let q1_location_map = map::q1::Map::parse(&source).unwrap();
                    let location_map = q1_location_map.build_wmap();
                    let compiled_map = bsp::builder::build(&location_map);

                    // Save map to map cache
                    if let Ok(mut file) = std::fs::File::create(&wbsp_path) {
                        compiled_map.save(&mut file).unwrap();
                    }

                    compiled_map
                }
            }
        } else {
            let source = std::fs::read_to_string(&map_path).unwrap();
            let q1_location_map = map::q1::Map::parse(&source).unwrap();
            let location_map = q1_location_map.build_wmap();
            bsp::builder::build(&location_map)
        }
    };

    let material_table = MaterialTable::for_bsp(&map);
    
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

    stat.visit(map.get_bsp());

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

    // Set spectial camera location
    camera.location = Vec3f::new(-174.0, 2114.6, -64.5); // -200, 2000, -50
    camera.direction = Vec3f::new(-0.4, 0.9, 0.1);

    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

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

        // Display some statistics
        if timer.get_frame_count() % 100 == 0 {
            println!("FPS: {}, SR={}, SL={}, OV={}",
                timer.get_fps(),
                do_enable_slow_rendering as u32,
                do_sync_logical_camera as u32,
                rasterization_mode as u32,
            );
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        let (frame_width, frame_height) = (
            window_width / 4,
            window_height / 4,
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

        // Compute projection matrix
        let projection_matrix = Mat4f::projection_frustum(
            -0.5 * aspect_x, 0.5 * aspect_x,
            -0.5 * aspect_y, 0.5 * aspect_y,
            0.66, 100.0
        );

        let view_projection_matrix = view_matrix * projection_matrix;

        // Resize frame buffer to fit window's size
        frame_buffer.resize(frame_width * frame_height, 0);

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

        // Build render context
        let mut render_context = RenderContext {
            camera_location: logical_camera.location,
            // camera_plane: geom::Plane::from_point_normal(
            //     logical_camera.location,
            //     logical_camera.direction
            // ),
            view_projection_matrix,
            map: &map,
            material_table: &material_table,

            frame_pixels: frame_buffer.as_mut_ptr(),
            frame_width: frame_width,
            frame_height: frame_height,
            frame_stride: frame_width,

            rasterization_mode,
            slow_render_context: if do_enable_slow_rendering {
                Some(SlowRenderContext {
                    delta_time: std::time::Duration::from_millis(50),
                    present_func: &present_frame,
                })
            } else {
                None
            },
        };

        // TODO: Parallel rendering/presentation

        // Clear framebuffer
        // frame_buffer.fill(0xCC774C);
        unsafe {
            std::ptr::write_bytes(
                frame_buffer.as_mut_ptr(),
                0,
                frame_buffer.len()
            );
        }

        // Render frame
        let start_volume_index_opt = map
            .get_bsp()
            .traverse(logical_camera.location);

        if let Some(start_volume_id) = start_volume_index_opt {
            render_context.render(start_volume_id);
        } else {
            render_context.render_all();
        }

        present_frame(
            frame_buffer.as_mut_ptr(),
            frame_width,
            frame_height,
            frame_width
        );
    }
}

// main.rs
