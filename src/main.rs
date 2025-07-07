/// Main project module

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Visible and Physical BSP building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with location BSP's.

use std::{collections::{HashMap, HashSet}, sync::Arc};
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

/// Camera
pub mod camera;

pub mod timer;

pub mod input;

/// Different rasterization modes
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum RasterizationMode {
    /// Solid color
    Standard = 0,

    /// Overdraw (brighter => more overdraw)
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

/// Clip polygon by z=1 plane
fn clip_polygon_z1(points: &mut Vec<Vec5UVf>, result: &mut Vec<Vec5UVf>) -> bool {
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

    result.len() >= 3
}

/// Clip polygon by octagon
/// # Return
/// True if there are some points
fn clip_polygon_oct(
    points: &mut Vec<Vec5UVf>,
    result: &mut Vec<Vec5UVf>,
    clip_oct: &geom::ClipOct
) -> bool {
    macro_rules! clip_edge {
        ($metric: ident, $clip_val: expr, $cmp: tt) => {
            {
                result.clear();
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
                std::mem::swap(points, result);

                points.len() >= 3
            }
        }
    }

    macro_rules! metric_x { ($e: ident) => { ($e.x) } }
    macro_rules! metric_y { ($e: ident) => { ($e.y) } }
    macro_rules! metric_y_a_x { ($e: ident) => { ($e.y + $e.x) } }
    macro_rules! metric_y_s_x { ($e: ident) => { ($e.y - $e.x) } }

    // Utilize '&&' hands calculation rules to stop clipping if there's <= 3 points
    let value = true
        && clip_edge!(metric_x,     clip_oct.min_x,     >=) // min X
        && clip_edge!(metric_x,     clip_oct.max_x,     <=) // max X
        && clip_edge!(metric_y,     clip_oct.min_y,     >=) // min Y
        && clip_edge!(metric_y,     clip_oct.max_y,     <=) // max Y
        && clip_edge!(metric_y_a_x, clip_oct.min_y_a_x, >=) // min Y+X
        && clip_edge!(metric_y_a_x, clip_oct.max_y_a_x, <=) // max Y+X
        && clip_edge!(metric_y_s_x, clip_oct.min_y_s_x, >=) // min Y-X
        && clip_edge!(metric_y_s_x, clip_oct.max_y_s_x, <=) // max Y-X
    ;

    // Swap results (again)
    std::mem::swap(points, result);

    value
}

/// Camera that (additionally) holds info about projection and frame size
pub struct RenderCamera {
    /// View-projection matrix
    pub view_projection: Mat4f,

    /// Camera location
    pub location: Vec3f,

    /// Half of frame width
    pub half_fw: f32,

    /// Half of frame height
    pub half_fh: f32,
}

impl RenderCamera {
    pub fn get_screenspace_projected_portal_polygon(
        &self,
        points: &mut Vec<Vec3f>,
        point_dst: &mut Vec<Vec3f>
    ) {
        point_dst.clear();
        for point in points.iter() {
            point_dst.push(self.view_projection.transform_point(*point));
        }
        std::mem::swap(points, point_dst);

        // Clip polygon by z=1
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

        point_dst.clear();
        for pt in points.iter() {
            let inv_z = pt.z.recip();

            point_dst.push(Vec3f::new(
                (1.0 + pt.x * inv_z) * self.half_fw,
                (1.0 - pt.y * inv_z) * self.half_fh,
                inv_z,
            ));
        }
    }

    pub fn project_sky_polygon(
        &self,
        polygon: &geom::Polygon,
        point_dst: &mut Vec<Vec5UVf>,
        uv_offset: Vec2f,
    ) {
        // Apply reprojection for sky polygon
        for point in polygon.points.iter() {
            // Skyplane distance
            const DIST: f32 = 128.0;

            let rp = *point - self.location;

            // Copy z sign to correct back-z case (s_dist - signed distance)
            let s_dist = DIST.copysign(rp.z);

            let u = rp.x / rp.z * s_dist;
            let v = rp.y / rp.z * s_dist;

            point_dst.push(Vec5UVf::from_32(
                // Vector-only transform is used here to don't add camera location back.
                self.view_projection.transform_vector(Vec3f::new(u, v, s_dist)),
                Vec2f::new(u + uv_offset.x, v + uv_offset.y)
            ));
        }
    }

    /// Apply projection to default polygon
    pub fn project_polygon(
        &self,
        polygon: &geom::Polygon,
        material_u: geom::Plane,
        material_v: geom::Plane,
        point_dst: &mut Vec<Vec5UVf>,
    ) {
        point_dst.clear();
        for point in polygon.points.iter() {
            point_dst.push(Vec5UVf::from_32(
                self.view_projection.transform_point(*point),
                Vec2f::new(
                    point.dot(material_u.normal) + material_u.distance,
                    point.dot(material_v.normal) + material_v.distance,
                )
            ));
        }
    }

    pub fn get_screenspace_polygon(
        &self,
        polygon: &mut [Vec5UVf],
    ) {
        for pt in polygon {
            let inv_z = pt.z.recip();

            *pt = Vec5UVf::new(
                (1.0 + pt.x * inv_z) * self.half_fw,
                (1.0 - pt.y * inv_z) * self.half_fh,
                inv_z,
                pt.u * inv_z,
                pt.v * inv_z,
            );
        }
    }
}

/// Renderer frame
#[derive(Copy, Clone)]
pub struct RenderFrame {
    /// Frame pixel data
    pixels: *mut u64,

    /// Count of pixels in line that are allowed to write
    width: usize,

    /// Count of lines
    height: usize,

    /// Count of pixels in line
    stride: usize,
}

/// Render context
struct RenderContext<'t, 'ref_table> {
    /// Projection info holder
    camera: RenderCamera,

    shadow_camera: Option<RenderCamera>,

    /// Offset of background from foreground
    sky_background_uv_offset: Vec2f,

    /// Foreground offset
    sky_uv_offset: Vec2f,

    /// Map reference
    map: &'t bsp::Map,

    /// Table of materials
    material_table: &'t res::MaterialReferenceTable<'ref_table>,

    /// Frame structure
    frame: RenderFrame,

    /// Rasterization mode
    rasterization_mode: RasterizationMode,
}

impl<'t, 'ref_table> RenderContext<'t, 'ref_table> {

    /// Render already clipped polygon
    unsafe fn render_clipped_polygon(
        &mut self,
        is_transparent: bool,
        is_sky: bool,
        points: &[Vec5UVf],
        color: u64,
        texture: SurfaceTexture
    ) { unsafe {
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
    }}

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
    ) { unsafe {
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
        let first_line = usize::min(min_y_value.floor() as usize, self.frame.height);
        let last_line = usize::min(max_y_value.ceil() as usize, self.frame.height);

        #[derive(Copy, Clone, Default)]
        struct LineContext {
            index: usize,
            prev_xzuv: math::FVec4,
            curr_xzuv: math::FVec4,
            prev_y: f32,
            curr_y: f32,
            slope_xzuv: math::FVec4,
        }

        let mut left = LineContext {
            index: min_y_index,
            curr_xzuv: points[min_y_index].xzuv().into(),
            curr_y: points[min_y_index].y,
            ..Default::default()
        };
        let mut right = left;

        macro_rules! side_line_step {
            ($side: ident, $ind: ident) => {
                $side.index = $ind!($side.index);

                $side.prev_y = $side.curr_y;
                $side.prev_xzuv = $side.curr_xzuv;

                $side.curr_y = points[$side.index].y;
                $side.curr_xzuv = points[$side.index].xzuv().into();

                let dy = $side.curr_y - $side.prev_y;

                // Check if edge is flat
                $side.slope_xzuv = if dy <= DY_EPSILON {
                    math::FVec4::zero()
                } else {
                    ($side.curr_xzuv - $side.prev_xzuv) / dy
                };
            }
        }

        side_line_step!(left, ind_next);
        side_line_step!(right, ind_prev);

        let width = texture.width as isize >> (IS_SKY as isize);
        let height = texture.height as isize;

        // Scan for lines
        'line_loop: for pixel_y in first_line..last_line {
            // Get current pixel y
            let y = pixel_y as f32 + 0.5;

            while y > left.curr_y {
                if left.index == max_y_index {
                    break 'line_loop;
                }
                side_line_step!(left, ind_next);
            }

            while y > right.curr_y {
                if right.index == max_y_index {
                    break 'line_loop;
                }
                side_line_step!(right, ind_prev);
            }

            let left_xzuv = math::FVec4::mul_add(
                left.slope_xzuv,
                math::FVec4::from_single(y - left.prev_y),
                left.prev_xzuv,
            );
            let right_xzuv = math::FVec4::mul_add(
                right.slope_xzuv,
                math::FVec4::from_single(y - right.prev_y),
                right.prev_xzuv,
            );

            // Get left x
            let left_x = left_xzuv.x();

            // Get right x
            let right_x = right_xzuv.x();

            // Calculate hline start/end, clip it
            let start = left_x.floor() as usize;
            let end = usize::min(right_x.floor() as usize, self.frame.width);

            let pixel_row = self.frame.pixels.add(self.frame.stride * pixel_y);

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

                        let fg_u = u
                            .to_int_unchecked::<isize>()
                            .rem_euclid(width)
                            .cast_unsigned();

                        let fg_v = v
                            .to_int_unchecked::<isize>()
                            .rem_euclid(height)
                            .cast_unsigned();

                        let fg_color = *texture.data.get_unchecked(fg_v * texture.stride + fg_u);

                        // Check foreground color and fetch backround if foreground is transparent
                        if fg_color == 0 {
                            let bg_u = (u + self.sky_background_uv_offset.x)
                                .to_int_unchecked::<isize>()
                                .rem_euclid(width)
                                .wrapping_add(width)
                                .cast_unsigned();

                            let bg_v = (v + self.sky_background_uv_offset.y)
                                .to_int_unchecked::<isize>()
                                .rem_euclid(height)
                                .cast_unsigned();

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

                        let u = (xzuv.z() * z)
                            .to_int_unchecked::<isize>()
                            .rem_euclid(width)
                            .cast_unsigned();

                        let v = (xzuv.w() * z)
                            .to_int_unchecked::<isize>()
                            .rem_euclid(height)
                            .cast_unsigned();

                        src_color = *texture.data.get_unchecked(v * texture.stride + u);
                    }

                } else {
                    panic!("Invalid rasterization mode: {}", MODE);
                }

                if IS_TRANSPARENT {
                    // SSE-based transparency calculation
                    #[cfg(target_feature = "sse")]
                    {
                        use std::arch::x86_64 as arch;

                        let src = std::mem::transmute(
                            arch::_mm_set_sd(f64::from_bits(src_color))
                        );
                        let src32 = arch::_mm_cvtepi16_epi32(src);
                        let src_m = arch::_mm_mul_ps(
                            arch::_mm_cvtepi32_ps(src32),
                            arch::_mm_set1_ps(0.6)
                        );

                        let dst = std::mem::transmute(
                            arch::_mm_load_sd(
                                std::mem::transmute(pixel_ptr)
                            )
                        );
                        let dst32 = arch::_mm_cvtepi16_epi32(dst);
                        let dst_m = arch::_mm_mul_ps(
                            arch::_mm_cvtepi32_ps(dst32),
                            arch::_mm_set1_ps(0.4)
                        );

                        let sum = arch::_mm_add_ps(src_m, dst_m);
                        let res = arch::_mm_cvtps_epi32(sum);
                        let cvt = arch::_mm_packus_epi32(res, res);
                        arch::_mm_store_sd(
                            std::mem::transmute(pixel_ptr),
                            std::mem::transmute(cvt)
                        );
                    }
                    
                    #[cfg(not(target_feature = "sse"))]
                    {
                        let dst_color = *pixel_ptr;
                        let [dr, dg, db, _] = u64_into_u16(dst_color);
                        let [sr, sg, sb, _] = u64_into_u16(src_color);

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
    }}

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
            self.camera.project_sky_polygon(polygon, points, self.sky_uv_offset);
        } else {
            self.camera.project_polygon(polygon, material_u, material_v, points);
        }

        // Clip polygon invisible part
        point_dst.clear();
        if !clip_polygon_z1(points, point_dst) {
            return;
        }
        std::mem::swap(points, point_dst);

        self.camera.get_screenspace_polygon(points);

        // Clip polygon by volume clipping octagon
        point_dst.clear();
        if !clip_polygon_oct(points, point_dst, clip_oct) {
            return;
        }
        std::mem::swap(points, point_dst);

        // Just for safety
        for pt in points.iter() {
            if !pt.x.is_finite() || !pt.y.is_finite() {
                return;
            }
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

        'polygon_rendering: for surface in &volume.surfaces {

            let polygon = self.map.get_polygon(surface.polygon_id).unwrap();

            // Backface cull with shadow camera
            if let Some(shadow_camera) = self.shadow_camera.as_ref() {
                if polygon.plane.get_signed_distance(shadow_camera.location) <= 0.0 {
                    continue 'polygon_rendering;
                }
            }

            // Backface cull with standard camera
            if polygon.plane.get_signed_distance(self.camera.location) <= 0.0 {
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
                    let dist_2 = (*point - self.camera.location).length2();
    
                    if dist_2 <= min_dist_2 {
                        min_dist_2 = dist_2;
                    }
                }

                let res = (self.frame.width * self.frame.height) as f32;
    
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
                self.frame.width as f32 + CLIP_OFFSET,
                self.frame.height as f32 + CLIP_OFFSET,
            ),
        }
    }

    /// Build set of rendered polygons (with visibility check)
    /// 
    /// # Algorithm
    /// 
    /// This function recursively traverses BSP in front-to-back order,
    /// if current volume isn't inserted in PVS (Potentially Visible Set),
    /// then it's ignored. If it is, it is added in render set with it's
    /// current clipping octagon (it **is not** visible from any of next
    /// traverse elements, so current clipping octagon is the final one)
    /// and then for every portal of current volume function calculates
    /// it's screenspace bounding octagon and inserts inserts destination
    /// volume in PVS with this octagon. If destination volume is already
    /// added, it's clipping octagon is extended to fit union of inserted
    /// and previous clipping octagons.
    pub fn build_render_set(
        &self,
        bsp_root: &bsp::Bsp,
        start_volume_id: bsp::VolumeId,
        start_clip_oct: &geom::ClipOct,
        camera: &RenderCamera,
    ) -> Vec<(bsp::VolumeId, geom::ClipOct)> {
        // Render set itself
        let mut inv_render_set_value = Vec::new();
        let inv_render_set = &mut inv_render_set_value;

        // Set of Potentially visible volumes with their clip octagons
        let mut pvs = HashMap::<bsp::VolumeId, geom::ClipOct>::new();

        // Initialize PVS
        pvs.insert(start_volume_id, *start_clip_oct);

        let traverse_fn = move |volume_id| {
            let Some(volume_clip_oct) = pvs.get(&volume_id) else {
                return;
            };

            let volume_clip_oct = *volume_clip_oct;

            // Insert volume in render set
            inv_render_set.push((volume_id, volume_clip_oct));

            let volume = self.map.get_volume(volume_id).unwrap();

            'portal_rendering: for portal in &volume.portals {
                let portal_polygon = self.map
                    .get_polygon(portal.polygon_id)
                    .unwrap();

                // Perform modified backface culling
                let backface_cull_result =
                    portal_polygon.plane.get_signed_distance(camera.location) >= 0.0;
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
                        .get_signed_distance(camera.location)
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

                    camera.get_screenspace_projected_portal_polygon(
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
        };

        self.traverse_bsp(
            bsp_root,
            camera.location,
            traverse_fn,
        );

        inv_render_set_value
    }

    /// Traverse BSP in order dictated by camera_location
    pub fn traverse_bsp<VolumeFn: FnMut(bsp::VolumeId)>(
        &self,
        bsp_root: &bsp::Bsp,
        camera_location: Vec3f,
        mut volume_fn: VolumeFn,
    ) {
        let mut visit_stack = vec![bsp_root];

        while let Some(bsp) = visit_stack.pop() {
            match bsp {
                bsp::Bsp::Partition { splitter_plane, front, back } => {
                    visit_stack.extend_from_slice(
                        &match splitter_plane.get_point_relation(camera_location) {
                            geom::PointRelation::Front | geom::PointRelation::OnPlane => {
                                [back, front]
                            }
                            geom::PointRelation::Back => {
                                [front, back]
                            }
                        }
                    );
                }
                bsp::Bsp::Volume(vol) => volume_fn(*vol),
                bsp::Bsp::Void => {},
            }
        }
    }

    pub fn render(&mut self) {
        let world_bsp = self.map
            .get_world_model()
            .get_bsp();
        let screen_clip_oct = geom::ClipOct::from_clip_rect(self.get_screen_clip_rect());

        let partial_render_set_opt = if let Some(shadow_camera) = self.shadow_camera.as_ref() {
            world_bsp
                .find_volume(shadow_camera.location)
                .map(|start_volume_id| {
                    let mut render_set = self.build_render_set(
                        world_bsp,
                        start_volume_id,
                        &geom::ClipOct::from_clip_rect(self.get_screen_clip_rect()),
                        &shadow_camera
                    );

                    // Make render set unordered
                    let mut unordered_render_set = render_set
                        .drain(..)
                        .map(|(id, _)| id)
                        .collect::<HashSet<bsp::VolumeId>>();

                    // Order it
                    self.traverse_bsp(
                        world_bsp,
                        self.camera.location,
                        |volume_id| {
                            if unordered_render_set.remove(&volume_id) {
                                render_set.push((volume_id, screen_clip_oct));
                            }
                        },
                    );

                    render_set
                })
        } else {
            world_bsp
                .find_volume(self.camera.location)
                .map(|start_volume_id| {
                    self.build_render_set(
                        world_bsp,
                        start_volume_id,
                        &screen_clip_oct,
                        &self.camera
                    )
                })
        };

        let inv_render_set = partial_render_set_opt
            .unwrap_or_else(|| {
                let mut render_set = Vec::new();
                let render_set_ref = &mut render_set;

                self.traverse_bsp(
                    world_bsp,
                    self.camera.location,
                    move |volume_id| {
                        render_set_ref.push((volume_id, screen_clip_oct));
                    },
                );

                render_set
            });

        let mut points = Vec::new();
        let mut point_dst = Vec::new();
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
}

/// Input render message
pub enum RenderInputMessage {
    NewFrame {
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
        shadow_camera: Option<camera::Camera>,
        camera: camera::Camera,
        projection_matrix: Mat4f,
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
    data: &'t mut [u64],
    src: res::ImageRef,
    light: f32
) -> SurfaceTexture<'t> {
    for (src, dst) in Iterator::zip(src.data.iter(), data.iter_mut()) {
        let [r, g, b, _] = src.to_le_bytes();

        *dst = u64_from_u16(if ENABLE_LIGHTING {
            [
                unsafe { (r as f32 * light).to_int_unchecked::<u16>() },
                unsafe { (g as f32 * light).to_int_unchecked::<u16>() },
                unsafe { (b as f32 * light).to_int_unchecked::<u16>() },
                0
            ]
        } else {
            [r as u16, g as u16, b as u16, 0]
        });
    }

    SurfaceTexture {
        width: src.width,
        height: src.height,
        stride: src.width,
        data: data,
    }
}

fn build_surface_texture<'t>(
    data: &'t mut Vec<u64>,
    src: res::ImageRef,
    light: f32,
    enable_lighting: bool
) -> SurfaceTexture<'t> {
    data.resize(src.width * src.height, 0);

    if enable_lighting {
        build_surface_texture_impl::<true>(data, src, light)
    } else {
        build_surface_texture_impl::<false>(data, src, light)
    }
}

fn hdr_to_ldr_clamp(hdr: &[u64], ldr: &mut [u32]) {
    for (src, dst) in Iterator::zip(hdr.iter(), ldr.iter_mut()) {
        let [r, g, b, _] = u64_into_u16(*src);
        *dst = u32::from_le_bytes([
            r as u8,
            g as u8,
            b as u8,
            0
        ]);
    }
}

fn hdr_to_ldr_tonemap(hdr: &[u64], ldr: &mut [u32]) {
    #[cfg(not(target_feature = "sse2"))]
    {
        hdr_to_ldr_clamp(hdr, ldr);
        return;
    }

    for (src_ptr, dst_ptr) in Iterator::zip(hdr.iter(), ldr.iter_mut()) {
        use std::arch::x86_64 as arch;

        unsafe {
            let src16 = std::mem::transmute(
                arch::_mm_load_sd(
                    std::mem::transmute(src_ptr)
                )
            );
            let src32 = arch::_mm_cvtepi16_epi32(src16);
            let src = arch::_mm_cvtepi32_ps(src32);

            // rgba / 256.0 + exposure
            let rgba_norm_aexp = arch::_mm_fmadd_ps(
                src,
                arch::_mm_set1_ps(1.0 / 256.0),
                arch::_mm_set1_ps(0.5) // exposure
            );

            // rgba / (rgba / 256.0 + exposure)
            let mapped = arch::_mm_div_ps(src, rgba_norm_aexp);

            let mapped32 = arch::_mm_cvtps_epi32(mapped);
            let mapped16 = arch::_mm_packus_epi32(mapped32, mapped32);
            let mapped8 = arch::_mm_packus_epi16(mapped16, mapped16);

            arch::_mm_store_ss(
                std::mem::transmute(dst_ptr),
                std::mem::transmute(mapped8)
            );
        }
    }
}

pub fn hdr_to_ldr(hdr: &[u64], ldr: &mut[u32], enable_tonemapping: bool) {
    if enable_tonemapping {
        hdr_to_ldr_tonemap(hdr, ldr);
    } else {
        hdr_to_ldr_clamp(hdr, ldr);
    }
}

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    // Enable/disable map caching
    let do_enable_map_caching = true;

    // Synchronize visible-set-building and projection cameras
    let mut shadow_camera: Option<camera::Camera> = None;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut rasterization_mode = RasterizationMode::Standard;

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
        let map_name = "q1/e1m1";
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
    let mut map = map;
    map.bake_lightmaps();
    let map = Arc::new(map);

    let material_table = {
        let mut wad_file = std::fs::File::open("temp/q1/gfx/base.wad").unwrap();

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

    let mut timer = timer::Timer::new();
    let mut input = input::Input::new();
    let mut camera = camera::Camera::new();

    // camera.location = Vec3f::new(-174.0, 2114.6, -64.5); // -200, 2000, -50
    // camera.direction = Vec3f::new(-0.4, 0.9, 0.1);

    camera.location = Vec3f::new(1402.4, 1913.7, -86.3);
    camera.direction = Vec3f::new(-0.74, 0.63, -0.24);

    // camera.direction = Vec3f::new(0.055, -0.946, 0.320); // (-0.048328593, -0.946524262, 0.318992347)

    // camera.location = Vec3f::new(-72.9, 698.3, -118.8);
    // camera.direction = Vec3f::new(0.37, 0.68, 0.63);

    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // Camera used for visible set building

    // Buffer that contains software-rendered pixels
    let mut hdr_frame_buffer = Vec::<u64>::new();

    // LDR framebuffer
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
            let mut timer = timer::Timer::new();

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
                        shadow_camera,
                        camera,
                        projection_matrix,
                        rasterization_mode
                    } => {
                        timer.response();

                        let time = timer.get_time();

                        // Clear framebuffer
                        unsafe {
                            std::ptr::write_bytes(
                                frame_buffer.as_mut_ptr(),
                                0,
                                frame_buffer.len()
                            );
                        }

                        // Very long function call, actually
                        let mut render_context = RenderContext {
                            camera: RenderCamera {
                                view_projection: camera.compute_view_matrix() * projection_matrix,
                                location: camera.location,
                                half_fw: width as f32 * 0.5,
                                half_fh: height as f32 * 0.5,
                            },

                            shadow_camera: shadow_camera.map(|shadow_camera| RenderCamera {
                                view_projection: shadow_camera.compute_view_matrix() * projection_matrix,
                                location: shadow_camera.location,
                                half_fw: width as f32 * 0.5,
                                half_fh: height as f32 * 0.5,
                            }),

                            frame: RenderFrame {
                                pixels: frame_buffer.as_mut_ptr(),
                                width: width as usize,
                                height: height as usize,
                                stride: width as usize
                            },

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
                        };

                        render_context.render();

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

        // Toggle shadow camera
        if input.is_key_clicked(Scancode::Num9) {
            shadow_camera = if shadow_camera.is_none() {
                Some(camera)
            } else {
                None
            };
        }

        if input.is_key_clicked(Scancode::Num0) {
            rasterization_mode = rasterization_mode.next();
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        /// Frame rendering scale
        const FRAME_SCALE: usize = 2;

        let (frame_width, frame_height) = (
            window_width / FRAME_SCALE,
            window_height / FRAME_SCALE,
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
            shadow_camera,
            camera,
            projection_matrix,
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
                    // Resize ldr buffer to fit screen size
                    ldr_frame_buffer.resize(width as usize * height as usize, 0);

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
                        .str(16, 16, &format!("SC={}, RM={}",
                            shadow_camera.is_some() as u32,
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
