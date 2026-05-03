//! Project root module

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Render and Physics BSP's building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with BSP's.

use std::{collections::{HashMap, HashSet}, io::Read, sync::{Arc, mpsc}};
use math::{Mat4f, Vec2f, Vec3f};
use zerocopy::IntoBytes;

use crate::{frame_slice::{FrameSlice, FrameSliceMut}, math::{FVec4, Vec2}};

pub mod math;
pub mod system_font;
pub mod frame_slice;
pub mod rand;
pub mod geom;
pub mod bsp;
pub mod map;
pub mod res;
pub mod camera;
pub mod timer;
pub mod input;
pub mod flags;

/// Different rasterization modes
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum RasterizationMode {
    /// Full rendering
    Full = 0,

    /// Monochromatic polygons
    Monochrome = 1,

    /// Overdraw (brighter => more overdraw)
    Overdraw = 2,

    /// Inverse depth value
    Depth = 3,

    /// Rasterize with UV checker
    UV = 4,

    /// Unlit textures
    Textures = 5,

    /// Lightmaps only
    Lightmaps = 6,
}

impl RasterizationMode {
    // Rasterization mode count
    const COUNT: u32 = 7;

    /// Build rasterization mode from u32
    const fn from_u32(n: u32) -> Option<RasterizationMode> {
        Some(match n {
            0 => Self::Full,
            1 => Self::Monochrome,
            2 => Self::Overdraw,
            3 => Self::Depth,
            4 => Self::UV,
            5 => Self::Textures,
            6 => Self::Lightmaps,
            _ => return None,
        })
    }

    /// Get next rasterization mode
    pub const fn next(self) -> RasterizationMode {
        Self::from_u32((self as u32 + 1) % Self::COUNT).unwrap()
    }
}

/// Make u64 from [u16; 4]
pub const fn u64_from_u16(value: [u16; 4]) -> u64 {
    ( value[0] as u64       ) |
    ((value[1] as u64) << 16) |
    ((value[2] as u64) << 32) |
    ((value[3] as u64) << 48)
}

/// Convert u64 into [u16; 4]
pub const fn u64_into_u16(value: u64) -> [u16; 4] {
    [
        ((value      ) & 0xFFFF) as u16,
        ((value >> 16) & 0xFFFF) as u16,
        ((value >> 32) & 0xFFFF) as u16,
        ((value >> 48) & 0xFFFF) as u16,
    ]
}

/// In-render vertex structure
#[derive(Copy, Clone)]
pub struct Vertex {
    /// Vertex position
    pub position: Vec3f,

    /// Vertex texture coordinate
    pub tex_coord: Vec2f,
}

impl Vertex {
    /// Get vertex XZUV (used during rasterization process)
    pub fn xzuv(self) -> math::FVec4 {
        math::FVec4::new(
            self.position.x(),
            self.position.z(),
            self.tex_coord.x(),
            self.tex_coord.y()
        )
    }
}

impl From<f32> for Vertex {
    fn from(v: f32) -> Self {
        Self {
            position: v.into(),
            tex_coord: v.into(),
        }
    }
}

impl std::ops::Add<Self> for Vertex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            position: self.position + rhs.position,
            tex_coord: self.tex_coord + rhs.tex_coord
        }
    }
}

impl std::ops::Sub<Self> for Vertex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            position: self.position - rhs.position,
            tex_coord: self.tex_coord - rhs.tex_coord
        }
    }
}

impl std::ops::Mul<Self> for Vertex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            position: self.position * rhs.position,
            tex_coord: self.tex_coord * rhs.tex_coord,
        }
    }
}

/// Clip type-generic polygon by some linear vertex norm function
/// # Note
/// `cmp` must be ordering function, `norm` must be linear in respect of `V` operators
fn clip_polygon<V>(
    points: &mut Vec<V>,
    temp: &mut Vec<V>,
    value: f32,
    cmp: impl Fn(f32, f32) -> bool,
    norm: impl Fn(V) -> f32,
) -> bool
where
    V: Copy
        + std::ops::Add<V, Output = V>
        + std::ops::Sub<V, Output = V>
        + std::ops::Mul<V, Output = V>
        + From<f32>
{
    temp.clear();
    for index in 0..points.len() {
        let curr = points[index];
        let next = points[(index + 1) % points.len()];

        if cmp(norm(curr), value) {
            temp.push(curr);

            if cmp(value, norm(next)) {
                let t = (value - norm(curr)) / (norm(next) - norm(curr));
                temp.push((next - curr) * V::from(t) + curr);
           }
        } else if cmp(norm(next), value) {
            let t = (value - norm(curr)) / (norm(next) - norm(curr));
            temp.push((next - curr) * V::from(t) + curr);
        }
    }
    std::mem::swap(points, temp);

    points.len() >= 3
}

/// Clip polygon by octagon
/// # Return
/// True if there are points rest
fn clip_polygon_oct(
    vertices: &mut Vec<Vertex>,
    temp: &mut Vec<Vertex>,
    clip_oct: &geom::BoundOct
) -> bool {
    // Norm functions
    fn norm_x     (vt: Vertex) -> f32 { vt.position.x() }
    fn norm_y     (vt: Vertex) -> f32 { vt.position.y() }
    fn norm_y_a_x (vt: Vertex) -> f32 { vt.position.y() + vt.position.x() }
    fn norm_y_s_x (vt: Vertex) -> f32 { vt.position.y() - vt.position.x() }

    // Comparison functions
    fn ge(l: f32, r: f32) -> bool { l >= r }
    fn le(l: f32, r: f32) -> bool { l <= r }

    // Utilize '&&' hands calculation rules to stop clipping if there's <= 3 points
    
           clip_polygon(vertices, temp, clip_oct.min.x(), ge, norm_x)
        && clip_polygon(vertices, temp, clip_oct.max.x(), le, norm_x)
        && clip_polygon(vertices, temp, clip_oct.min.y(), ge, norm_y)
        && clip_polygon(vertices, temp, clip_oct.max.y(), le, norm_y)
        && clip_polygon(vertices, temp, clip_oct.min.z(), ge, norm_y_s_x)
        && clip_polygon(vertices, temp, clip_oct.max.z(), le, norm_y_s_x)
        && clip_polygon(vertices, temp, clip_oct.min.w(), ge, norm_y_a_x)
        && clip_polygon(vertices, temp, clip_oct.max.w(), le, norm_y_a_x)
}

/// Camera that (additionally) holds info about projection and frame size
pub struct RenderCamera {
    /// Projection * View matrix
    pub view_projection: Mat4f,

    /// Camera location
    pub location: Vec3f,

    /// Frame width half
    pub half_fw: f32,

    /// Frame height half
    pub half_fh: f32,
}

impl RenderCamera {
    /// Simplified function for vertex without portals
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
        clip_polygon(points, point_dst, 1.0, |l, r| l > r, |p| p.z());

        point_dst.clear();
        for pt in points.iter() {
            let inv_z = pt.z().recip();

            point_dst.push(Vec3f::new(
                (1.0 + pt.x() * inv_z) * self.half_fw,
                (1.0 - pt.y() * inv_z) * self.half_fh,
                inv_z,
            ));
        }
    }

    /// Make world->camera space transition sky for polygon
    pub fn project_sky_polygon(
        &self,
        polygon: &geom::Polygon,
        point_dst: &mut Vec<Vertex>,
        uv_offset: Vec2f,
    ) {
        // Apply reprojection for sky polygon
        for point in polygon.points.iter() {
            // Skyplane distance
            const DIST: f32 = 128.0;

            let rp = *point - self.location;

            // Copy z sign to correct back-z case (s_dist - signed distance)
            let s_dist = DIST.copysign(rp.z());

            let u = rp.x() / rp.z() * s_dist;
            let v = rp.y() / rp.z() * s_dist;

            point_dst.push(Vertex {
                // Vector-only transform is used here to don't add camera location back.
                position: self.view_projection.transform_vector(Vec3f::new(u, v, s_dist)),
                tex_coord: Vec2f::new(u, v) + uv_offset,
            });
        }
    }

    /// Apply projection to polygon
    pub fn project_polygon(
        &self,
        polygon: &geom::Polygon,
        u: geom::Plane,
        v: geom::Plane,
        point_dst: &mut Vec<Vertex>,
    ) {
        point_dst.clear();
        for point in polygon.points.iter() {
            point_dst.push(Vertex {
                position: self.view_projection.transform_point(*point),
                tex_coord: Vec2f::new(u.get_signed_distance(*point), v.get_signed_distance(*point)),
            });
        }
    }

    /// Reproject from camera to screen space
    pub fn get_screenspace_polygon(
        &self,
        polygon: &mut [Vertex],
    ) {
        for pt in polygon {
            let inv_z = pt.position.z().recip();

            *pt = Vertex {
                position: Vec3f::new(
                    (1.0 + pt.position.x() * inv_z) * self.half_fw,
                    (1.0 - pt.position.y() * inv_z) * self.half_fh,
                    inv_z,
                ),
                tex_coord: pt.tex_coord * inv_z.into(),
            };
        }
    }
}

/// Render context
struct RenderContext<'t, 'ref_table> {
    /// Projection info holder
    camera: RenderCamera,

    /// Camera used for calculation of clipping
    shadow_camera: Option<RenderCamera>,

    /// Offset of background from foreground
    sky_background_uv_offset: Vec2f,

    /// Foreground offset
    sky_uv_offset: Vec2f,

    /// Map reference
    map: &'t bsp::Map,

    /// Table of materials
    material_table: &'t res::MaterialReferenceTable<'ref_table>,

    /// Rendering destination
    frame: FrameSliceMut<'t, u64>,

    /// Rasterization mode
    rasterization_mode: RasterizationMode,
}

impl<'t, 'ref_table> RenderContext<'t, 'ref_table> {
    /// Call pixel_fn for all polygon pixels
    fn render_clipped_polygon_impl<PixelFn: FnMut(&mut u64, FVec4)>(
        &mut self,
        vertices: &[Vertex],
        mut pixel_fn: PixelFn
    ) {
        // Find polygon min/max (e.g. split into left and right parts)
        let (min_y_index, min_y_value, max_y_index, max_y_value) = {
            let mut min_y_index = 0;
            let mut max_y_index = 0;

            let mut min_y = vertices[0].position.y();
            let mut max_y = min_y;

            for (index, y) in vertices.iter().map(|v| v.position.y()).enumerate() {
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
        let last_line = usize::min(max_y_value.ceil() as usize, self.frame.height());
        let first_line = usize::min(min_y_value.floor() as usize, last_line);

        /// Line vertical traversal context
        #[derive(Copy, Clone, Default)]
        struct LineContext {
            index: usize,
            prev_xzuv: math::FVec4,
            curr_xzuv: math::FVec4,
            prev_y: f32,
            curr_y: f32,
            d_xzuv: math::FVec4,
        }

        impl LineContext {
            /// Step to the next point
            fn next_point<const IND_NEXT: bool>(&mut self, points: &[Vertex]) {

                self.index += if IND_NEXT { 1 } else { points.len() - 1 };
                self.index %= points.len();

                self.prev_y = self.curr_y;
                self.prev_xzuv = self.curr_xzuv;

                self.curr_y = points[self.index].position.y();
                self.curr_xzuv = points[self.index].xzuv();

                let dy = self.curr_y - self.prev_y;

                // Check if edge is flat
                self.d_xzuv = if dy <= DY_EPSILON {
                    math::FVec4::zero()
                } else {
                    (self.curr_xzuv - self.prev_xzuv) / dy.into()
                };
            }
        }

        let mut left = LineContext {
            index: min_y_index,
            curr_xzuv: vertices[min_y_index].xzuv(),
            curr_y: vertices[min_y_index].position.y(),
            ..Default::default()
        };
        let mut right = left;

        left.next_point::<true>(vertices);
        right.next_point::<false>(vertices);

        // Scan for lines
        'line_loop: for pixel_y in first_line..last_line {
            // Get current pixel y
            let y = pixel_y as f32 + 0.5;

            while y > left.curr_y {
                if left.index == max_y_index {
                    break 'line_loop;
                }
                left.next_point::<true>(vertices);
            }

            while y > right.curr_y {
                if right.index == max_y_index {
                    break 'line_loop;
                }
                right.next_point::<false>(vertices);
            }

            let left_xzuv = math::FVec4::mul_add(
                left.d_xzuv,
                math::FVec4::broadcast(y - left.prev_y),
                left.prev_xzuv,
            );

            let right_xzuv = math::FVec4::mul_add(
                right.d_xzuv,
                math::FVec4::broadcast(y - right.prev_y),
                right.prev_xzuv,
            );

            let left_x = left_xzuv.x();
            let right_x = right_xzuv.x();

            let dx = right_x - left_x;

            let d_xzuv = if dx <= DY_EPSILON {
                math::FVec4::zero()
            } else {
                (right_xzuv - left_xzuv) / dx.into()
            };

            // Destination hline part
            let pixel_row_slice = {
                let end = usize::min(right_x.floor() as usize, self.frame.width());
                let start = usize::min(left_x.floor() as usize, end);

                self.frame
                    .get_mut(pixel_y)
                    .unwrap()
                    .get_mut(start..end)
                    .unwrap()
            };

            // Calculate pixel position 'remainder'
            let pixel_off = left_x.fract() - 0.5;

            for (x, p) in pixel_row_slice.iter_mut().enumerate() {
                pixel_fn(p, math::FVec4::mul_add(
                    d_xzuv,
                    math::FVec4::broadcast(x as f32 - pixel_off),
                    left_xzuv
                ));
            }
        }
    }

    /// Render polygon
    unsafe fn render_clipped_polygon(
        &mut self,
        is_transparent: bool,
        is_sky: bool,
        points: &[Vertex],
        color: u64,
        texture: FrameSlice<u64>
    ) {
        /// Add transparency layer after fragment function
        fn wrap_transparent(mut f: impl FnMut(&mut u64, FVec4)) -> impl FnMut(&mut u64, FVec4) {
            move |pixel_ptr: &mut u64, xzuv: FVec4| {
                // uugh transparency performance...
                let mut src_color = *pixel_ptr;
                f(&mut src_color, xzuv);

                // SIMD-based transparency
                #[cfg(target_feature = "sse")]
                unsafe {
                    use std::arch::x86_64 as arch;

                    let src = std::mem::transmute::<arch::__m128d, arch::__m128i>(arch::_mm_set_sd(f64::from_bits(src_color)));
                    let src32 = arch::_mm_cvtepi16_epi32(src);
                    let src_m = arch::_mm_mul_ps(
                        arch::_mm_cvtepi32_ps(src32),
                        arch::_mm_set1_ps(0.6)
                    );

                    let dst = std::mem::transmute::<arch::__m128d, arch::__m128i>(
                        arch::_mm_load_sd((pixel_ptr as *mut u64).cast())
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
                        pixel_ptr as *mut u64 as *mut f64,
                        std::mem::transmute::<arch::__m128i, arch::__m128d>(cvt)
                    );
                }

                // Fallback (slow) transparency
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
            }
        }

        // Macro that wraps actual rendering function call
        macro_rules! run {
            ($func: expr) => {
                {
                    // Make life of macro expander a bit happier
                    let f = $func;

                    if is_transparent {
                        self.render_clipped_polygon_impl(points, wrap_transparent(f));
                    } else {
                        self.render_clipped_polygon_impl(points, f);
                    }
                }
            };
        }

        match self.rasterization_mode {
            RasterizationMode::Monochrome => run!(move |dst: &mut u64, _| {
                *dst = color;
            }),
            RasterizationMode::Overdraw => run!(|dst: &mut u64, _| {
                *dst = dst.wrapping_add(0x0010_0010_0010u64);
            }),
            RasterizationMode::Depth => run!(|dst: &mut u64, xzuv: FVec4| {
                let color = (xzuv.y() * 25500.0) as u16;
                *dst = u64_from_u16([color; 4]);
            }),
            RasterizationMode::UV => run!(|dst: &mut u64, xzuv: FVec4| {
                let inv_z = xzuv.y().recip();
                let u = xzuv.z() * inv_z;
                let v = xzuv.w() * inv_z;

                let xi = (u as i64 & 0xFF) as u8;
                let yi = (v as i64 & 0xFF) as u8;

                *dst = color.wrapping_mul((((xi >> 5) ^ (yi >> 5)) & 1) as u64);
            }),
            RasterizationMode::Full | RasterizationMode::Textures | RasterizationMode::Lightmaps => {
                if is_sky {
                    let width = texture.width() as isize >> (is_sky as isize);
                    let height = texture.height() as isize;
                    let uv_offset = self.sky_uv_offset;
                    let background_uv_offset = self.sky_background_uv_offset;

                    run!(move |dst: &mut u64, xzuv: FVec4| {
                        let inv_z = xzuv.y().recip();
                        let u = xzuv.z() * inv_z + uv_offset.x();
                        let v = xzuv.w() * inv_z + uv_offset.y();

                        let fg_u = unsafe { u.to_int_unchecked::<isize>() }
                            .rem_euclid(width)
                            .cast_unsigned();

                        let fg_v = unsafe { v.to_int_unchecked::<isize>() }
                            .rem_euclid(height)
                            .cast_unsigned();

                        let fg_color = *unsafe { texture.get2_unchecked(fg_v, fg_u) };

                        // Check foreground color and fetch backround if foreground is transparent
                        if fg_color != 0 {
                            *dst = fg_color;
                            return;
                        }

                        let bg_u = unsafe { (u + background_uv_offset.x()).to_int_unchecked::<isize>() }
                            .rem_euclid(width)
                            .wrapping_add(width)
                            .cast_unsigned();

                        let bg_v = unsafe { (v + background_uv_offset.y()).to_int_unchecked::<isize>() }
                            .rem_euclid(height)
                            .cast_unsigned();

                        *dst = *unsafe { texture.get2_unchecked(bg_v, bg_u) };
                    })
                } else {
                    let max_u = texture.width() - 1;
                    let max_v = texture.height() - 1;

                    run!(move |dst: &mut u64, xzuv: FVec4| {
                        let inv_z = xzuv.y().recip();

                        // min just for get2_unchecked safety
                        let u = unsafe { (xzuv.z() * inv_z).to_int_unchecked::<isize>() }
                            .cast_unsigned().min(max_u);
                        let v = unsafe { (xzuv.w() * inv_z).to_int_unchecked::<isize>() }
                            .cast_unsigned().min(max_v);

                        *dst = *unsafe { texture.get2_unchecked(v, u) };
                    })
                }
            }
        }
    }

    /// Render volume surface
    fn render_surface(
        &mut self,
        surface: &bsp::Surface,
        clip_oct: &geom::BoundOct,
        vertices: &mut Vec<Vertex>,
        vertices_dst: &mut Vec<Vertex>,
        surface_texture_data: &mut Vec<u64>,
    ) {
        let polygon = self.map.get_polygon(surface.polygon_id).unwrap();

        // Perform backface culling with shadow camera
        if let Some(shadow_camera) = self.shadow_camera.as_ref()
            && polygon.plane.get_signed_distance(shadow_camera.location) <= 0.0 {

            return;
        }

        // Backface cull with standard camera
        if polygon.plane.get_signed_distance(self.camera.location) <= 0.0 {
            return;
        }

        // Get material texture
        let texture = self.material_table
            .get_texture(surface.material_id)
            .unwrap();

        // Find mip index (sky uses 0 by default)
        let mip_index = if surface.is_sky() {
            0
        } else {
            let mut min_dist_2 = f32::MAX;

            for point in &polygon.points {
                let dist_2 = (*point - self.camera.location).length2();

                if dist_2 <= min_dist_2 {
                    min_dist_2 = dist_2;
                }
            }

            let res = (self.frame.width() * self.frame.height()) as f32;

            // Strange, but...
            ((min_dist_2 / res).log2() / 2.0 + 1.3) as usize
        };

        // Recalculate mip index
        let mip_index = mip_index.min(texture.mip_levels() - 1);

        // Image-induced UV scale is ignored now...
        let (image, image_uv_scale) = texture.get_mip_level(mip_index).unwrap();

        vertices.clear();

        // Projection and surface UVs
        if surface.is_sky() {
            self.camera.project_sky_polygon(polygon, vertices, self.sky_uv_offset);
        } else {
            self.camera.project_polygon(polygon, surface.u / image_uv_scale, surface.v / image_uv_scale, vertices);
        }

        // Clip polygon by Z=1
        if !clip_polygon(vertices, vertices_dst, 1.0, |l, r| l > r, |v| v.position.z()) {
            return;
        }

        // Project polygon to the screen space
        self.camera.get_screenspace_polygon(vertices);

        // Clip polygon by volume clipping octagon
        if !clip_polygon_oct(vertices, vertices_dst, clip_oct) {
            return;
        }

        // Just for safety
        for pt in vertices.iter() {
            if !pt.position.x().is_finite() || !pt.position.y().is_finite() {
                return;
            }
        }

        // Calculate simple per-face lighting used for during monochrome rendering
        let static_light = {
            let diffuse = Vec3f::new(0.30, 0.47, 0.80)
                .normalized()
                .dot(polygon.plane.normal)
                .abs()
                .min(0.99);

            diffuse * 0.9 + 0.09
        };

        // Calculate color from light and material color
        let static_color = {
            let color = self.material_table.get_color(surface.material_id).unwrap().to_le_bytes();
            u64_from_u16([
                (color[0] as f32 * static_light) as u16,
                (color[1] as f32 * static_light) as u16,
                (color[2] as f32 * static_light) as u16,
                0
            ])
        };

        let needs_surface_texture = match self.rasterization_mode {
            // Does not build lightmaps for special rasterization modes
            RasterizationMode::Depth | RasterizationMode::Overdraw | RasterizationMode::UV | RasterizationMode::Monochrome => {
                false
            }

            // These rasterization modes are actually about texture building
            RasterizationMode::Full | RasterizationMode::Lightmaps | RasterizationMode::Textures => {
                true
            }
        };

        let surface_texture = if needs_surface_texture && !surface.is_sky() {
            // Map tex_coords into space-linear state
            for vt in vertices.iter_mut() {
                vt.tex_coord *= vt.position.z().recip().into();
            }

            // Get UV bound rectangle and use it for surface used subrange recalculation
            let uv_bounds = geom::BoundRect::for_points(vertices.iter().map(|v| v.tex_coord));

            let uv_int_min = uv_bounds.min.map(|v| v.floor() as isize);
            let uv_int_max = uv_bounds.max.map(|v| v.ceil() as isize);
            let texture_res = (uv_int_max - uv_int_min).map(|v| v.cast_unsigned().max(1));

            // Calculate difference between UVimage and UVtexture
            let image_uv_off = Vec2::new(
                uv_int_min.x().rem_euclid(image.width() as isize).cast_unsigned(),
                uv_int_min.y().rem_euclid(image.height() as isize).cast_unsigned()
            );

            // Replace UVimage with UVtexture * inv_z (e.g. with screen-linear UVs)
            for vt in vertices.iter_mut() {
                vt.tex_coord = (vt.tex_coord - uv_int_min.map(|v| v as f32)) * vt.position.z().into();
            }

            // Resize surface texture to fit surface resolution
            surface_texture_data.resize(texture_res.x() * texture_res.y(), 0);
            let mut texture = FrameSliceMut::new(
                texture_res.x(),
                texture_res.y(),
                texture_res.x(),
                surface_texture_data.as_mut_slice()
            );

            match self.rasterization_mode {
                RasterizationMode::Full => {
                    if let Some(lightmap) = surface.lightmap.as_ref() {
                        // Build surface texture using target
                        build_surface_texture_impl::<true, true>(
                            texture.reborrow(),
                            image,
                            image_uv_off,
                            lightmap.image.as_slice(),
                            (uv_int_min.map(|i| i) - lightmap.uv_min.map(|i| i >> mip_index)).map(|i| i.max(0).cast_unsigned()),
                            3 - mip_index,
                        );
                    } else {
                        build_surface_texture_static(
                            texture.reborrow(),
                            image,
                            image_uv_off,
                            static_light,
                        );
                    }
                }
                RasterizationMode::Textures => {
                    build_surface_texture_impl::<true, false>(
                        texture.reborrow(),
                        image,
                        image_uv_off,
                        FrameSlice::empty(),
                        Vec2::new(0, 0),
                        0
                    );
                }
                RasterizationMode::Lightmaps => {
                    if let Some(lightmap) = surface.lightmap.as_ref() {
                        build_surface_texture_impl::<false, true>(
                            texture.reborrow(),
                            FrameSlice::empty(),
                            Vec2::new(0, 0),
                            lightmap.image.as_slice(),
                            (uv_int_min.map(|i| i << mip_index) - lightmap.uv_min).map(|i| i.max(0).cast_unsigned() >> mip_index),
                            3 - mip_index,
                        );
                    } else {
                        build_surface_texture_static(
                            texture.reborrow(),
                            image,
                            image_uv_off,
                            static_light,
                        );
                    }
                }

                // Should not be there
                _ => unreachable!("Trying to build surface in mode not requiring it."),
            }


            texture.into()
        } else if needs_surface_texture {
            let width = image.width();
            let height = image.height();

            // TODO: Cache sky textures, rebuilding'em every time is a bit dogshit in terms of performance.
            surface_texture_data.resize(width * height, 0);
            let mut texture = FrameSliceMut::new(width, height, width, surface_texture_data.as_mut_slice());

            // Build simple surface texture without UV remapping
            build_surface_texture_plain(texture.reborrow(), image);

            texture.into()
        } else {
            FrameSlice::empty()
        };

        // Rasterize polygon
        unsafe {
            self.render_clipped_polygon(
                surface.is_transparent(),
                surface.is_sky(),
                vertices,
                static_color,
                surface_texture
            );
        }
    }

    /// Get main clipping rectangle
    fn get_screen_clip_rect(&self) -> geom::BoundRect {
        /// Clipping offset
        const CLIP_OFFSET: f32 = 0.2;

        // Surface clipping rectangle
        geom::BoundRect {
            min: Vec2f::new(
                CLIP_OFFSET,
                CLIP_OFFSET,
            ),
            max: Vec2f::new(
                self.frame.width() as f32 + CLIP_OFFSET,
                self.frame.height() as f32 + CLIP_OFFSET,
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
        start_clip_oct: &geom::BoundOct,
        camera: &RenderCamera,
    ) -> Vec<(bsp::VolumeId, geom::BoundOct)> {
        // Render set itself
        let mut inv_render_set_value = Vec::new();
        let inv_render_set = &mut inv_render_set_value;

        // Potentially Visible volume (with corresponding clip octagon) Set
        let mut pvs = HashMap::<bsp::VolumeId, geom::BoundOct>::new();

        // Initialize PVS
        pvs.insert(start_volume_id, *start_clip_oct);

        let mut polygon_points = Vec::with_capacity(32);
        let mut proj_polygon_points = Vec::with_capacity(32);

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
                let backface_cull_result = portal_polygon.plane.get_signed_distance(camera.location) >= 0.0;
                if backface_cull_result != portal.is_facing_front {
                    continue 'portal_rendering;
                }

                let clip_oct = 'portal_validation: {
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

                    polygon_points.clear();
                    polygon_points.extend_from_slice(&portal_polygon.points);

                    if !portal.is_facing_front {
                        polygon_points.reverse();
                    }

                    proj_polygon_points.clear();
                    camera.get_screenspace_projected_portal_polygon(
                        &mut polygon_points,
                        &mut proj_polygon_points
                    );

                    // Check if it's even a polygon
                    if proj_polygon_points.len() < 3 {
                        continue 'portal_rendering;
                    }

                    let proj_oct = geom::BoundOct::for_points(proj_polygon_points.iter().map(|v| Vec2f::new(v.x(), v.y())));

                    let Some(clip_oct) = geom::BoundOct::intersection(
                        &volume_clip_oct,
                        &proj_oct.extend(0.1, 0.1, 0.1, 0.1)
                    ) else {
                        continue 'portal_rendering;
                    };

                    clip_oct
                };

                // Insert clipping octagon in PVS
                match pvs.entry(portal.dst_volume_id) {
                    std::collections::hash_map::Entry::Occupied(mut occupied) => {
                        let existing_rect: &mut geom::BoundOct = occupied.get_mut();
                        *existing_rect = existing_rect.union(&clip_oct);
                    }
                    std::collections::hash_map::Entry::Vacant(vacant) => {
                        vacant.insert(clip_oct);
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
    pub fn traverse_bsp<TraverseFn: FnMut(bsp::VolumeId)>(
        &self,
        bsp_root: &bsp::Bsp,
        camera_location: Vec3f,
        mut volume_fn: TraverseFn,
    ) {
        let mut visit_stack = vec![bsp_root];

        while let Some(bsp) = visit_stack.pop() {
            match bsp {
                bsp::Bsp::Partition { splitter_plane, front, back } => {
                    match splitter_plane.get_point_relation(camera_location) {
                        geom::PointRelation::Front | geom::PointRelation::OnPlane => {
                            visit_stack.push(back);
                            visit_stack.push(front);
                        }
                        geom::PointRelation::Back => {
                            visit_stack.push(front);
                            visit_stack.push(back);
                        }
                    }
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
        let screen_clip_oct = geom::BoundOct::from_clip_rect(self.get_screen_clip_rect());

        let partial_render_set_opt = if let Some(shadow_camera) = self.shadow_camera.as_ref() {
            world_bsp
                .find_volume(shadow_camera.location)
                .map(|start_volume_id| {
                    let mut render_set = self.build_render_set(
                        world_bsp,
                        start_volume_id,
                        &geom::BoundOct::from_clip_rect(self.get_screen_clip_rect()),
                        shadow_camera
                    );

                    // Unorder render set
                    let mut unordered_render_set = render_set
                        .drain(..)
                        .map(|(id, _)| id)
                        .collect::<HashSet<bsp::VolumeId>>();

                    // Reorder render set again
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
                    // Move is used to move screen_clip_oct here
                    move |volume_id| {
                        render_set_ref.push((volume_id, screen_clip_oct));
                    },
                );

                render_set
            });

        // Pre-allocate memory to reduce total transient allocation number.
        let mut points = Vec::with_capacity(32);
        let mut point_dst = Vec::with_capacity(32);
        let mut surface_texture = Vec::new();

        // Render volumes
        for (volume_id, volume_clip_oct) in inv_render_set.iter().rev() {
            let volume = self.map.get_volume(*volume_id).unwrap();

            for surface in volume.surfaces.iter() {
                self.render_surface(
                    surface,
                    volume_clip_oct,
                    &mut points,
                    &mut point_dst,
                    &mut surface_texture
                );
            }
        }
    }
}

/// Input render message
pub enum RenderInputMessage {
    /// Request for frame rendering
    NewFrame {
        /// Frame target buffer
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
    /// Response containing rendered frame
    RenderedFrame {
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
        stride: u32,
    }
}

/// Build surface texture with constant (whole-surface) light level
fn build_surface_texture_static(
    mut texture: FrameSliceMut<u64>,
    image: FrameSlice<u32>,
    image_uv_off: Vec2::<usize>,
    light: f32,
) {
    for y in 0..texture.height() {
        let src = image.get((y + image_uv_off.y()) % image.height()).unwrap();
        let dst = texture.get_mut(y).unwrap();

        let iter = dst.iter_mut().zip(src.iter().cycle().skip(image_uv_off.x()));
        for (dst, src) in iter {
            let [r, g, b, _] = src.to_le_bytes();

            *dst = u64_from_u16([
                unsafe { (r as f32 * light).to_int_unchecked::<u16>() },
                unsafe { (g as f32 * light).to_int_unchecked::<u16>() },
                unsafe { (b as f32 * light).to_int_unchecked::<u16>() },
                0
            ]);
        }
    }
}

/// Build surface texture for lightmap
fn build_surface_texture_impl<const IMAGE: bool, const LIGHTMAP: bool>(
    mut target: FrameSliceMut<u64>,

    image: FrameSlice<u32>,
    image_uv_off: Vec2::<usize>,

    lightmap: FrameSlice<u64>,
    lightmap_uv_off: Vec2::<usize>,
    lightmap_scale_log2: usize,
) {
    let t_w = target.width();
    let t_h = target.height();

    for y in 0..t_h {
        // Rows
        let target_r = target.get_mut(y).unwrap();
        let image_r = if IMAGE {
            image.get((y + image_uv_off.y()) % image.height()).unwrap()
        } else {
            &[]
        };
        let lightmap_r = if LIGHTMAP {
            lightmap.get(((y + lightmap_uv_off.y()) >> lightmap_scale_log2).min(lightmap.height() - 1)).unwrap()
        } else {
            &[]
        };

        for x in 0..t_w {
            let image_x = if IMAGE {
                image_r[(x + image_uv_off.x()) % image.width()]
            } else {
                0xFFFFFF
            };
            let lightmap_x = if LIGHTMAP {
                lightmap_r[((x + lightmap_uv_off.x()) >> lightmap_scale_log2).min(lightmap.width() - 1)]
            } else {
                0x00FF_00FF_00FF
            };

            target_r[x] = if LIGHTMAP && IMAGE {
                let [r, g, b, _] = image_x.to_le_bytes();
                let [lr, lg, lb, _] = u64_into_u16(lightmap_x);

                u64_from_u16([
                    ((r as u32 * lr as u32) >> 8) as u16,
                    ((g as u32 * lg as u32) >> 8) as u16,
                    ((b as u32 * lb as u32) >> 8) as u16,
                    0
                ])
            } else if LIGHTMAP {
                lightmap_x
            } else if IMAGE {
                let [r, g, b, _] = image_x.to_le_bytes();
                u64_from_u16([r as u16, g as u16, b as u16, 0])
            } else {
                // Unit white light and white color
                0x00FF_00FF_00FF
            };
        }
    }
}

/// Just cast high-res to low-res
fn build_surface_texture_plain(
    mut texture: FrameSliceMut<u64>,
    image: FrameSlice<u32>,
) {
    let w = texture.width();
    let h = texture.height();

    // Assert dimension equality
    assert!(w == image.width() && h == image.height());

    for y in 0..texture.height() {
        let dst = texture.get_mut(y).unwrap();
        let src = image.get(y).unwrap();

        let iter = dst.iter_mut().zip(src.iter());

        for (dst, src) in iter {
            let [r, g, b, _] = src.to_le_bytes();
            *dst = u64_from_u16([r as u16, g as u16, b as u16, 0]);
        }
    }
}

/// Convert HDR frame buffer to LDR
pub fn hdr_to_ldr(hdr: &[u64], ldr: &mut [u32], enable_tonemapping: bool) {

    /// Just clamp color, no tonemapping at all
    fn clamp(hdr: &u64, ldr: &mut u32) {
        let [r, g, b, _] = u64_into_u16(*hdr);
        *ldr = u32::from_le_bytes([
            u16::min(r, 0xFF) as u8,
            u16::min(g, 0xFF) as u8,
            u16::min(b, 0xFF) as u8,
            0
        ]);
    }


    /// Reinhard exposure parameter
    const REINHARD_EXPOSURE: f32 = 0.5;

    /// Simple Reinhard tonemapping
    #[allow(unused)]
    fn reinhard(src: &u64, dst: &mut u32) {
        let [r, g, b, _] = u64_into_u16(*src);
        let [r, g, b] = [r as f32, g as f32, b as f32];

        /// 256 inverse value
        const INV256: f32 = 256.0f32.recip();

        *dst = u32::from_ne_bytes([
            (r / (r * INV256 + REINHARD_EXPOSURE)) as u8,
            (g / (g * INV256 + REINHARD_EXPOSURE)) as u8,
            (b / (b * INV256 + REINHARD_EXPOSURE)) as u8,
            0,
        ]);
    }

    /// SSE2-based reinhard tonemapping
    #[cfg(target_feature = "sse2")]
    #[allow(unused)]
    fn reinhard_sse2(src_ptr: &u64, dst_ptr: &mut u32) {

        use std::arch::x86_64 as arch;

        unsafe {
            let src16 = std::mem::transmute::<arch::__m128d, arch::__m128i>(
                arch::_mm_load_sd(src_ptr as *const u64 as *const f64)
            );
            let src32 = arch::_mm_cvtepi16_epi32(src16);
            let src = arch::_mm_cvtepi32_ps(src32);

            // rgba / 256.0 + exposure
            let rgba_norm_aexp = arch::_mm_fmadd_ps(
                src,
                arch::_mm_set1_ps(1.0 / 256.0),
                arch::_mm_set1_ps(REINHARD_EXPOSURE)
            );

            // rgba / (rgba / 256.0 + exposure)
            let mapped = arch::_mm_div_ps(src, rgba_norm_aexp);

            let mapped32 = arch::_mm_cvtps_epi32(mapped);
            let mapped16 = arch::_mm_packus_epi32(mapped32, mapped32);
            let mapped8 = arch::_mm_packus_epi16(mapped16, mapped16);

            arch::_mm_store_ss(
                dst_ptr as *mut u32 as *mut f32,
                std::mem::transmute::<arch::__m128i, arch::__m128>(mapped8)
            );
        }
    }

    /// Tonemapping implementation
    fn impl_<const ENABLE: bool>(hdr: &[u64], ldr: &mut [u32]) {
        for (src, dst) in Iterator::zip(hdr.iter(), ldr.iter_mut()) {
            if ENABLE {
                #[cfg(target_feature = "sse2")]
                reinhard_sse2(src, dst);

                #[cfg(not(target_feature = "sse2"))]
                reinhard(src, dst);
            } else {
                clamp(src, dst);
            }
        }
    }

    if enable_tonemapping {
        impl_::<true>(hdr, ldr);
    } else {
        impl_::<false>(hdr, ldr);
    }
}

/// Present frame from slice to window surface
fn present_frame(
    mut frame: FrameSliceMut<'_, u32>,
    window_surface: &mut sdl2::video::WindowSurfaceRef,
) {
    let width = frame.width() as u32;
    let height = frame.height() as u32;
    let stride = frame.stride() as u32 * 4;
    let surface_bytes = frame.as_flat().unwrap().as_mut_bytes();

    let mut render_surface = match sdl2::surface::Surface::from_data(
        surface_bytes,
        width,
        height,
        stride,
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
    let (window_w, window_h) = window_surface.size();
    let src_rect = sdl2::rect::Rect::new(0, 0, width, height);
    let dst_rect = sdl2::rect::Rect::new(0, 0, window_w, window_h);

    if let Err(err) = render_surface.blit_scaled(src_rect, window_surface, dst_rect) {
        eprintln!("Surface blit failed: {}", err);
    }

    if let Err(err) = window_surface.update_window() {
        eprintln!("Window update failed: {}", err);
    }
}

/// Wrap function call with time calculation
fn with_time_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let value = f();
    let end = std::time::Instant::now();

    (value, end.duration_since(start).as_nanos() as f64 / 1_000_000f64)
}

fn init_render_thread(
    map: Arc<bsp::Map>,
    material_table: Arc<res::MaterialTable>
) -> (mpsc::Sender<RenderInputMessage>, mpsc::Receiver<RenderOutputMessage>) {
    let (render_in_sender, render_in_reciever) = mpsc::channel::<RenderInputMessage>();
    let (render_out_sender, render_out_reciever) = mpsc::channel::<RenderOutputMessage>();

    // Spawn render thread
    _ = std::thread::spawn(move || {
        // Create new local timer
        let mut timer = timer::Timer::default();

        let msg_reciever = render_in_reciever;
        let msg_sender = render_out_sender;

        // Map reference
        let map_arc = map.clone();
        let material_table_arc = material_table;

        let map = map_arc.as_ref();
        let material_table = material_table_arc.as_ref();

        let material_reference_table = material_table
            .build_reference_table(map);

        // Send empty frame to match sending order
        _ = msg_sender.send(RenderOutputMessage::RenderedFrame {
            frame_buffer: Vec::new(),
            width: 0,
            height: 0,
            stride: 0
        });

        'frame_render_loop: loop {
            let Ok(message) = msg_reciever.recv() else {
                break 'frame_render_loop;
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
                    frame_buffer.fill(0);

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

                        frame: FrameSliceMut::<u64>::new(
                            width as usize,
                            height as usize,
                            width as usize,
                            frame_buffer.as_mut_slice()
                        ),

                        map,
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
                    _ = msg_sender.send(RenderOutputMessage::RenderedFrame {
                        frame_buffer,
                        width,
                        height,
                        stride: width,
                    });
                }
            }
        }
    });

    (render_in_sender, render_out_reciever)
}

fn main() {
    // Enable/disable map caching
    let do_enable_map_caching = true;

    // Synchronize visible-set-building and projection cameras
    let mut shadow_camera: Option<camera::Camera> = None;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut rasterization_mode = RasterizationMode::Full;

    // Load map
    let map = {
        // yay, this code will not work on non-local builds)))
        // --
        let (map_name, map_src_format) = ("quake/e1m1", "map");
        // let (map_name, map_src_format) = ("d1_trainstation_01", "vmf");
        let data_path = ".local/";

        let wbsp_path = format!("{}wbsp/{}.wbsp", data_path, map_name);
        let map_src_path = format!("{}{}.{}", data_path, map_name, map_src_format);

        /// Load map source from local storage and compile it
        fn load_and_compile(path: &str, src_format: &str) -> bsp::Map {
            let source = std::fs::read_to_string(path).unwrap();

            let map = match src_format {
                "map" => map::q1_map::Map::parse(&source).unwrap().build_wmap(),
                "vmf" => map::source_vmf::Entry::parse_vmf(&source).unwrap().build_wmap().unwrap(),
                _ => panic!("'{}' map format is not implemented", src_format)
            };

            // Build BSP
            let mut bsp = bsp::compiler::compile(&map).unwrap();

            // Bake lightmaps
            bsp::lightmap_baker::bake(&mut bsp, &map);

            bsp
        }

        if do_enable_map_caching {
            match std::fs::File::open(&wbsp_path) {
                Ok(mut bsp_file) => {
                    // Load map from map cache
                    let mut bsp_file_data = Vec::new();
                    bsp_file.read_to_end(&mut bsp_file_data).unwrap();
                    bsp::wbsp::load(&bsp_file_data).unwrap()
                }
                Err(_) => {
                    // Compile map
                    let map = load_and_compile(&map_src_path, map_src_format);

                    if let Some(directory) = std::path::Path::new(&wbsp_path).parent() {
                        _ = std::fs::create_dir(directory);
                    }

                    // Save map to map cache
                    if let Ok(mut file) = std::fs::File::create(&wbsp_path) {
                        bsp::wbsp::save(&map, &mut file).unwrap()
                    }

                    map
                }
            }
        } else {
            load_and_compile(&map_src_path, map_src_format)
        }
    };

    let map = Arc::new(map);

    // Display some BSP statistics
    {
        pub struct BspStat {
            pub nodes: u64,
            pub leafs: u64,
            pub leaf_depth_sum: u64,
            pub depth_max: u64,
            pub total_disbalance: u64,
        }

        fn mk_bsp_stat(bsp: &bsp::Bsp, curr_depth: u64) -> BspStat {
            match bsp {
                bsp::Bsp::Partition { front, back, .. } => {
                    let fstat = mk_bsp_stat(front, curr_depth + 1);
                    let bstat = mk_bsp_stat(back, curr_depth + 1);

                    BspStat {
                        nodes: fstat.nodes + bstat.nodes + 1,
                        leafs: fstat.leafs + bstat.leafs,
                        leaf_depth_sum: fstat.leaf_depth_sum + bstat.leaf_depth_sum,
                        depth_max: u64::max(fstat.depth_max, bstat.depth_max),
                        total_disbalance: fstat.total_disbalance + bstat.total_disbalance
                            + u64::abs_diff(fstat.nodes, bstat.nodes)
                    }
                }
                bsp::Bsp::Volume(_) | bsp::Bsp::Void => {
                    BspStat {
                        nodes: 1,
                        leafs: 1,
                        leaf_depth_sum: curr_depth,
                        depth_max: curr_depth,
                        total_disbalance: 0,
                    }
                }
            }
        }

        let bsp = map.get_world_model().get_bsp();
        let stat = mk_bsp_stat(bsp, 0);

        println!("nodes           : {}", stat.nodes);
        println!("leafs           : {}", stat.leafs);
        println!("avg. leaf depth : {}", stat.leaf_depth_sum as f64 / stat.leafs as f64);
        println!("depth           : {}", stat.depth_max);
        println!("disbalance      : {}", stat.total_disbalance);
        println!("avg. disbalance : {}", stat.total_disbalance as f64 / (stat.nodes - stat.leafs) as f64);
    }

    let material_table = {
        let mut wad_file = std::fs::File::open(".local/quake/gfx/base.wad").unwrap();
        let mut wad_file_data = Vec::new();
        wad_file.read_to_end(&mut wad_file_data).unwrap();

        res::MaterialTable::load_wad2(&wad_file_data).unwrap()
    };
    let material_table = Arc::new(material_table);

    // Setup window
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("WEIRD-2", 1280, 720)
        .position_centered()
        .resizable()
        .build()
        .unwrap();

    // Setup systems
    let mut timer = timer::Timer::default();
    let mut input = input::Input::default();

    // camera.location = Vec3f::new(-174.0, 2114.6, -64.5); // -200, 2000, -50
    // camera.direction = Vec3f::new(-0.4, 0.9, 0.1);

    // heavy scene
    let mut camera = camera::Camera::new(Vec3f::new(1402.4, 1913.7, -86.3), Vec3f::new(-0.74, 0.63, -0.24));

    // camera.location = Vec3f::new(-543.3503, 1378.1802, 434.5833);
    // camera.direction = Vec3f::new(-0.004, 0.935, -0.354).normalized();

    // camera.direction = Vec3f::new(0.055, -0.946, 0.320); // (-0.048328593, -0.946524262, 0.318992347)

    // // sky
    // camera.location = Vec3f::new(-72.9, 698.3, -118.8);
    // camera.direction = Vec3f::new(0.37, 0.68, 0.63);

    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // Camera used for visible set building

    // Buffer that contains rendered pixels
    let mut hdr_frame_buffer = Vec::<u64>::new();

    // LDR framebuffer
    let mut ldr_frame_buffer = Vec::<u32>::new();

    // Render thread IO channels
    let (render_in, render_out) = init_render_thread(map.clone(), material_table.clone());

    'main_loop: loop {
        input.release_changed();

        while let Some(event) = event_pump.poll_event() {
            match event {
                sdl2::event::Event::Quit { .. } => {
                    break 'main_loop;
                }
                sdl2::event::Event::KeyUp { scancode: Some(code), .. } => {
                    input.on_state_changed(code, false);
                }
                sdl2::event::Event::KeyDown { scancode: Some(code), .. } => {
                    input.on_state_changed(code, true);
                }
                _ => {}
            }
        }

        timer.response();
        camera.response(&timer, &input);

        // Toggle shadow camera
        if input.is_key_clicked(input::Key::Num9) {
            shadow_camera = if shadow_camera.is_none() {
                Some(camera)
            } else {
                None
            };
        }

        if input.is_key_clicked(input::Key::Num0) {
            rasterization_mode = rasterization_mode.next();
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        /// Rendering resolution scale
        const FRAME_SCALE: usize = 2;

        let (frame_width, frame_height) = (
            window_width / FRAME_SCALE,
            window_height / FRAME_SCALE,
        );

        // Calculate aspect ratio
        let (aspect_x, aspect_y) = if window_width > window_height {
            (window_width as f32 / window_height as f32, 1.0)
        } else {
            (1.0, window_height as f32 / window_width as f32)
        };

        // Build projection matrix
        let projection_matrix = Mat4f::projection_frustum_inf_far(
            -0.5 * aspect_x, 0.5 * aspect_x,
            -0.5 * aspect_y, 0.5 * aspect_y,
            2.0 / 3.0
        );

        // Resize frame buffer to fit window's size
        hdr_frame_buffer.resize(frame_width * frame_height, 0);

        let send_res = render_in.send(RenderInputMessage::NewFrame {
            frame_buffer: hdr_frame_buffer,
            width: frame_width as u32,
            height: frame_height as u32,
            shadow_camera,
            camera,
            projection_matrix,
            rasterization_mode,
        });

        if let Err(e) = send_res {
            println!("Sending error: {}", e);
            break 'main_loop;
        }

        let Ok(render_result) = render_out.recv() else {
            eprintln!("Render thread dropped");
            break 'main_loop;
        };

        // Previous frame contents
        let prev_hdr_frame_buffer = match render_result {
            RenderOutputMessage::RenderedFrame {
                frame_buffer: rendered_hdr_buffer,
                width,
                height,
                stride
            } => {
                // Resize ldr buffer to match hdr buffer's size
                ldr_frame_buffer.resize(stride as usize * height as usize, 0);

                // Map from hdr to ldr
                let tm_time = with_time_ms(
                    || hdr_to_ldr(&rendered_hdr_buffer, &mut ldr_frame_buffer, true)
                ).1;

                // Render statistics
                let mut ldr_frame = FrameSliceMut::new(width as usize, height as usize, stride as usize, &mut ldr_frame_buffer);
                system_font::write(ldr_frame.reborrow())
                    .str(16, 8, &format!("FPS: {} ({}ms)", timer.get_fps(), 1000.0 / timer.get_fps()))
                    .str(16, 16, &format!("SC={}, RM={}",
                        shadow_camera.is_some() as u32,
                        rasterization_mode as u32
                    ))
                    .str(16, 24, &format!("TM: {}ms", tm_time))
                    .str(16, 32, &format!("RES: {}x{}", width, height))
                    ;

                // Present rendered frame
                match window.surface(&event_pump) {
                    Ok(mut window_surface) => present_frame(ldr_frame.reborrow(), &mut window_surface),
                    Err(err) => eprintln!("Cannot get window surface: {}", err),
                };

                /* Set previous buffer memory */
                Some(rendered_hdr_buffer)
            }
        };

        hdr_frame_buffer = prev_hdr_frame_buffer.unwrap_or(Vec::new());
    }
}
