//! Rendering logic

use std::collections::{HashMap, HashSet};

use crate::{bsp, camera, frame_slice::{FrameSlice, FrameSliceMut}, geom, math::{Mat4f, Vec2, Vec2f, Vec3f, Vec4f}, rand, res, u64_from_u16, u64_into_u16};

/// Different rasterization modes
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum RasterizationMode {
    /// Full rendering
    Full = 0,

    /// Material-id-defined monochrome
    MonochromeMaterial = 1,

    /// Polygon-id-defined monochrome
    MonochromePolygon = 2,

    /// Overdraw (brighter => more overdraw)
    Overdraw = 3,

    /// Inverse depth value
    Depth = 4,

    /// Rasterize with UV checker
    UV = 5,

    /// Unlit textures
    Textures = 6,

    /// Lightmaps only
    Lightmaps = 7,
}

impl RasterizationMode {
    // Rasterization mode count
    const COUNT: u32 = 8;

    /// Build rasterization mode from u32
    const fn from_u32(n: u32) -> Option<RasterizationMode> {
        Some(match n {
            0 => Self::Full,
            1 => Self::MonochromeMaterial,
            2 => Self::MonochromePolygon,
            3 => Self::Overdraw,
            4 => Self::Depth,
            5 => Self::UV,
            6 => Self::Textures,
            7 => Self::Lightmaps,
            _ => return None,
        })
    }

    /// Get next rasterization mode
    pub const fn next(self) -> RasterizationMode {
        Self::from_u32((self as u32 + 1) % Self::COUNT).unwrap()
    }
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
    pub fn xzuv(self) -> Vec4f {
        Vec4f::new(
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

/// Clip polygon by octagon
/// # Return
/// True if there are points rest
fn clip_polygon_oct(
    vertices: &mut Vec<Vertex>,
    temp: &mut Vec<Vertex>,
    clip_oct: &geom::BoundOct
) -> bool {
    // Vertex norm functions
    fn norm_x     (vt: Vertex) -> f32 { vt.position.x() }
    fn norm_y     (vt: Vertex) -> f32 { vt.position.y() }
    fn norm_y_a_x (vt: Vertex) -> f32 { vt.position.y() + vt.position.x() }
    fn norm_y_s_x (vt: Vertex) -> f32 { vt.position.y() - vt.position.x() }

    // Comparison functions
    fn ge(l: f32, r: f32) -> bool { l >= r }
    fn le(l: f32, r: f32) -> bool { l <= r }

    // Utilize '&&' hands calculation rules to stop clipping if there's <= 3 points
    
           geom::clip_polygon(vertices, temp, clip_oct.min.x(), ge, norm_x)
        && geom::clip_polygon(vertices, temp, clip_oct.max.x(), le, norm_x)
        && geom::clip_polygon(vertices, temp, clip_oct.min.y(), ge, norm_y)
        && geom::clip_polygon(vertices, temp, clip_oct.max.y(), le, norm_y)
        && geom::clip_polygon(vertices, temp, clip_oct.min.z(), ge, norm_y_s_x)
        && geom::clip_polygon(vertices, temp, clip_oct.max.z(), le, norm_y_s_x)
        && geom::clip_polygon(vertices, temp, clip_oct.min.w(), ge, norm_y_a_x)
        && geom::clip_polygon(vertices, temp, clip_oct.max.w(), le, norm_y_a_x)
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
            point_dst.push(self.view_projection.transform_hom(*point));
        }
        std::mem::swap(points, point_dst);

        // Clip polygon by z=1
        geom::clip_polygon(points, point_dst, 1.0, |l, r| l > r, |p| p.z());

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
                position: self.view_projection.transform_aff(Vec3f::new(u, v, s_dist)),
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
                position: self.view_projection.transform_hom(*point),
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
pub struct RenderContext<'t, 'ref_table> {
    /// Projection info holder
    pub camera: RenderCamera,

    /// Camera used for calculation of clipping
    pub shadow_camera: Option<RenderCamera>,

    /// Offset of background from foreground
    pub sky_background_uv_offset: Vec2f,

    /// Foreground offset
    pub sky_uv_offset: Vec2f,

    /// Map reference
    pub map: &'t bsp::Map,

    /// Table of materials
    pub material_table: &'t res::MaterialReferenceTable<'ref_table>,

    /// Rendering destination
    pub frame: FrameSliceMut<'t, u64>,

    /// Rasterization mode
    pub rasterization_mode: RasterizationMode,
}

impl<'t, 'ref_table> RenderContext<'t, 'ref_table> {
    /// Call pixel_fn for all polygon pixels
    fn render_clipped_polygon_impl<PixelFn: FnMut(&mut u64, Vec4f)>(
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
            prev_xzuv: Vec4f,
            curr_xzuv: Vec4f,
            prev_y: f32,
            curr_y: f32,
            d_xzuv: Vec4f,
        }

        impl LineContext {
            /// Step to the next point
            fn next_point<const IND_NEXT: bool>(&mut self, points: &[Vertex]) {

                self.index += if IND_NEXT { 1 } else { points.len() - 1 };
                self.index %= points.len();

                self.prev_y = self.curr_y;
                self.prev_xzuv = self.curr_xzuv;

                let vt = &points[self.index];
                self.curr_y = vt.position.y();
                self.curr_xzuv = vt.xzuv();

                let dy = self.curr_y - self.prev_y;

                // Check if edge is flat
                self.d_xzuv = if dy <= DY_EPSILON {
                    Vec4f::zero()
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

            let left_xzuv = Vec4f::mul_add(
                left.d_xzuv,
                Vec4f::broadcast(y - left.prev_y),
                left.prev_xzuv,
            );

            let right_xzuv = Vec4f::mul_add(
                right.d_xzuv,
                Vec4f::broadcast(y - right.prev_y),
                right.prev_xzuv,
            );

            let left_x = left_xzuv.x();
            let right_x = right_xzuv.x();

            let dx = right_x - left_x;

            let d_xzuv = if dx <= DY_EPSILON {
                Vec4f::zero()
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
                pixel_fn(p, Vec4f::mul_add(
                    d_xzuv,
                    Vec4f::broadcast(x as f32 - pixel_off),
                    left_xzuv
                ));
            }
        }
    }

    /// Wrap pixel function with transparency
    fn pixelfn_wrap_transparent(mut f: impl FnMut(&mut u64, Vec4f)) -> impl FnMut(&mut u64, Vec4f) {
        move |pixel_ptr: &mut u64, xzuv: Vec4f| {
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

    /// Render polygon
    unsafe fn render_clipped_polygon(
        &mut self,
        is_transparent: bool,
        is_sky: bool,
        points: &[Vertex],
        color: u64,
        texture: FrameSlice<u64>,
        sky_texture: FrameSlice<u32>,
    ) {
        // Macro that wraps actual rendering function call
        macro_rules! run {
            ($func: expr) => {
                {
                    // Make life of macro expander a bit happier
                    let f = $func;

                    if is_transparent {
                        self.render_clipped_polygon_impl(points, Self::pixelfn_wrap_transparent(f));
                    } else {
                        self.render_clipped_polygon_impl(points, f);
                    }
                }
            };
        }

        match self.rasterization_mode {
            RasterizationMode::MonochromeMaterial | RasterizationMode::MonochromePolygon => run!(move |dst: &mut u64, _| {
                *dst = color;
            }),
            RasterizationMode::Overdraw => run!(|dst: &mut u64, _| {
                *dst = dst.wrapping_add(0x0010_0010_0010u64);
            }),
            RasterizationMode::Depth => run!(|dst: &mut u64, xzuv: Vec4f| {
                let color = (xzuv.y() * 25500.0) as u16;
                *dst = u64_from_u16([color; 4]);
            }),
            RasterizationMode::UV => run!(|dst: &mut u64, xzuv: Vec4f| {
                let inv_z = xzuv.y().recip();
                let u = xzuv.z() * inv_z;
                let v = xzuv.w() * inv_z;

                let xi = (u as i64 & 0xFF) as u8;
                let yi = (v as i64 & 0xFF) as u8;

                *dst = color.wrapping_mul((((xi >> 5) ^ (yi >> 5)) & 1) as u64);
            }),
            RasterizationMode::Full | RasterizationMode::Textures | RasterizationMode::Lightmaps => {
                if is_sky {
                    let width = sky_texture.width() as isize >> 1;
                    let height = sky_texture.height() as isize;
                    let uv_offset = self.sky_uv_offset;
                    let background_uv_offset = self.sky_background_uv_offset;

                    run!(move |dst: &mut u64, xzuv: Vec4f| {
                        let inv_z = xzuv.y().recip();
                        let u = xzuv.z() * inv_z + uv_offset.x();
                        let v = xzuv.w() * inv_z + uv_offset.y();

                        let fg_u = unsafe { u.to_int_unchecked::<isize>() }
                            .rem_euclid(width)
                            .cast_unsigned();

                        let fg_v = unsafe { v.to_int_unchecked::<isize>() }
                            .rem_euclid(height)
                            .cast_unsigned();

                        let fg_color = *unsafe { sky_texture.get2_unchecked(fg_v, fg_u) };

                        // Check foreground color and fetch backround if foreground is transparent
                        if fg_color != 0 {
                            let [r, g, b, _] = fg_color.to_le_bytes();
                            *dst = u64_from_u16([r as u16, g as u16, b as u16, 0]);
                            return;
                        }

                        let bg_u = unsafe { (u + background_uv_offset.x()).to_int_unchecked::<isize>() }
                            .rem_euclid(width)
                            .wrapping_add(width)
                            .cast_unsigned();

                        let bg_v = unsafe { (v + background_uv_offset.y()).to_int_unchecked::<isize>() }
                            .rem_euclid(height)
                            .cast_unsigned();

                        let [r, g, b, _] = unsafe { sky_texture.get2_unchecked(bg_v, bg_u) }.to_le_bytes();
                        *dst = u64_from_u16([r as u16, g as u16, b as u16, 0]);
                    })
                } else {
                    let max_u = texture.width() - 1;
                    let max_v = texture.height() - 1;

                    run!(move |dst: &mut u64, xzuv: Vec4f| {
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
        let polygon = &self.map[surface.polygon_id];

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
            // self.camera.project_polygon(polygon, surface.u, surface.v, vertices);
        }

        // Clip polygon by Z=1
        if !geom::clip_polygon(vertices, vertices_dst, 1.0, |l, r| l > r, |v| v.position.z()) {
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
            let [r, g, b, _] = if matches!(self.rasterization_mode, RasterizationMode::MonochromePolygon) {
                rand::xorshift32(surface.polygon_id.into_index() as u32)
            } else {
                self.material_table.get_color(surface.material_id).unwrap()
            }.to_le_bytes();

            u64_from_u16([
                (r as f32 * static_light) as u16,
                (g as f32 * static_light) as u16,
                (b as f32 * static_light) as u16,
                0
            ])
        };

        let needs_surface_texture = {
            use RasterizationMode::*;
            match self.rasterization_mode {
                Depth | Overdraw | UV | MonochromeMaterial | MonochromePolygon => false,
                Full | Lightmaps | Textures => true,
            }
        };

        let sky_texture = if surface.is_sky() {
            image
        } else {
            FrameSlice::empty()
        };

        let surface_texture = if needs_surface_texture && !surface.is_sky() {
            // Map tex coords into space-linear coordinates
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

            // Resize surface texture data to fit resolution
            surface_texture_data.resize(texture_res.x() * texture_res.y(), 0);
            let mut texture = FrameSliceMut::new(
                texture_res.x(),
                texture_res.y(),
                texture_res.x(),
                surface_texture_data.as_mut_slice()
            );

            let rm_is_lit = matches!(self.rasterization_mode, RasterizationMode::Full | RasterizationMode::Lightmaps);
            let lighting = if rm_is_lit && let Some(lightmap) = surface.lightmap.as_ref() {
                Some(SurfaceLightmap {
                    img: FrameSlice::new(lightmap.width, lightmap.height, lightmap.width, &lightmap.data),
                    uv_off: (uv_int_min.map(|i| i << mip_index) - lightmap.uv_min.map(|i| i)).map(|i| i.max(0).cast_unsigned() >> mip_index),
                    scale_log2: 3 - mip_index,
                })
            } else {
                None
            };

            let rm_is_textured = matches!(self.rasterization_mode, RasterizationMode::Full | RasterizationMode::Textures);
            let color = if rm_is_textured {
                Some(SurfaceColormap {
                    img: image,
                    uv_off: image_uv_off,
                })
            } else {
                None
            };

            build_surface_texture(
                texture.reborrow_mut(),
                lighting,
                color
            );

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
                surface_texture,
                sky_texture,
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
        bsp_root: &bsp::Bsp<Option<bsp::VolumeId>>,
        start_volume_id: bsp::VolumeId,
        start_clip_oct: &geom::BoundOct,
        camera: &RenderCamera,
    ) -> Vec<(bsp::VolumeId, geom::BoundOct)> {
        // Render set itself
        let mut inv_render_set = Vec::new();

        // Potentially Visible volume (with corresponding clip octagon) Set
        let mut pvs = HashMap::<bsp::VolumeId, geom::BoundOct>::new();

        // Initialize PVS
        pvs.insert(start_volume_id, *start_clip_oct);

        let mut polygon_points = Vec::with_capacity(32);
        let mut proj_polygon_points = Vec::with_capacity(32);

        'traverse: for volume_id in bsp_root.traverse_around_pt(camera.location).filter_map(|x| *x) {
            let Some(volume_clip_oct) = pvs.get(&volume_id) else {
                continue 'traverse;
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
        }

        inv_render_set
    }

    pub fn render(&mut self) {
        let world_bsp = self.map.get_world_model().get_bsp();
        let screen_clip_oct = geom::BoundOct::from_clip_rect(self.get_screen_clip_rect());

        let partial_render_set_opt = if let Some(shadow_camera) = self.shadow_camera.as_ref() {
            world_bsp
                .find(shadow_camera.location)
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

                    for volume_id in world_bsp.traverse_around_pt(self.camera.location).filter_map(|x| *x) {
                        if unordered_render_set.remove(&volume_id) {
                            render_set.push((volume_id, screen_clip_oct));
                        }
                    }

                    render_set
                })
        } else {
            world_bsp
                .find(self.camera.location)
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

                for volume_id in world_bsp.traverse_around_pt(self.camera.location).filter_map(|x| *x) {
                    render_set_ref.push((volume_id, screen_clip_oct));
                }

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

/// Input to render about world change
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

/// Surface texture building lightmap descriptor
#[derive(Default)]
pub struct SurfaceLightmap<'t> {
    /// Lightmap image
    pub img: FrameSlice<'t, u64>,

    /// Offset from texture UVs
    pub uv_off: Vec2::<usize>,

    /// Base-2 logarithm of relative lightmap-texture scale
    pub scale_log2: usize,
}

/// Surface texture building color map descrpitor
#[derive(Default)]
pub struct SurfaceColormap<'t> {
    /// Color map image
    pub img: FrameSlice<'t, u32>,

    /// Offset from texture UVs
    pub uv_off: Vec2::<usize>,
}

/// Build surface texture for lightmap
fn build_surface_texture_impl<const IMAGE: bool, const LIGHTMAP: bool>(
    mut target: FrameSliceMut<u64>,
    colormap: SurfaceColormap,
    lightmap: SurfaceLightmap,
) {
    let mut iy = colormap.uv_off.y();
    let mut ly = !0;
    let mut lightmap_r: &[u64] = &[];

    for (y, target_r) in target.iter_mut().enumerate() {
        let image_r = if IMAGE {
            if iy >= colormap.img.height() {
                iy -= colormap.img.height();
            }
            &colormap.img[iy]
        } else {
            &[]
        };

        if IMAGE { iy += 1; }

        if LIGHTMAP {
            let ly1 = (y + lightmap.uv_off.y()) >> lightmap.scale_log2;
            if ly1 != ly {
                ly = ly1;
                lightmap_r = &lightmap.img[ly.min(lightmap.img.height() - 1)];
            }
        }

        let mut ix = colormap.uv_off.x();
        let mut lx = !0;
        let mut lightmap_p = 0x00FF_00FF_00FF;

        for (x, dst) in target_r.iter_mut().enumerate() {
            let [r, g, b, _] = if IMAGE {
                if ix >= colormap.img.width() {
                    ix -= colormap.img.width();
                }
                image_r[ix]
            } else {
                0xFF_FF_FF
            }.to_le_bytes();

            if IMAGE { ix += 1; }

            if LIGHTMAP {
                let lx1 = (x + lightmap.uv_off.x()) >> lightmap.scale_log2;
                if lx1 != lx {
                    lx = lx1;
                    lightmap_p = lightmap_r[lx.min(lightmap.img.width() - 1)];
                }
            }

            let [lr, lg, lb, _] = u64_into_u16(lightmap_p);

            *dst = u64_from_u16(match (IMAGE, LIGHTMAP) {
                (true, true) => [
                    ((r as u32 * lr as u32) >> 8) as u16,
                    ((g as u32 * lg as u32) >> 8) as u16,
                    ((b as u32 * lb as u32) >> 8) as u16,
                    0
                ],
                (false, true) => [lr, lg, lb, 0],
                (true, false) => [r as u16, g as u16, b as u16, 0],
                (false, false) => [0xFF, 0xFF, 0xFF, 0],
            });
        }
    }
}

/// Build surface texture
fn build_surface_texture(
    target: FrameSliceMut<u64>,
    lightmap: Option<SurfaceLightmap>,
    colormap: Option<SurfaceColormap>,
) {
    let build_fn = match (colormap.is_some(), lightmap.is_some()) {
        (false, false) => build_surface_texture_impl::<false, false>,
        (false, true ) => build_surface_texture_impl::<false, true >,
        (true , false) => build_surface_texture_impl::<true , false>,
        (true , true ) => build_surface_texture_impl::<true , true >,
    };

    build_fn(target, colormap.unwrap_or_default(), lightmap.unwrap_or_default());
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
