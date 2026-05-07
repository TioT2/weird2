//! BSP lightmap baker implementation

use crate::{bsp::{self, Id, SurfaceLightmap}, geom, map, math::{Vec2f, Vec3f}, u64_from_u16};

/// Point light structure
struct PointLight {
    /// Light origin
    pub origin: Vec3f,

    /// Per-color intensity
    pub intensity: Vec3f,

    /// Attenuation distance
    pub att_distance: f32,
}

impl PointLight {
    /// Build point light from it's origin and intensity
    pub fn from_origin_intensity(origin: Vec3f, intensity: Vec3f) -> Self {
        Self {
            origin,
            intensity,
            att_distance: (intensity.fold1(f32::max) * 4096.0).sqrt(),
        }
    }

    /// Try to parse self from entity
    pub fn from_entity(ent: &map::Entity) -> Option<Self> {
        if !ent.properties.get("classname")?.starts_with("light") {
            return None;
        }
        let light = if let Some(light_s) = ent.properties.get("light") {
            light_s.parse::<f32>().ok()?
        } else {
            // Default light value (by spec)
            200.0
        };
        Some(Self::from_origin_intensity(ent.origin()?, light.into()))
    }
}

/// Bake volume BSP
fn bake_volume(
    bsp: &mut bsp::Map,
    lights: &[PointLight],
    light_idx: &[u32],
    volume_id: bsp::VolumeId,
) {
    let volume = bsp.volume_set.get_mut(volume_id.into_index()).unwrap();

    for surface in volume.surfaces.iter_mut() {
        // Lightmapping for transparent surface and liquids is not required (at least now)
        if surface.is_sky() || surface.is_transparent() || surface.is_liquid() {
            continue;
        }

        let polygon = bsp.polygon_set.get(surface.polygon_id.into_index()).unwrap();

        // Calculate polygon UV bounds
        let uvs = polygon.points.iter().map(|p| Vec2f::new(
            surface.u.get_signed_distance(*p),
            surface.v.get_signed_distance(*p),
        )).collect::<Vec<_>>();

        let uv_bounds = geom::BoundRect::for_points(uvs.iter().copied());
        let uv_int_min = uv_bounds.min.map(|x| ((x / 8.0).floor() * 8.0) as isize);
        let uv_int_max = uv_bounds.max.map(|x| ((x / 8.0).ceil() * 8.0) as isize);
        let lightmap_res = (uv_int_max - uv_int_min).map(|x| x.cast_unsigned() / 8 + 1);

        // Lightmap data
        let mut data = Vec::<u64>::with_capacity(lightmap_res.y() * lightmap_res.x());

        let uvd = surface.u.normal % surface.v.normal;

        // Build lightmap data
        for y in 0..lightmap_res.y() {
            for x in 0..lightmap_res.x() {
                // Light collector
                let mut light_sum = Vec3f::zero();

                // Average by subpixels
                for sy in 0..8 {
                    let ty = y * 8 + sy;
                    let vaxis = surface.v.point_at(uv_int_min.y() as f32 + ty as f32 + 0.5);

                    for sx in 0..8 {
                        let tx = x * 8 + sx;
                        let uaxis = surface.u.point_at(uv_int_min.x() as f32 + tx as f32 + 0.5);
                        let point = polygon.plane.project_along(uaxis + vaxis, uvd);

                        for i in light_idx {
                            let light = &lights[*i as usize];
                            let att = (point - light.origin).length2().recip();
                            light_sum += light.intensity * att.into();
                        }
                    }
                }

                // Divide for subpixel count
                light_sum /= 64.0.into();

                // 3 is just constant to made it look a bit better
                let [r, g, b] = light_sum.map(|x| (x * 256.0 * 3.0).clamp(0.0, 65535.0) as u16).into_array();

                data.push(u64_from_u16([r, g, b, 0]));
            }
        }

        // Assign surface lightmap
        surface.lightmap = Some(SurfaceLightmap {
            width: lightmap_res.x(),
            height: lightmap_res.y(),
            data: data.into_boxed_slice(),
            uv_min: uv_int_min,
            uv_max: uv_int_max,
        });
    }
}

/// Bake BSP node
fn bake_bsp(
    bsp: &mut bsp::Map,
    node: &bsp::Bsp,
    lights: &[PointLight],
    idx: &mut Vec<u32>,
) {
    match node {
        bsp::Bsp::Partition { splitter_plane, front, back } => {
            // Back index array
            let mut back_idx = Vec::new();

            idx.retain(|i| {
                let light = &lights[*i as usize];
                let dist = splitter_plane.get_signed_distance(light.origin);

                if dist > light.att_distance {
                    true
                } else if dist < -light.att_distance {
                    back_idx.push(*i);
                    false
                } else {
                    back_idx.push(*i);
                    true
                }
            });

            // Reduce transient memory usage (a bit)
            bake_bsp(bsp, back, lights, &mut back_idx);
            std::mem::drop(back_idx);

            bake_bsp(bsp, front, lights, idx);
        }
        bsp::Bsp::Volume(id) => bake_volume(bsp, lights, idx, *id),
        bsp::Bsp::Void => {
        }
    }
}

/// Bake lightmap for BSP reading map metadata
pub fn bake(bsp: &mut bsp::Map, map: &map::Map) {
    // Set of active point lights
    let point_lights = map
        .entities
        .iter()
        .filter_map(PointLight::from_entity)
        .collect::<Vec<_>>();

    let world = bsp.bsp_models.get_mut(bsp.world_model_id.into_index()).unwrap();

    let world_bsp = std::mem::replace(world.bsp.as_mut(), bsp::Bsp::Void);

    let mut idx = (0..point_lights.len() as u32).collect::<Vec<_>>();

    bake_bsp(
        bsp,
        &world_bsp,
        &point_lights,
        &mut idx
    );

    // Return world_bsp to place
    *bsp.bsp_models.get_mut(bsp.world_model_id.into_index()).unwrap().bsp = world_bsp;
}
