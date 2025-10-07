use crate::{bsp::{Id, Lightmap}, math::Vec3f, u64_from_u16};

// Bake the most primitive lightmap ever
pub fn bake(map: &mut crate::bsp::Map) {
    for volume in map.volume_set.iter_mut() {
        for surface in volume.surfaces.iter_mut() {
            let normal = map.polygon_set[surface.polygon_id.into_index()].plane.normal;
            let light_diffuse = Vec3f::new(0.30, 0.47, 0.80)
                .normalized()
                .dot(normal)
                .abs()
                .min(0.99);

            // Add ambiance)
            let light = light_diffuse * 0.9 + 0.09;
            let color = u64_from_u16([(light * 65536.0) as u16; 4]);

            let width = (surface.u_max - surface.u_min) as usize / 4 + 2;
            let height = (surface.v_max - surface.v_min) as usize / 4 + 2;
            let data = vec![color; width * height].into_boxed_slice();

            surface.lightmap = Some(Lightmap { width, height, data });
        }
    }
}
