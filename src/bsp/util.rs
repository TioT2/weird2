///! BSP utility set

use crate::{geom, math::{Vec2f, Vec3f}};

/// Calculate rounded UV bonuds for point set
/// # Returns
/// (u_min, u_max, v_min, v_max) tuple
pub fn calculate_uv_ranges(points: &[Vec3f], u: geom::Plane, v: geom::Plane) -> (i32, i32, i32, i32) {
    // Calculate UV bounds
    let mut uv_bounds = geom::ClipRect {
        max: Vec2f::from_single(f32::MIN),
        min: Vec2f::from_single(f32::MAX),
    };

    for point in points {
        uv_bounds = uv_bounds.extend_to_contain(Vec2f::new(
            point.dot(u.normal) + u.distance,
            point.dot(v.normal) + v.distance,
        ));
    }

    (
        uv_bounds.min.x.floor() as i32,
        uv_bounds.max.x.ceil() as i32,
        uv_bounds.min.y.floor() as i32,
        uv_bounds.max.y.ceil() as i32,
    )
}

// util.rs
