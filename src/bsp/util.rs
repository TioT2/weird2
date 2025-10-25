///! BSP utility set

use crate::{geom::{self, GEOM_EPSILON}, math::{Vec2f, Vec3f}};

pub struct ShadowMapInfo {
    /// U coordinate
    pub u: geom::Plane,

    /// V coordinate
    pub v: geom::Plane,

    /// ShadowMap width
    pub width: u32,

    /// ShadowMap height
    pub height: u32,
}

/// Build some basis from normal vector
pub fn build_normal_basis(normal: Vec3f) -> (Vec3f, Vec3f) {
    let make = |v| {
        let mut axis = Vec3f::cross(normal, v);
        if axis.length2() <= GEOM_EPSILON {
            axis = Vec3f::cross(normal, Vec3f::new(0.0, 0.0, 1.0));
        }
        axis.normalized()
    };

    (make(Vec3f::new(1.0, 0.0, 0.0)), make(Vec3f::new(0.0, 1.0, 0.0)))
}

/// Calculate shadowmapping axis bounds
pub fn calculate_shadowmap_ranges(points: &[Vec3f], plane: geom::Plane) -> ShadowMapInfo {
    let (u_dir, v_dir) = build_normal_basis(plane.normal);
    let uv_bounds = points.iter().fold(
        geom::ClipRect {
            max: Vec2f::from_single(f32::MIN),
            min: Vec2f::from_single(f32::MAX),
        },
        |b, p| b.extend_to_contain(Vec2f::new(p.dot(u_dir), p.dot(v_dir)))
    );

    todo!()
}

/// Calculate rounded UV bonuds for point set
/// # Returns
/// (u_min, u_max, v_min, v_max) tuple
pub fn calculate_uv_ranges(points: &[Vec3f], u: geom::Plane, v: geom::Plane) -> (i32, i32, i32, i32) {
    let uv_bounds = points.iter().fold(
        geom::ClipRect {
            max: Vec2f::from_single(f32::MIN),
            min: Vec2f::from_single(f32::MAX),
        },
        |b, point| b.extend_to_contain(Vec2f::new(
            point.dot(u.normal) + u.distance,
            point.dot(v.normal) + v.distance,
        ))
    );

    (
        uv_bounds.min.x.floor() as i32,
        uv_bounds.max.x.ceil() as i32,
        uv_bounds.min.y.floor() as i32,
        uv_bounds.max.y.ceil() as i32,
    )
}

// util.rs
