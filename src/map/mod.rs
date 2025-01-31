///! Weird Map implementation module

use std::collections::HashMap;
use crate::{geom, math::Vec3f};

/// Quake 1 map
pub mod q1;

/// Map material
pub struct Material {
    /// Straight color
    pub color: u32,
}

/// Single brush face
pub struct BrushFace {
    /// Brush face plane
    pub plane: geom::Plane,

    /// U texture axis
    pub basis_u: Vec3f,

    /// V texture axis
    pub basis_v: Vec3f,

    /// U offset (in texels)
    pub mtl_offset_u: f32,

    /// V offset (in texels)
    pub mtl_offset_v: f32,

    /// U scale
    pub mtl_scale_u: f32,

    /// V scale
    pub mtl_scale_v: f32,

    /// Index of material in map material table
    pub mtl_index: u32,
}

/// Map brush
pub struct Brush {
    /// Brush face set
    pub faces: Vec<BrushFace>,
}

/// Map entity
pub struct Entity {
    /// Entity properties
    pub properties: HashMap<String, String>,

    /// Entity brushes
    pub brushes: Vec<Brush>,
}

/// Map main structure
pub struct Map {
    /// Entry set
    pub entities: Vec<Entity>,
}

// mod.rs
