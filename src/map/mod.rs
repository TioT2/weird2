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
    pub mtl_name: String,
}

/// Map brush
pub struct Brush {
    /// Brush face set
    pub faces: Vec<BrushFace>,
}

/// Map entity
pub struct Entity {
    /// Entity brushes
    pub brushes: Vec<Brush>,

    /// Entity properties
    pub properties: HashMap<String, String>,
}

/// Map main structure
pub struct Map {
    /// Entry set
    pub entities: Vec<Entity>,
}

impl Map {
    /// Find entity by property
    pub fn find_entity(&self, key: &str, value: Option<&str>) -> Option<&Entity> {
        if let Some(value) = value {
            for entity in &self.entities {
                if let Some(actual_value) = entity.properties.get(key) {
                    if actual_value == value {
                        return Some(entity);
                    }
                }
            }
        } else {
            for entity in &self.entities {
                if entity.properties.get(key).is_some() {
                    return Some(entity);
                }
            }
        }

        return None;
    }

    /// Extract all 'origin' properties from map
    /// (this function is used in invisible volume removal pass)
    pub fn get_all_origins(&self) -> Vec<Vec3f> {
        self
            .entities
            .iter()

            // Map entities to their 'origin' properties
            .filter_map(|entity| entity.properties.get("origin"))

            // Parse origin property values into vectors
            .filter_map(|origin| {
                let flt_arr = origin
                    .split_whitespace()
                    .map(|str| str.parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    ?;

                Some(Vec3f::new(
                    *flt_arr.get(0)?,
                    *flt_arr.get(1)?,
                    *flt_arr.get(2)?,
                ))
            })
            .collect::<Vec<_>>()
    }
}

// mod.rs
