//! Map (BSP collection) format implementation

use std::collections::HashMap;
use crate::{geom, math::Vec3f};

/// Quake 1 map
pub mod q1_map;

/// Source map
pub mod source_vmf;

crate::flags! {
    /// Brush face flags
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub struct BrushFaceFlags: u8 {
        /// Transparency bit
        const TRANSPARENT = 0b0000_0001;

        /// Sky bit
        const SKY         = 0b0000_0010;

        /// Bursh texture should vary with time.
        const LIQUID      = 0b0000_0100;

        /// Brush is not used during map compilation
        const INVISIBLE   = 0b0000_1000;
    }
}

crate::flags! {
    /// Flags denoting properties of the certain brush.
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub struct BrushFlags: u8 {
        /// Brush should not be used during graphical BSP compilation
        const INVISIBLE = 0b0000_0001;

        /// Brush volume is filled with water
        const WATER     = 0b0000_0010;
    }
}

/// Single brush face
pub struct BrushFace {
    /// Brush face plane
    pub plane: geom::Plane,

    /// U texture axis
    pub u: geom::Plane,

    /// V texture axis
    pub v: geom::Plane,

    /// Name of material applied to brush face
    pub mtl_name: String,

    /// Face property flag bits
    pub flags: BrushFaceFlags,
}

/// Map brush
pub struct Brush {
    /// Brush face set
    pub faces: Vec<BrushFace>,

    /// Brush flags
    pub flags: BrushFlags,
}

/// Map entity
pub struct Entity {
    /// Entity brushes
    pub brushes: Vec<Brush>,

    /// Entity string property set
    pub properties: HashMap<String, String>,
}

impl Entity {
    /// Try to extract entity origin
    pub fn origin(&self) -> Option<Vec3f> {
        self.properties
            .get("origin")?
            .split_whitespace()
            .map(|str| str.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?
            .try_into()
            .ok()
            .map(Vec3f::from_array)
    }
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
                if let Some(actual_value) = entity.properties.get(key) && actual_value == value {
                    return Some(entity);
                }
            }
        } else {
            for entity in &self.entities {
                if entity.properties.contains_key(key) {
                    return Some(entity);
                }
            }
        }

        None
    }

    /// Extract all 'origin' properties from map
    /// (this function is used in invisible volume removal pass)
    pub fn get_all_origins(&self) -> Vec<Vec3f> {
        self.entities.iter().filter_map(Entity::origin).collect::<Vec<_>>()
    }
}
