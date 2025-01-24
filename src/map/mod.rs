use std::num::NonZeroU32;

use crate::{geom, math::Vec3f};

/// Declare actual map builder module
pub mod builder;

/// Id generic implementation
macro_rules! impl_id {
    ($name: ident) => {
        /// Unique identifier
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
        pub struct $name(NonZeroU32);

        impl $name {
            /// Build id from index
            pub fn from_index(index: usize) -> Self {
                $name(NonZeroU32::try_from(index as u32 + 1).unwrap())
            }

            /// Get index by id
            pub fn into_index(self) -> usize {
                self.0.get() as usize - 1
            }
        }
    };
}

impl_id!(VolumeId);
impl_id!(PolygonId);
impl_id!(MaterialId);

/// Visible volume part
pub struct Surface {
    /// Polygon material identifier
    pub material_id: MaterialId,

    /// Polygon itself identifier
    pub polygon_id: PolygonId,
}

/// Portal (volume-volume connection)
pub struct Portal {
    /// Portal polygon identifier
    pub polygon_id: PolygonId,
    
    /// Is portal polygon facing 'into' volume it belongs to.
    /// This flag is used to share same portal polygons between different volumes
    pub is_facing_front: bool,

    /// Destination volume's identifier
    pub dst_volume_id: VolumeId,
}

/// Convex collection of polygons and portals
pub struct Volume {
    /// Set of visible volume elements
    surfaces: Vec<Surface>,

    /// Set of connections to another volumes
    portals: Vec<Portal>,
}

impl Volume {
    /// Get physical polygon set
    pub fn get_surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    /// Get portal set
    pub fn get_portals(&self) -> &[Portal] {
        &self.portals
    }
}

#[repr(C, packed)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Rgb8 {
    /// Red color component
    pub r: u8,

    /// Gree color component
    pub g: u8,

    /// Blue color component
    pub b: u8,
}

impl From<u32> for Rgb8 {
    fn from(value: u32) -> Self {
        let [r, g, b, _] = value.to_le_bytes();

        Self { r, g, b }
    }
}

impl Into<u32> for Rgb8 {
    fn into(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
    }
}

/// Material descriptor
pub struct Material {
    /// Only single-color materials are supported) (At least now)
    pub color: Rgb8,
}

/// Binary Space Partition, used during 
pub enum Bsp {
    /// Space partition
    Partition {
        /// Plane that splits front/back volume sets. Front volume set is located in front of plane.
        splitter_plane: geom::Plane,

        /// Pointer to front polygon part
        front: Box<Bsp>,

        /// Pointer to back polygon part
        back: Box<Bsp>,
    },

    /// Final volume
    Volume(VolumeId),

    /// Nothing there
    Void,
}

impl Bsp {
    /// Traverse map BSP by point
    pub fn traverse(&self, point: Vec3f) -> Option<VolumeId> {
        match self {
            Bsp::Partition { splitter_plane, front, back } => {
                let side_ref = match splitter_plane.get_point_relation(point) {
                    geom::PointRelation::Front | geom::PointRelation::OnPlane => front,
                    geom::PointRelation::Back => back,
                };

                side_ref.traverse(point)
            }
            Bsp::Volume(id) => Some(*id),
            Bsp::Void => None,
        }
    }
}

/// Map
pub struct Map {
    /// Map BSP. Actually, empty map *may not* contain any volumes, so BSP is an option.
    bsp: Box<Bsp>,

    /// Set of map polygons
    polygon_set: Vec<geom::Polygon>,

    /// Set of map materials
    material_set: Vec<Material>,

    /// Set of map volumes
    volume_set: Vec<Volume>,
}

impl Map {
    /// Build map from BSP and sets.
    pub fn new(bsp: Box<Bsp>, polygon_set: Vec<geom::Polygon>, material_set: Vec<Material>, volume_set: Vec<Volume>) -> Self {
        Self {
            bsp,
            polygon_set,
            material_set,
            volume_set
        }
    }

    /// Get volume by id
    pub fn get_volume(&self, id: VolumeId) -> Option<&Volume> {
        self.volume_set.get(id.into_index())
    }

    /// Get iterator on ids of all volumes
    pub fn all_volume_ids(&self) -> impl Iterator<Item = VolumeId> {
        (0..self.volume_set.len())
            .map(VolumeId::from_index)
    }

    /// Get material by id
    pub fn get_material(&self, id: MaterialId) -> Option<&Material> {
        self.material_set.get(id.into_index())
    }

    /// Get polygon by id
    pub fn get_polygon(&self, id: PolygonId) -> Option<&geom::Polygon> {
        self.polygon_set.get(id.into_index())
    }

    /// Get location BSP
    pub fn get_bsp(&self) -> &Bsp {
        &self.bsp
    }
}

// mod.rs
