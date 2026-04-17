//! BSP (Binary Space Partition) run-time type, compiler and lightmapper implementation module

use std::num::NonZeroU32;

use crate::{geom, map, math::Vec3f};

pub mod compiler;
pub mod wbsp;
pub mod lightmap_baker;

/// Id type
pub trait Id: Copy + Clone + Eq + PartialEq + std::hash::Hash + std::fmt::Debug + Ord + PartialOrd {
    /// Construct Id from index
    fn from_index(index: usize) -> Self;

    /// Build Id into index
    fn into_index(self) -> usize;
}

/// Generic id implementation
macro_rules! impl_id {
    ($name: ident) => {
        /// Unique identifier
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
        pub struct $name(NonZeroU32);

        impl Id for $name {
            /// Build id from index
            fn from_index(index: usize) -> Self {
                $name(NonZeroU32::try_from(index as u32 + 1).unwrap())
            }
    
            /// Get index by id
            fn into_index(self) -> usize {
                self.0.get() as usize - 1
            }
        }
    };
}

impl_id!(VolumeId);
impl_id!(PolygonId);
impl_id!(MaterialId);
impl_id!(BspModelId);
impl_id!(DynamicModelId);
impl_id!(SurfaceId);

crate::flags! {
    /// Surface property bits
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub struct SurfaceFlags: u8 {
        /// Transparency (e.g. should surface be not)
        const TRANSPARENT = 0b0000_0001;

        /// Sky (should render apply inf-far reprojection)
        const SKY         = 0b0000_0010;

        /// Should render apply time-dependend variation
        const LIQUID      = 0b0000_0100;
    }
}

/// Volume face convex visible part.
pub struct Surface {
    /// Polygon material identifier
    pub material_id: MaterialId,

    /// Surface polygon identifier
    pub polygon_id: PolygonId,

    /// Surface U axis
    pub u: geom::Plane,

    /// Surface V axis
    pub v: geom::Plane,

    /// Flags denoting surface properties
    pub flags: SurfaceFlags,

    /// Lightmap (if present)
    pub lightmap: Option<Lightmap>,
}

impl Surface {
    /// Check if surface is sky
    pub const fn is_sky(&self) -> bool {
        self.flags.check(SurfaceFlags::SKY)
    }

    /// Check if surface is transparent
    pub const fn is_transparent(&self) -> bool {
        self.flags.check(SurfaceFlags::TRANSPARENT)
    }

    /// Check if surface is liquid
    pub const fn is_liquid(&self) -> bool {
        self.flags.check(SurfaceFlags::LIQUID)
    }
}

/// Directional lightmap
pub struct Lightmap {
    /// Light color array
    pub data: Box<[u64]>,

    /// Width
    pub width: usize,

    /// Height
    pub height: usize,
}

/// Portal (volume-volume connection descriptor)
pub struct Portal {
    /// Portal polygon identifier
    pub polygon_id: PolygonId,

    /// Destination volume's identifier
    pub dst_volume_id: VolumeId,

    /// Is portal polygon facing 'into' volume it belongs to.
    /// This flag is used to share same portal polygons between different volumes
    pub is_facing_front: bool,
}

/// BSP leaf, convex polyhedron containing sets of drawable surfaces and infos about neighbours.
pub struct Volume {
    /// Set of visible volume elements
    pub surfaces: Vec<Surface>,

    /// Set of connections with another volumes
    pub portals: Vec<Portal>,

    /// Volume bounding box
    pub bound_box: geom::BoundBox,
}

/// Binary Space Partition structure
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

    /// BSP tree leaf
    Volume(VolumeId),

    /// Nothing there
    Void,
}

impl Bsp {
    /// Find BSP cell that contains the point
    pub fn find_volume(&self, point: Vec3f) -> Option<VolumeId> {
        let mut curr = self;

        loop {
            match curr {
                Bsp::Partition { splitter_plane, front, back } => {
                    curr = match splitter_plane.get_point_relation(point) {
                        geom::PointRelation::Front | geom::PointRelation::OnPlane => front,
                        geom::PointRelation::Back => back,
                    };
                }
                Bsp::Volume(volume_id) => return Some(*volume_id),
                Bsp::Void => return None,
            }
        }
    }

    /// Calculate BSP tree depth
    pub fn depth(&self) -> usize {
        match self {
            Bsp::Partition { splitter_plane: _, front, back }
                => usize::max(front.depth(), back.depth()) + 1,
            _
                => 1,
        }
    }

    /// Count of BSP elements
    pub fn size(&self) -> usize {
        match self {
            Bsp::Partition { splitter_plane: _, front, back } => front.size() + back.size() + 1,
            _ => 1,
        }
    }
}

/// Static model
pub struct BspModel {
    /// Model BSP
    bsp: Box<Bsp>,

    /// (Simple) Bounding volume, used during split process
    bound_box: geom::BoundBox,
}

impl BspModel {
    /// Get BSP 
    pub fn get_bsp(&self) -> &Bsp {
        &self.bsp
    }

    /// Get bounding volume
    pub fn get_bound_box(&self) -> &geom::BoundBox {
        &self.bound_box
    }
}

/// Dynamic BSP element
pub struct DynamicModel {
    /// Model translation
    pub origin: Vec3f,

    /// Model rotation (along Y axis)
    pub rotation: f32,

    /// Corresponding BSP model Id
    pub model_id: BspModelId,
}

/// Map
pub struct Map {
    /// Set of map polygons
    polygon_set: Vec<geom::Polygon>,

    /// Set of map materials
    material_name_set: Vec<String>,

    /// Set of map volumes
    volume_set: Vec<Volume>,

    /// Set of BSP models
    bsp_models: Vec<BspModel>,

    /// Set of dynamically-rendered objects
    dynamic_models: Vec<DynamicModel>,

    /// Id of the BSP model used as the world
    world_model_id: BspModelId,
}

impl Map {
    /// Get volume by id
    pub fn get_volume(&self, id: VolumeId) -> Option<&Volume> {
        self.volume_set.get(id.into_index())
    }

    /// Get iterator on ids of all volumes
    pub fn all_volume_ids(&self) -> impl Iterator<Item = VolumeId> + use<> {
        (0..self.volume_set.len()).map(VolumeId::from_index)
    }

    /// Iterate though dynamic model IDs
    pub fn all_dynamic_model_ids(&self) -> impl Iterator<Item = DynamicModelId> + use<> {
        (0..self.dynamic_models.len()).map(DynamicModelId::from_index)
    }

    /// Get material name by it's id
    pub fn get_material_name(&self, id: MaterialId) -> Option<&str> {
        self.material_name_set.get(id.into_index()).map(|s| s.as_str())
    }

    /// Iterate by material names
    pub fn all_material_names(&self) -> impl Iterator<Item = (MaterialId, &str)> {
        self.material_name_set
            .iter()
            .enumerate()
            .map(|(index, name)| (MaterialId::from_index(index), name.as_ref()))
    }

    /// Get dynamic model by id
    pub fn get_dynamic_model(&self, id: DynamicModelId) -> Option<&DynamicModel> {
        self.dynamic_models.get(id.into_index())
    }

    /// Get polygon by id
    pub fn get_polygon(&self, id: PolygonId) -> Option<&geom::Polygon> {
        self.polygon_set.get(id.into_index())
    }

    /// Get ID of the world BSP model
    pub fn get_world_model_id(&self) -> BspModelId {
        self.world_model_id
    }

    /// Get root BSP model
    pub fn get_world_model(&self) -> &BspModel {
        self.bsp_models.get(self.world_model_id.into_index()).unwrap()
    }

    /// Get BSP model by id
    pub fn get_bsp_model(&self, id: BspModelId) -> Option<&BspModel> {
        self.bsp_models.get(id.into_index())
    }

    /// Test if volume contains point or not
    pub fn volume_contains_point(&self, id: VolumeId, point: Vec3f) -> Option<bool> {
        let volume = self.get_volume(id)?;

        for portal in &volume.portals {
            let mut plane = self.polygon_set[portal.polygon_id.into_index()].plane;

            if !portal.is_facing_front {
                plane = plane.negate_direction();
            }

            if plane.get_point_relation(point) == geom::PointRelation::Back {
                return Some(false);
            }
        }

        for surface in &volume.surfaces {
            let plane = self.polygon_set[surface.polygon_id.into_index()].plane;

            if plane.get_point_relation(point) == geom::PointRelation::Back {
                return Some(false);
            }
        }

        Some(true)
    }
}

impl Map {
    /// Compile map to WBSP
    pub fn compile(map: &map::Map) -> Result<Self, compiler::Error> {
        compiler::compile(map)
    }

    /// Bake map lightmaps
    pub fn bake_lightmaps(&mut self) {
        lightmap_baker::bake(self);
    }
}
