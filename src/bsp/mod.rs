//! BSP (Binary Space Partition) run-time type, compiler and lightmapper implementation module

use std::num::NonZeroU32;

use crate::{frame_slice::FrameSlice, geom, math::{Vec2, Vec3f}};

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
    ($Id: ident) => {
        /// Some unique identifier
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
        pub struct $Id(NonZeroU32);

        impl From<usize> for $Id {
            fn from(v: usize) -> $Id {
                Self::from_index(v)
            }
        }

        impl From<$Id> for usize {
            fn from(v: $Id) -> usize {
                v.into_index()
            }
        }

        impl Id for $Id {
            /// Build id from index
            fn from_index(index: usize) -> Self {
                $Id(NonZeroU32::try_from(!(index as u32)).unwrap())
            }
    
            /// Get index by id
            fn into_index(self) -> usize {
                (!self.0.get()) as usize
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

/// Surface lightmap structure
pub struct SurfaceLightmap {
    /// Lightmap data bytes
    pub data: Box<[u64]>,

    /// Image width (in u64 pixels)
    pub width: usize,

    /// Image height (in u64 pixels)
    pub height: usize,

    /// Polygon lightmapped UV minimum
    pub uv_min: Vec2::<isize>,

    /// Polygon lightmapped UV maximum
    pub uv_max: Vec2::<isize>,
}

impl SurfaceLightmap {
    /// Get lightmap image slice
    pub fn as_slice<'t>(&'t self) -> FrameSlice<'t, u64> {
        FrameSlice::new(self.width, self.height, self.width, &self.data)
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

    /// Surface lightmap texture
    pub lightmap: Option<SurfaceLightmap>,
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

/// Binary Space Partition enumeration
pub enum Bsp<S> {
    /// Space partition
    Partition {
        /// Plane that splits front/back volume sets. Front volume set is located in front of plane.
        splitter_plane: geom::Plane,

        /// Pointer to front polygon part
        front: Box<Self>,

        /// Pointer to back polygon part
        back: Box<Self>,
    },

    /// Space itself
    Space(S),
}

impl<S> Bsp<S> {

    /// Use fold but by reference
    pub fn fold_ref<T>(&self, leaf: impl FnMut(&S, usize) -> T, branch: impl FnMut(T, T) -> T) -> T {
        /// Traverse helper structure using stack instead of heap
        struct Tr<Lf, Bf> {
            leaf: Lf,
            branch: Bf
        }

        impl<Lf, Bf> Tr<Lf, Bf> {
            fn with<L, T>(&mut self, node: &Bsp<L>, depth: usize) -> T
            where
                Lf: FnMut(&L, usize) -> T,
                Bf: FnMut(T, T) -> T
            {
                match node {
                    Bsp::Partition { front, back, .. } => {
                        let f = self.with(front, depth + 1);
                        let b = self.with(back, depth + 1);
                        (self.branch)(f, b)
                    }
                    Bsp::Space(l) => (self.leaf)(l, depth),
                }
            }
        }

        Tr { leaf, branch }.with(self, 0)
    }

    /// Calculate BSP tree depth
    pub fn depth(&self) -> usize {
        self.fold_ref(|_, d| d, usize::max)
    }

    /// Count of BSP elements
    pub fn size(&self) -> usize {
        self.fold_ref(|_, _| 1, |l, r| l + r + 1)
    }

    /// Traverse with some function
    pub fn traverse<'t, T: TraverseFn>(&'t self, tf: T) -> BspIter<'t, S, T> {
        BspIter { nodes: vec![self], tf }
    }

    /// Descend using traverse function
    pub fn descend<T: TraverseFn>(&self, mut tf: T) -> &S {
        let mut curr = self;
        loop {
            curr = match curr {
                Self::Partition { splitter_plane, front, back } => if tf.qualify(*splitter_plane) {
                    front
                } else {
                    back
                },
                Self::Space(s) => return s,
            };
        }
    }

    /// Traverse around point
    pub fn traverse_around_pt<'t>(&'t self, pt: Vec3f) -> BspIter<'t, S, AroundPoint> {
        self.traverse(AroundPoint(pt))
    }

    /// Find BSP cell that contains the point
    pub fn find(&self, point: Vec3f) -> &S {
        self.descend(AroundPoint(point))
    }
}

/// BSP traverse trait
pub trait TraverseFn {
    fn qualify(&mut self, plane: geom::Plane) -> bool;
}

impl<T: FnMut(geom::Plane) -> bool> TraverseFn for T {
    fn qualify(&mut self, plane: geom::Plane) -> bool {
        (self)(plane)
    }
}

pub struct FrontToBack;
impl TraverseFn for FrontToBack {
    fn qualify(&mut self, _plane: geom::Plane) -> bool {
        true
    }
}

pub struct BackToFront;
impl TraverseFn for BackToFront {
    fn qualify(&mut self, _plane: geom::Plane) -> bool {
        false
    }
}

pub struct AroundPoint(Vec3f);
impl TraverseFn for AroundPoint {
    fn qualify(&mut self, plane: geom::Plane) -> bool {
        match plane.get_point_relation(self.0) {
            geom::PointRelation::Front | geom::PointRelation::OnPlane => true,
            geom::PointRelation::Back => false,
        }
    }
}

/// BSP traverse iterator
pub struct BspIter<'t, S, Tf> {
    nodes: Vec<&'t Bsp<S>>,
    tf: Tf,
}

impl<'t, S, Tf: TraverseFn> Iterator for BspIter<'t, S, Tf> {
    type Item = &'t S;

    fn next(&mut self) -> Option<&'t S> {
        let mut node = self.nodes.pop()?;

        'descend: loop {
            match node {
                Bsp::Partition { splitter_plane, front, back } => {
                    let stn;
                    (node, stn) = if self.tf.qualify(*splitter_plane) {
                        (front, back)
                    } else {
                        (back, front)
                    };
                    self.nodes.push(stn);
                }
                Bsp::Space(l) => break 'descend Some(l),
            };
        }
    }
}

/// Bsp of volumes
pub type VolumeBsp = Bsp<Option<VolumeId>>;

/// Static model
pub struct BspModel {
    /// Model BSP
    bsp: Box<Bsp<Option<VolumeId>>>,

    /// (Simple) Bounding volume, used during split process
    bound_box: geom::BoundBox,
}

impl BspModel {
    /// Get BSP 
    pub fn get_bsp(&self) -> &Bsp<Option<VolumeId>> {
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

macro_rules! impl_map_index {
    ($Id: ty, $Val: ty, $get: ident) => {
        impl std::ops::Index<$Id> for Map {
            type Output = $Val;

            fn index(&self, id: $Id) -> &$Val {
                self.$get(id).unwrap()
            }
        }
    }
}

impl_map_index!(PolygonId, geom::Polygon, get_polygon);
impl_map_index!(VolumeId, Volume, get_volume);
impl_map_index!(BspModelId, BspModel, get_bsp_model);
impl_map_index!(DynamicModelId, DynamicModel, get_dynamic_model);

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
