//! Map run-time format implementation module. Uses BSP (Binary Space Partition) data structure as it's core.

use std::{hash::Hash, marker::PhantomData, num::NonZeroU32};

use crate::{frame_slice::FrameSlice, geom, math::{Vec2, Vec3f}};

pub mod compiler;
pub mod wbsp;
pub mod lightmap_baker;

/// Identifier type
pub struct Id<T>(NonZeroU32, PhantomData<fn(T) -> T>);

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, othr: &Self) -> bool {
        self.0 == othr.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.0.get());
    }
}

impl<T> std::fmt::Display for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Id<{}>", std::any::type_name::<T>())
    }
}

impl<T> Id<T> {
    /// Create Id for type from index
    pub const fn from_index(i: usize) -> Self {
        Self(NonZeroU32::new(!(i as u32)).unwrap(), PhantomData)
    }

    /// Convert type-generic Id into index
    pub const fn into_index(self) -> usize {
        !self.0.get() as usize
    }
}

impl<T> From<usize> for Id<T> {
    fn from(v: usize) -> Self {
        Self::from_index(v)
    }
}

impl<T> From<Id<T>> for usize {
    fn from(v: Id<T>) -> usize {
        v.into_index()
    }
}

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

/// Lightmap contents
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
    /// Lightmap bits as slice
    pub fn as_slice<'t>(&'t self) -> FrameSlice<'t, u64> {
        FrameSlice::new(self.width, self.height, self.width, &self.data)
    }
}

/// Volume face homogeneous rendered subset
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

/// Binary Space Partition, core map structure.
pub enum Bsp<S> {
    /// Space partition
    Partition {
        /// Plane splitting front and back subspaces
        splitter_plane: geom::Plane,

        /// Front subspace bsp
        front: Box<Self>,

        /// Back subspace bsp
        back: Box<Self>,
    },

    /// Convex space subregion, tree leaf
    Space(S),
}

impl<S: Default> Default for Bsp<S> {
    fn default() -> Self {
        Self::Space(S::default())
    }
}

/// Anamorphism (kind of BSP construction command) result
pub enum AnaResult<T, S> {
    /// Continue construction with given plane as splitter
    Partition(geom::Plane, T, T),

    /// Finish subtree construction (emit leaf)
    Space(S),
}

impl<S> Bsp<S> {
    /// Construct tree with function
    pub fn ana<T>(init: T, f: impl FnMut(T) -> AnaResult<T, S>) -> Bsp<S> {
        struct Ana<F>(F);
        impl<F> Ana<F> {
            fn build<T, S>(&mut self, state: T) -> Bsp<S>
            where
                F: FnMut(T) -> AnaResult<T, S>
            {
                match (self.0)(state) {
                    AnaResult::Partition(p, f, b) => Bsp::Partition {
                        splitter_plane: p,
                        front: Box::new(self.build(f)),
                        back: Box::new(self.build(b)),
                    },
                    AnaResult::Space(s) => Bsp::Space(s),
                }
            }
        }

        Ana(f).build(init)
    }

    /// Collapse tree in single value (without actual modification)
    pub fn cata_ref<T>(&self, leaf: impl FnMut(&S) -> T, branch: impl FnMut(T, T) -> T) -> T {
        struct Tr<Lf, Bf>(Lf, Bf);
        impl<Lf, Bf> Tr<Lf, Bf> {
            fn with<L, T>(&mut self, node: &Bsp<L>) -> T
            where
                Lf: FnMut(&L) -> T,
                Bf: FnMut(T, T) -> T
            {
                match node {
                    Bsp::Space(l) => (self.0)(l),
                    Bsp::Partition { front, back, .. } => {
                        let f = self.with(front);
                        let b = self.with(back);
                        (self.1)(f, b)
                    }
                }
            }
        }

        Tr(leaf, branch).with(self)
    }

    /// `cata_ref` function alias
    pub fn fold_ref<T>(&self, leaf: impl FnMut(&S) -> T, branch: impl FnMut(T, T) -> T) -> T {
        self.cata_ref(leaf, branch)
    }

    /// Calculate BSP tree depth
    pub fn depth(&self) -> usize {
        self.fold_ref(|_| 1, |l, r| usize::max(l, r) + 1)
    }

    /// Calculate number of BSP elements
    pub fn size(&self) -> usize {
        self.fold_ref(|_| 1, |l, r| l + r + 1)
    }

    /// Traverse with some function
    pub fn traverse<'t, T: TraverseFn>(&'t self, tf: T) -> BspIter<'t, S, T> {
        BspIter { nodes: vec![self], tf }
    }

    /// Map BSP to another type
    pub fn map<T>(self, f: impl FnMut(S) -> T) -> Bsp<T> {
        struct Mp<F>(F);
        impl<F> Mp<F> {
            fn map<S, T>(&mut self, bsp: Bsp<S>) -> Bsp<T>
                where F: FnMut(S) -> T
            {
                match bsp {
                    Bsp::Partition { splitter_plane, front, back } => Bsp::<T>::Partition {
                        splitter_plane,
                        front: Box::new(self.map(*front)),
                        back: Box::new(self.map(*back)),
                    },
                    Bsp::Space(l) => Bsp::<T>::Space((self.0)(l)),
                }
            }
        }

        Mp(f).map(self)
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

    /// Trace line inside space to some space border
    pub fn trace_space_border<'t>(&'t self, line: geom::Line) -> Option<TraceStep<'t, S>> {
        let mut node = self;
        let mut best_noff: Option<(Vec3f, f32)> = None;

        loop {
            match node {
                Bsp::Partition { splitter_plane, front, back } => {
                    let off = splitter_plane.intersect_line_coef(line);

                    if off.is_sign_positive() {
                        let (normal, offset) = best_noff.get_or_insert_default();
                        if off < *offset {
                            *offset = off;
                            *normal = splitter_plane.normal;
                        }
                    }

                    node = match splitter_plane.get_point_relation(line.origin) {
                        geom::PointRelation::Front | geom::PointRelation::OnPlane => front,
                        geom::PointRelation::Back => back,
                    };
                }
                Bsp::Space(space) => return best_noff.map(|(normal, offset)| TraceStep {
                    normal,
                    offset,
                    space
                })
            }
        }
    }

    /// Trace BSP along line
    pub fn trace<'t>(&'t self, line: geom::Line) -> BspTraceIter<'t, S> {
        BspTraceIter {
            line,
            offset: 0.0,
            bsp: self,
        }
    }
}

/// BSP tracing iterator
pub struct BspTraceIter<'t, S> {
    /// Traced line (remains unchanged)
    line: geom::Line,

    /// Current line offset
    offset: f32,

    /// Traced BSP root
    bsp: &'t Bsp<S>,
}

/// Trace step descriptor
#[derive(Copy, Clone)]
pub struct TraceStep<'t, S> {
    /// Hit normal
    pub normal: Vec3f,

    /// Hit offset
    pub offset: f32,

    /// Hit space reference
    pub space: &'t S,
}

impl<'t, S> Iterator for BspTraceIter<'t, S> {
    type Item = TraceStep<'t, S>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut hit = self.bsp.trace_space_border(self.line.offset(self.offset))?;
        hit.offset += self.offset;
        self.offset = hit.offset;
        Some(hit)
    }
}

/// Traverse function trait. Used for tree iteration and descend
pub trait TraverseFn {
    /// Qualify partition plane traverse order. True result means front-first.
    fn qualify(&mut self, plane: geom::Plane) -> bool;
}

impl<T: FnMut(geom::Plane) -> bool> TraverseFn for T {
    fn qualify(&mut self, plane: geom::Plane) -> bool {
        (self)(plane)
    }
}

/// Front-to-back `Bsp` traverse
pub struct FrontToBack;

impl TraverseFn for FrontToBack {
    fn qualify(&mut self, _plane: geom::Plane) -> bool {
        true
    }
}

/// Back-to-front `Bsp` traverse
pub struct BackToFront;

impl TraverseFn for BackToFront {
    fn qualify(&mut self, _plane: geom::Plane) -> bool {
        false
    }
}

/// Point-relative `Bsp` traverse (e.g. front-to-back if point is in front of plane, back-to-front otherwise)
pub struct AroundPoint(pub Vec3f);

impl TraverseFn for AroundPoint {
    fn qualify(&mut self, plane: geom::Plane) -> bool {
        match plane.get_point_relation(self.0) {
            geom::PointRelation::Front | geom::PointRelation::OnPlane => true,
            geom::PointRelation::Back => false,
        }
    }
}

/// `Bsp` iterator
pub struct BspIter<'t, S, Tf: TraverseFn> {
    /// BSP node visit stack
    nodes: Vec<&'t Bsp<S>>,

    /// Traverse function
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

/// Rendering BSP type alias
pub type RenderBsp = Bsp<Option<VolumeId>>;

/// Collision BSP type alias
pub type PhysicsBsp = Bsp<Medium>;

/// BSP medium type
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub enum Medium {
    /// Just air, space where entity can move
    #[default]
    Air,

    /// Solid area, does not permits entities to be inside
    Solid,
}

/// Traced object size
pub enum ObjectSize {
    /// Point object
    Point,

    /// Medium-sized object
    Medium,

    /// Large object
    Large,
}

/// Set of BSPs used for collisions of different object types
#[derive(Default)]
pub struct CollisionBsp {
    /// Map-size hull, 0x0x0
    pub point: Box<PhysicsBsp>,

    /// Medium-size hull, 32x32x56
    pub medium: Box<PhysicsBsp>,

    /// Large hull, 64x64x88
    pub large: Box<PhysicsBsp>,
}

/// `Bsp`-based model
pub struct BspModel {
    /// Rendered BSP
    bsp: Box<RenderBsp>,

    /// Model bounding volume
    bound_box: geom::BoundBox,
}

impl BspModel {
    /// Get render BSP
    pub fn get_bsp(&self) -> &Bsp<Option<VolumeId>> {
        self.bsp.as_ref()
    }

    /// Get bounding volume
    pub fn get_bound_box(&self) -> &geom::BoundBox {
        &self.bound_box
    }
}

/// Dynamic `Bsp`-based model
pub struct DynamicModel {
    /// Model translation
    pub origin: Vec3f,

    /// Model rotation (along Y axis)
    pub rotation: f32,

    /// Corresponding BSP model Id
    pub model_id: BspModelId,
}

/// World map structure
pub struct Map {
    /// Polygon set (used for polygon share (portal polygons, for example))
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

pub type VolumeId = Id<Volume>;
pub type PolygonId = Id<geom::Polygon>;
pub type MaterialId = Id<String>;
pub type BspModelId = Id<BspModel>;
pub type DynamicModelId = Id<DynamicModel>;
pub type SurfaceId = Id<Surface>;

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

    /// Iterate by material names
    pub fn all_material_names(&self) -> impl Iterator<Item = (MaterialId, &str)> {
        self.material_name_set
            .iter()
            .enumerate()
            .map(|(index, name)| (MaterialId::from_index(index), name.as_ref()))
    }

    /// Get material name by it's id
    pub fn get_material_name(&self, id: MaterialId) -> Option<&str> {
        self.material_name_set.get(id.into_index()).map(|s| s.as_str())
    }

    /// Get dynamic model by id
    pub fn get_dynamic_model(&self, id: DynamicModelId) -> Option<&DynamicModel> {
        self.dynamic_models.get(id.into_index())
    }

    /// Get polygon by id
    pub fn get_polygon(&self, id: PolygonId) -> Option<&geom::Polygon> {
        self.polygon_set.get(id.into_index())
    }

    /// Get BSP model by id
    pub fn get_bsp_model(&self, id: BspModelId) -> Option<&BspModel> {
        self.bsp_models.get(id.into_index())
    }

    /// Get ID of the world BSP model
    pub fn get_world_model_id(&self) -> BspModelId {
        self.world_model_id
    }

    /// Get world `BspModel`
    pub fn get_world_model(&self) -> &BspModel {
        self.get_bsp_model(self.world_model_id).unwrap()
    }
}
