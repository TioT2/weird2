/*
Location building process
1. Build BSP polyhedrons from plane sets
2. Calculate hull volume
3. 
 */

use crate::{geom, math::Vec3f};

/// Volume identifier
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct VolumeId(u32);

/// Polygon identifier
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct PolygonId(u32);

/// Volume's polygon
pub struct VolumePolygon {
    /// Volume portal polygon
    polygon: geom::Polygon,

    /// Polygon color
    color: Vec3f,
}

impl VolumePolygon {
    /// Get polygon
    pub fn get_polygon(&self) -> &geom::Polygon {
        &self.polygon
    }

    /// Get polygon color
    pub fn get_color(&self) -> Vec3f {
        self.color
    }
}

/// Portal to another volume
pub struct VolumePortal {
    /// Polygon to cut output by
    polygon: geom::Polygon,

    /// Destination volume index
    destination: VolumeId,
}

/// Volume structure (single convex volume)
pub struct Volume {
    /// Unique identifier
    id: VolumeId,
}

impl Volume {
    /// Get volume unique identifier
    pub fn get_id(&self) -> VolumeId {
        self.id
    }
}

/// Location BSP element (thing)
pub enum LocationBsp {
    Plane {
        /// Splitter plane
        plane: geom::Plane,

        /// Front
        front: Option<Box<LocationBsp>>,

        /// Back
        back: Option<Box<LocationBsp>>,
    },
    /// Destination volume Id
    Volume(VolumeId),
}

impl LocationBsp {
    /// Find corresponding volume
    pub fn find_volume(&self, position: Vec3f) -> Option<VolumeId> {
        match self {
            Self::Plane {
                plane,
                front,
                back
            } => {
                let leaf = match plane.get_point_relation(position) {
                    geom::PointRelation::Front | geom::PointRelation::OnPlane => front.as_ref(),
                    geom::PointRelation::Back => back.as_ref(),
                };

                leaf
                    .map(|v| v.find_volume(position))
                    .flatten()
            }
            Self::Volume(id) => Some(*id),
        }
    }
}

pub struct Location {
    /// Set of location volumes
    volumes: Vec<Volume>,
}

pub struct BuilderHullPolygon {
    /// Polygon itself
    pub polygon: geom::Polygon,

    /// Do polygon belong to set of polygons of initial hull
    pub is_external: bool,

    /// Was polygon plane already used as splitter
    pub is_splitter: bool,
}
pub struct BuilderHullVolume {
    /// Polygons
    pub hull_polygons: Vec<BuilderHullPolygon>,

    pub internal_polygons: Vec<BuilderHullPolygon>,
}


/// Portal polygon resolution step
pub struct FinalPolygonFaceId {
    pub volume_id: VolumeId,
    pub portal_face_id: u32,
}

/// Set of 
pub struct PortalResolvePromise {
    /// Common polygon of all portals
    pub common_polygon: geom::Polygon,

    /// Filter polygons 
    pub resolution_id: u32,
}

impl BuilderHullVolume {
    /// Build hull volume from boundbox
    pub fn from_bound_box(boundbox: geom::BoundBox) -> BuilderHullVolume {

        todo!()
    }
}

pub struct Builder {
    volumes: Vec<BuilderHullVolume>,
}

impl Location {
    pub fn build(polygons: Vec<geom::Polygon>) -> Location {
        // Calculate boundbox for
        let global_bound_box = polygons
            .iter()
            .fold(geom::BoundBox::zero(), |bb, polygon| {
                let polygon_bound_box = geom::BoundBox::for_points(
                    polygon.points.iter().copied()
                );

                bb.total(&polygon_bound_box)
            });

        let _first_volume = BuilderHullVolume::from_bound_box(
            global_bound_box.extend(Vec3f::new(30.0, 30.0, 30.0))
        );
        // Build initial volume from global bound box


        todo!()
    }

    
    /// Determine which (unit) volume point is locacted in
    pub fn get_point_volume(&self, _point: Vec3f) -> Option<VolumeId> {

        unimplemented!()
    } // get_point_volume
}

// map_bsp.rs
