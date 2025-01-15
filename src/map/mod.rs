use crate::{geom, math::Vec3f};

/// Declare actual map builder module
mod builder;


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
impl Location {

    pub fn build(brushes: Vec<crate::Brush>) -> Location {
        builder::build(brushes)
    }

    
    /// Determine which (unit) volume point is locacted in
    pub fn get_point_volume(&self, _point: Vec3f) -> Option<VolumeId> {
        unimplemented!()
    } // get_point_volume
}

// map_bsp.rs
