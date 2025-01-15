/*
1. Build BSP polyhedra from plane sets
2. Calculate hull volume
3. Build 'render polyhedra' and corresponding rendering BSP
4. Resolve render polyhedra portals
*/

use std::num::NonZeroU32;

use crate::{brush::Brush, geom, math::Vec3f};
use super::Location;

/// Actual polygon
pub struct PhysicalPolygon {
    /// Actual polygon
    pub polygon: geom::Polygon,
}

/// Hull polygon
struct HullPolygon {
    /// Polygon itself
    pub polygon: geom::Polygon,

    /// Do polygon belong to set of polygons of initial hull
    /// (used for invisible surface removal)
    pub is_external: bool,

    /// Set of all polygons that will be actually displayed
    pub physical_polygons: Vec<PhysicalPolygon>,

    /// Reference to split this polygon born at
    pub split_reference: Option<SplitReference>,
}

/// Polygon split id
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct SplitId(NonZeroU32);

/// Polygon split id
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct VolumeId(NonZeroU32);

/// Reference to certain split pass during split pass
pub struct SplitReference {
    /// split operation index
    pub split_id: SplitId,

    /// true if portal belongs to front part of split, false - if to back
    pub edge_is_front: bool,
}

/// Hull volume
struct HullVolume {
    /// Set of external polygons
    pub polygons: Vec<HullPolygon>,
}

/// Volume BSP
/// Note: option is used, because some volumes can be destroyed
/// during invisible surface removal pass
pub enum VolumeBsp {
    /// Volume BSP node, corresponds to some split
    Node {
        /// Splitter plane
        plane: geom::Plane,

        /// Volume bsp leaf to go to in case if point is in front of plane
        front: Option<Box<VolumeBsp>>,

        /// Volume bsp leaf to go to in case if point is in back of plane
        back: Option<Box<VolumeBsp>>,
    },

    /// Volume BSP leaf, corresponds to final index
    Leaf(VolumeId),
}

impl VolumeBsp {
    /// Traverse volume BSP
    pub fn traverse(&self, point: Vec3f) -> Option<VolumeId> {
        match self {
            Self::Node { plane, front, back } => {
                let side = match plane.get_point_relation(point) {
                    geom::PointRelation::Front | geom::PointRelation::OnPlane => front,
                    geom::PointRelation::Back => back,
                };

                side.as_ref()?.traverse(point)
            }
            Self::Leaf(volume_id) => Some(*volume_id),
        }
    }
}

/// Statistics about volume split, used during splitter selection process to
/// find best fitting splitter
pub struct VolumeSplitStat {
    /// Count of polygons in front of splitter plane
    pub front: u32,

    /// Count of polygon behind splitter plane
    pub back: u32,

    /// Count of splits occured
    pub split: u32,

    // Indicent volumes aren't accounted, because it's assumed that all possible splitter planes 
    // are located IN volume user is getting statistics for.

    // pub indicent: u32,
}

impl HullVolume {
    /// Build hull volume from boundbox
    pub fn from_bound_box(boundbox: geom::BoundBox) -> HullVolume {

        // Box vertices
        let mut vertices = [
            Vec3f::new(0.0, 0.0, 0.0),
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(1.0, 1.0, 0.0),
            Vec3f::new(0.0, 1.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            Vec3f::new(1.0, 0.0, 1.0),
            Vec3f::new(1.0, 1.0, 1.0),
            Vec3f::new(0.0, 1.0, 1.0),
        ];
        
        // t (x/y/z): [0, 1] -> [bb.min.t, bb.max.t]
        for vertex in &mut vertices {
            *vertex = *vertex * (boundbox.max() - boundbox.min()) + boundbox.min();
        }
        
        // Box indices
        let indices = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ];

        HullVolume {
            polygons: indices
                .iter()
                .copied()
                .map(|[a, b, c, d]| HullPolygon {
                    polygon: geom::Polygon::from_cw(vec![
                        vertices[a],
                        vertices[b],
                        vertices[c],
                        vertices[d],
                    ]),
                    is_external: true,
                    physical_polygons: Vec::new(),
                    split_reference: None,
                })
                .collect()
        }
    }

    /// Calculate statistics about splitting
    pub fn get_split_stat(&self, plane: geom::Plane) -> VolumeSplitStat {
        let mut result = VolumeSplitStat {
            front: 0,
            back: 0,
            split: 0,
        };

        _ = self.polygons
            .iter()
            .flat_map(|polygon| {
                std::iter::once(&polygon.polygon)
                    .chain(polygon.physical_polygons
                        .iter()
                        .map(|p| &p.polygon)
                    )
            })
            .fold(&mut result, |stat, polygon| {
                match plane.get_polygon_relation(polygon) {
                    geom::PolygonRelation::Front => {
                        stat.front += 1;
                    },
                    geom::PolygonRelation::Back => {
                        stat.back += 1;
                    },
                    geom::PolygonRelation::Intersects => {
                        stat.front += 1;
                        stat.back += 1;
                        stat.split += 1;
                    },
                    geom::PolygonRelation::OnPlane => {
                        // It's technically **should not** be reached,
                        // but I don't want to have a panic here...
                        eprintln!("Potential splitter plane is incident face of volume it holds.");
                    }
                }

                stat
            })
        ;

        result
    }

    /// Split hull volume by plane
    /// # Inputs
    /// - `plane` plane to split by
    /// - `indicent_front_polygons` polygons
    /// - `indicent_back_polygons` polygons
    /// 
    /// # Result
    /// - front polygon
    /// - back polygon
    /// - polygon || plane
    pub fn split(
        self,
        plane: geom::Plane,
        incident_front_polygons: Vec<PhysicalPolygon>,
        incident_back_polygon: Vec<PhysicalPolygon>,
        split_id: SplitId,
    ) -> (HullVolume, HullVolume, geom::Polygon) {
        todo!()
    }
}

pub struct SplitInfo {
    /// Polygon used to split
    pub split_polygon: geom::Polygon,
}

pub struct Builder {
    /// Final volumes
    volumes: Vec<HullVolume>,

    /// Split infos
    split_infos: Vec<SplitInfo>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            split_infos: Vec::new(),
        }
    }

    fn add_volume(&mut self, volume: HullVolume) -> VolumeId {
        let id = NonZeroU32::try_from(self.volumes.len() as u32 + 1).unwrap();
        self.volumes.push(volume);
        VolumeId(id)
    }

    fn get_next_split_id(&self) -> SplitId {
        SplitId(NonZeroU32::try_from(self.split_infos.len() as u32 + 1).unwrap())
    }

    fn add_split(&mut self, info: SplitInfo) {
        self.split_infos.push(info);
    }

    fn get_volume(&self, id: VolumeId) -> Option<&HullVolume> {
        self.volumes.get(id.0.get() as usize - 1)
    }

    fn build_volume(&mut self, volume: HullVolume, physical_polygons: Vec<PhysicalPolygon>) -> VolumeBsp {
        // final polygon, insert hull in volume set and return
        if physical_polygons.is_empty() {
            return VolumeBsp::Leaf(self.add_volume(volume));
        }

        // try to find optimal splitter
        let mut best_splitter_rate: f32 = std::f32::MAX;
        let mut best_splitter_index: usize = 0; // use polygon 0 as splitter by default

        // iterate through all polygons and try to find best one
        for (physical_polygon_index, physical_polygon) in physical_polygons.iter().enumerate() {
            let mut current_front: u32 = 0;
            let mut current_back: u32 = 0;
            let mut current_on: u32 = 0;
            let mut current_split: u32 = 0;

            // collect physical polygon statistics
            for second_physical_polygon in physical_polygons.iter() {
                if std::ptr::eq(physical_polygon, second_physical_polygon) {
                    continue;
                }

                let relation = physical_polygon.polygon.plane
                    .get_polygon_relation(&second_physical_polygon.polygon);

                match relation {
                    geom::PolygonRelation::Front => current_front += 1,
                    geom::PolygonRelation::OnPlane => current_on += 1,
                    geom::PolygonRelation::Back => current_back += 1,
                    geom::PolygonRelation::Intersects => {
                        current_front += 1;
                        current_back += 1;
                        current_split += 1;
                    }
                }
            }

            // // check if splitter is splitter, actually
            // if current_front == 0 && current_back == 0 && current_on == 0 {
            //     continue;
            // }

            let volume_split_stat = volume.get_split_stat(physical_polygon.polygon.plane);

            let volume_polygon_rate = 
                volume_split_stat.front.abs_diff(volume_split_stat.back) as f32 + volume_split_stat.split as f32;

            // kinda heuristics
            let splitter_rate = 0.0
                + current_back.abs_diff(current_front) as f32
                + current_split as f32 * 0.5
                - current_on as f32
                + volume_polygon_rate * 0.25
            ;

            if splitter_rate < best_splitter_rate {
                best_splitter_rate = splitter_rate;
                best_splitter_index = physical_polygon_index;
            }
        }
        
        // use splitter to...split!
        let splitter_plane = physical_polygons
            .get(best_splitter_index)
            .unwrap()
            .polygon
            .plane
        ;

        let mut front_polygons: Vec<PhysicalPolygon> = Vec::new();
        let mut front_incident_polygons: Vec<PhysicalPolygon> = Vec::new();
        let mut back_incident_polygons: Vec<PhysicalPolygon> = Vec::new();
        let mut back_polygons: Vec<PhysicalPolygon> = Vec::new();

        // classify all polygons to 4 types
        for physical_polygon in physical_polygons {
            match splitter_plane.split_polygon(&physical_polygon.polygon) {
                // to front
                geom::PolygonSplitResult::Front => front_polygons.push(physical_polygon),
                // to back
                geom::PolygonSplitResult::Back => back_polygons.push(physical_polygon),
                // to front/back incident
                geom::PolygonSplitResult::OnPlane => {
                    let cos = physical_polygon.polygon.plane.normal ^ splitter_plane.normal;

                    // che
                    if cos >= 0.0 {
                        front_incident_polygons.push(physical_polygon);
                    } else {
                        back_incident_polygons.push(physical_polygon);
                    }
                }
                // to front and back
                geom::PolygonSplitResult::Intersects { front, back } => {
                    // it's not good to split polygon
                    front_polygons.push(PhysicalPolygon { polygon: front });
                    back_polygons.push(PhysicalPolygon { polygon: back });
                }
            }
        }
    
        // get current split identifier
        let split_id = self.get_next_split_id();

        let (front_volume, back_volume, split_polygon) = volume.split(splitter_plane, front_incident_polygons, back_incident_polygons, split_id);

        let front_bsp = self.build_volume(front_volume, front_polygons);
        let back_bsp = self.build_volume(back_volume, back_polygons);

        // add split info
        self.add_split(SplitInfo { split_polygon });

        VolumeBsp::Node {
            front: Some(Box::new(front_bsp)),
            back: Some(Box::new(back_bsp)),
            plane: splitter_plane,
        }
    }

    /// Start volume building pass
    pub fn start_build_volumes(&mut self, brushes: &[Brush]) {
        // calculate polygon set hull
        let hull = brushes
            .iter()
            .fold(
                geom::BoundBox::zero(),
                |total, brush| total.total(&brush.bound_box)
            )
            .extend(Vec3f::new(16.0, 16.0, 16.0));

        // calculate hull volume
        let hull_volume = HullVolume::from_bound_box(hull);

        // get brush polygons
        let polygons = brushes
            .iter()
            .flat_map(|v| v.polygons.iter())
            .map(|polygon| PhysicalPolygon { polygon: polygon.clone() })
            .collect::<Vec<_>>();

        self.build_volume(hull_volume, polygons);
    }

    /// Start portal resolve pass
    pub fn start_resolve_portals(&mut self) {

    }

    /// Remove invisible volumes
    pub fn start_remove_invisible(&mut self) {

    }
}

pub fn build(brushes: Vec<Brush>) -> Location {
    let mut builder = Builder::new();

    // Build volumes
    builder.start_build_volumes(&brushes);

    // Resolve volume portals
    builder.start_resolve_portals();

    // Remove invisible surfaces
    builder.start_remove_invisible();

    unimplemented!()
}
