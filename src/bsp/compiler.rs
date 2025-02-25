///! WBSP compiler implementation file

/*
1. Build BSP polyhedra from plane sets
2. Calculate hull volume
3. Build 'render polyhedra' and corresponding rendering BSP
4. Resolve render polyhedra portals
5. Remove invisible sections (all volumes that contains external
    polygons or portal to such are considered invisible.
    That's the exact reason why levels must not have 'holes' in their structure)
*/

use std::{collections::{BTreeMap, BTreeSet, HashMap}, num::NonZeroU32};
use crate::{bsp::util, geom, map, math::{Mat3f, Vec3f}};

use super::Id;

/// Polygon that should be drawn
#[derive(Debug)]
pub struct PhysicalPolygon {
    /// Actual polygon
    pub polygon: geom::Polygon,

    /// All material info (just for debug)
    pub material_index: usize,

    /// Material mapping U axis
    pub material_u: geom::Plane,

    /// Material mapping V axis
    pub material_v: geom::Plane,

    /// True if physical polygon is built from transparent material, false if not
    pub is_transparent: bool,

    /// True if this physical polygon refers to sky, false if not
    pub is_sky: bool,
}

/// Reference to another volume
#[derive(Debug)]
pub struct PortalPolygon {
    /// Index of actual polygon
    pub polygon_set_index: usize,

    /// True denotes that portal polygon should be used 'as-is', False - as check results should be reversed
    pub is_front: bool,

    /// Destination volume
    pub dst_volume_index: usize,
}

/// Hull polygon
#[derive(Debug)]
pub struct HullFace {
    /// Polygon itself
    pub polygon: geom::Polygon,

    /// Do polygon belong to set of polygons of initial hull
    /// (used in invisible surface removal pass)
    pub is_external: bool,

    /// Set of all polygons that will be actually displayed
    pub physical_polygons: Vec<PhysicalPolygon>,

    /// Polygons, that 'points' to another volumes
    pub portal_polygons: Vec<PortalPolygon>,

    /// Reference to split this polygon born at
    /// (used in portal resolution pass)
    pub split_reference: Option<SplitReference>,
}

/// Polygon split id
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SplitId(NonZeroU32);

/// Reference to certain split pass during split pass
#[derive(Copy, Clone, Debug)]
pub struct SplitReference {
    /// split operation index
    pub split_id: SplitId,

    /// true if portal belongs to front part of split, false - if to back
    pub edge_is_front: bool,
}

/// Hull volume
#[derive(Debug)]
pub struct HullVolume {
    /// Set of external polygons
    pub faces: Vec<HullFace>,
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
    Leaf(usize),
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
            faces: indices
                .iter()
                .copied()
                .map(|[a, b, c, d]| HullFace {
                    polygon: geom::Polygon::from_cw(vec![
                        vertices[a],
                        vertices[b],
                        vertices[c],
                        vertices[d],
                    ]),
                    is_external: true,
                    physical_polygons: Vec::new(),
                    portal_polygons: Vec::new(),
                    split_reference: None,
                })
                .collect()
        }
    }


    /// Check if volume contains point or not
    pub fn contains_point(&self, point: Vec3f) -> bool {
        self
            .faces
            .iter()
            .all(|face| {
                face.polygon.plane.get_point_relation(point) != geom::PointRelation::Back
            })
    }

    /// Calculate statistics about splitting
    pub fn get_split_stat(&self, plane: geom::Plane) -> VolumeSplitStat {
        let mut result = VolumeSplitStat {
            front: 0,
            back: 0,
            split: 0,
        };

        _ = self.faces
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
                    geom::PolygonRelation::Coplanar => {
                        // It's technically **should not** be reached,
                        // but I don't want to have a panic here...
                        eprintln!("Potential splitter plane is incident to face of volume it holds.");
                    }
                }

                stat
            })
        ;

        result
    }

    // Get intersection polygon of current volume and plane
    fn get_intersection_polygon(&self, plane: geom::Plane) -> Option<geom::Polygon> {
        // Get ALL splitter plane intersection points

        let mut intersection_points = self.faces
            .iter()
            .map(|hull_polygon| plane.intersect_polygon(&hull_polygon.polygon))
            .filter_map(|v| v)
            .flat_map(|(begin, end)| [begin, end].into_iter())
            .collect::<Vec<Vec3f>>();

        // Deduplicate and sort them
        intersection_points = geom::deduplicate_points(intersection_points);

        if intersection_points.len() < 3 {
            eprintln!("Something strange happened...");
            None
        } else {
            Some(geom::Polygon {
                points: geom::sort_points_by_angle(intersection_points, plane.normal),
                plane
            })
        }
    }

    /// Split hull volume by plane
    /// 
    /// # Inputs
    /// - `plane` plane to split by
    /// - `indicent_front_polygons` polygons that should belong to front hull
    /// - `indicent_back_polygons` polygons that should belong to back hull
    /// - `split_id` ID of split operation (used during portal resolve pass)
    /// 
    /// # Result
    /// Triple, that contains:
    /// - front hull
    /// - back hull
    /// - split polygon (polygon normal directed as plane)
    /// 
    /// # Note
    /// This function is one of the main BSP building functions - it,
    /// actually, performs polygon volume split.
    pub fn split(
        self,
        plane: geom::Plane,
        incident_front_polygons: Vec<PhysicalPolygon>,
        incident_back_polygons: Vec<PhysicalPolygon>,
        split_id: SplitId,
    ) -> Result<(Self, Self, geom::Polygon), Self> {
  
        // intersection polygon MUST exist

        let mut intersection_polygon = match self.get_intersection_polygon(plane) {
            Some(polygon) => polygon,
            None => return Err(self),
        };

        let mut front_hull_polygons = Vec::new();
        let mut back_hull_polygons = Vec::new();

        /*  */

        for hull_polygon in self.faces {
            match plane.split_polygon(&hull_polygon.polygon) {
                geom::PolygonSplitResult::Front => {
                    front_hull_polygons.push(hull_polygon);
                }
                geom::PolygonSplitResult::Back => {
                    back_hull_polygons.push(hull_polygon);
                }
                geom::PolygonSplitResult::Intersects { front: front_hull_polygon, back: back_hull_polygon } => {
                    // split physical polygons
                    let mut front_physical_polygons = Vec::new();
                    let mut back_physical_polygons = Vec::new();

                    for physical_polygon in hull_polygon.physical_polygons {
                        match plane.split_polygon(&physical_polygon.polygon) {
                            geom::PolygonSplitResult::Front => {
                                front_physical_polygons.push(physical_polygon);
                            }
                            geom::PolygonSplitResult::Back => {
                                back_physical_polygons.push(physical_polygon);
                            }
                            geom::PolygonSplitResult::Intersects { front, back } => {
                                front_physical_polygons.push(PhysicalPolygon {
                                    polygon: front,
                                    material_index: physical_polygon.material_index,
                                    material_u: physical_polygon.material_u,
                                    material_v: physical_polygon.material_v,
                                    is_transparent: physical_polygon.is_transparent,
                                    is_sky: physical_polygon.is_sky,
                                });

                                back_physical_polygons.push(PhysicalPolygon {
                                    polygon: back,
                                    material_index: physical_polygon.material_index,
                                    material_u: physical_polygon.material_u,
                                    material_v: physical_polygon.material_v,
                                    is_transparent: physical_polygon.is_transparent,
                                    is_sky: physical_polygon.is_sky,
                                });
                            }
                            geom::PolygonSplitResult::Coplanar => {
                                // probably panic here...
                                eprintln!("Physical polygon somehow lies on volume splitter plane...");
                            }
                        }
                    }

                    front_hull_polygons.push(HullFace {
                        is_external: hull_polygon.is_external,
                        physical_polygons: front_physical_polygons,
                        portal_polygons: Vec::new(),
                        polygon: front_hull_polygon,
                        split_reference: hull_polygon.split_reference,
                    });

                    back_hull_polygons.push(HullFace {
                        is_external: hull_polygon.is_external,
                        physical_polygons: back_physical_polygons,
                        portal_polygons: Vec::new(),
                        polygon: back_hull_polygon,
                        split_reference: hull_polygon.split_reference,
                    });
                }
                geom::PolygonSplitResult::Coplanar => {
                    eprintln!("Hull face somehow lies on volume splitter plane...");
                }
            }
        }

        let split_polygon = intersection_polygon.clone();

        let front_intersection_hull_polygon = HullFace {
            is_external: false,
            physical_polygons: incident_front_polygons,
            portal_polygons: Vec::new(), // constructed during portal_resolve pass
            polygon: intersection_polygon.clone(),
            split_reference: Some(SplitReference { edge_is_front: false, split_id })
        };
        
        // negate intersection polygon's orientation to match back
        intersection_polygon.negate_orientation();
        let back_intersection_hull_polygon = HullFace {
            is_external: false,
            physical_polygons: incident_back_polygons,
            portal_polygons: Vec::new(),
            polygon: intersection_polygon,
            split_reference: Some(SplitReference { edge_is_front: true, split_id })
        };

        front_hull_polygons.push(front_intersection_hull_polygon);
        back_hull_polygons.push(back_intersection_hull_polygon);

        let front_hull_volume = HullVolume { faces: front_hull_polygons };
        let back_hull_volume = HullVolume { faces: back_hull_polygons };

        Ok((front_hull_volume, back_hull_volume, split_polygon))
    }
}

/// Information about split operation
pub struct SplitInfo {
    /// Polygon all splitter polygons (should be) contained in
    pub split_polygon: geom::Polygon,

    /// Split Id
    pub id: SplitId,
}

/// Portal resolution pass internal polygon structure
pub struct SplitTaskPolygon {
    /// Index of volume portals generated by the polygon will belong to
    pub dst_volume_index: usize,

    /// Destination
    pub dst_face_index: usize,

    /// Polygon boundbox (calculated during task polygon building subpass)
    pub polygon_bb: geom::BoundBox,

    /// Polygon
    pub polygon: geom::Polygon,
}

/// Single BSP model compilation context
pub struct BspModelCompileContext {
    /// 'Final' volumes
    pub volumes: Vec<HullVolume>,

    /// Split infos
    pub split_infos: Vec<SplitInfo>,

    /// Volume
    pub volume_bsp: Option<Box<VolumeBsp>>,

    /// Set of portal polygons (They should be shared to optimize total memory consume)
    pub portal_polygons: Vec<geom::Polygon>,

    /// Global bounding box
    pub bound_box: geom::BoundBox,
}

impl BspModelCompileContext {
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            split_infos: Vec::new(),
            portal_polygons: Vec::new(),
            volume_bsp: None,
            bound_box: geom::BoundBox::zero(),
        }
    }

    // Generate new split id
    fn get_next_split_id(&self) -> SplitId {
        SplitId(NonZeroU32::try_from(self.split_infos.len() as u32 + 1).unwrap())
    }

    fn add_split(&mut self, info: SplitInfo) {
        self.split_infos.push(info);
    }

    fn build_volume(&mut self, volume: HullVolume, physical_polygons: Vec<PhysicalPolygon>) -> VolumeBsp {
        // final polygon, insert hull in volume set and return
        if physical_polygons.is_empty() {

            // for face in &volume.faces {
            //     if let Some(split_ref) = face.split_reference {
            //         let split_info = self.split_infos.get(split_ref.split_id.0.get() as usize - 1).unwrap();
            //         if face.polygon.plane != split_info.split_polygon.plane && face.polygon.plane != split_info.split_polygon.plane.negate_direction() {
            //             panic!("Early planarity check failed");
            //         }
            //     }
            // }

            self.volumes.push(volume);
            return VolumeBsp::Leaf(self.volumes.len() - 1);
        }

        for physical_polygon in &physical_polygons {
            for point in &physical_polygon.polygon.points {
                if !self.bound_box.contains_point(point) {
                    eprintln!("Boundbox somehow does not contain point");
                }
            }
        }

        // try to find optimal splitter
        let mut best_splitter_rate: f32 = std::f32::MAX;
        let mut best_splitter_index: Option<usize> = None; // use polygon 0 as splitter by default

        // iterate through all polygons and try to find best one
        'splitter_search_loop: for (physical_polygon_index, physical_polygon) in physical_polygons.iter().enumerate() {
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
                    geom::PolygonRelation::Coplanar => current_on += 1,
                    geom::PolygonRelation::Back => current_back += 1,
                    geom::PolygonRelation::Intersects => {
                        current_front += 1;
                        current_back += 1;
                        current_split += 1;
                    }
                }
            }

            let volume_split_stat = volume.get_split_stat(physical_polygon.polygon.plane);

            if volume_split_stat.front == 0 || volume_split_stat.back == 0 {
                continue 'splitter_search_loop;
            }

            let volume_polygon_rate = 
            volume_split_stat.front.abs_diff(volume_split_stat.back) as f32 + volume_split_stat.split as f32;

            std::hint::black_box(&volume_split_stat);

            // -- kind of heuristics
            let splitter_rate = 0.0
                + current_back.abs_diff(current_front) as f32 * 1.0
                + current_split as f32 * 4.0
                - current_on as f32 * 4.0
                + volume_polygon_rate * 0.125
            ;

            if splitter_rate < best_splitter_rate {
                best_splitter_rate = splitter_rate;
                best_splitter_index = Some(physical_polygon_index);
            }
        }
        
        // Check case where there's no splitter at
        let Some(best_splitter_index) = best_splitter_index else {
            self.volumes.push(volume);
            return VolumeBsp::Leaf(self.volumes.len() - 1);
        };

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
                geom::PolygonSplitResult::Coplanar => {
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
                    front_polygons.push(PhysicalPolygon {
                        polygon: front,
                        material_index: physical_polygon.material_index,
                        material_u: physical_polygon.material_u,
                        material_v: physical_polygon.material_v,
                        is_transparent: physical_polygon.is_transparent,
                        is_sky: physical_polygon.is_sky,
                    });

                    back_polygons.push(PhysicalPolygon {
                        polygon: back,
                        material_index: physical_polygon.material_index,
                        material_u: physical_polygon.material_u,
                        material_v: physical_polygon.material_v,
                        is_transparent: physical_polygon.is_transparent,
                        is_sky: physical_polygon.is_sky,
                    });
                }
            }
        }
    
        // Get current split identifier
        let split_id = self.get_next_split_id();

        // Split MUST be possible, because splitter plane face belongs to hull volume (by definition)
        // If it's not the case, something went totally wrong.
        let volume_split_result = volume.split(
            splitter_plane,
            front_incident_polygons,
            back_incident_polygons,
            split_id
        );

        match volume_split_result {
            Ok((front_volume, back_volume, split_polygon)) => {
                // add split info
                self.add_split(SplitInfo { split_polygon, id: split_id });
        
                let front_bsp = self.build_volume(front_volume, front_polygons);
                let back_bsp = self.build_volume(back_volume, back_polygons);
        
                VolumeBsp::Node {
                    front: Some(Box::new(front_bsp)),
                    back: Some(Box::new(back_bsp)),
                    plane: splitter_plane,
                }
            }
            // Split failed
            Err(old_volume) => {
                self.volumes.push(old_volume);

                VolumeBsp::Leaf(self.volumes.len() - 1)
            }
        }
    }

    /// Start volume building pass
    pub fn start_build_volumes(&mut self, physical_polygons: Vec<PhysicalPolygon>) {

        // Calculate boundbox
        self.bound_box = geom::BoundBox::for_points(physical_polygons
            .iter()
            .flat_map(|polygon| polygon
                .polygon
                .points
                .iter()
                .copied()
            )
        );

        // Slightly extend model boundbox
        self.bound_box = self.bound_box.extend(Vec3f::new(1.0, 1.0, 1.0));

        // calculate hull volume
        let hull_volume = HullVolume::from_bound_box(
            self.bound_box.extend(Vec3f::new(200.0, 200.0, 200.0))
        );

        // Build volumes
        let volume_bsp = self.build_volume(hull_volume, physical_polygons);

        // write volume BSP
        self.volume_bsp = Some(Box::new(volume_bsp));
    }

    /// Build resolve tasks
    /// (mapping between split id and sets of references (volume_id, volume_face_id))
    fn build_resolve_tasks(&self) -> BTreeMap<SplitId, Vec<(usize, usize)>> {
        // Construct initial structure
        let mut tasks = self.split_infos
            .iter()
            .map(|split_info| (split_info.id, Vec::new()))
            .collect::<BTreeMap<SplitId, Vec<(usize, usize)>>>();

        // Fill volume arrays
        for (volume_index, volume) in self.volumes.iter().enumerate() {
            'face_loop: for (hull_face_index, hull_face) in volume.faces.iter().enumerate() {
                let Some(split_ref) = hull_face.split_reference else {
                    continue 'face_loop;
                };
                
                tasks
                    .get_mut(&split_ref.split_id)
                    .unwrap()
                    .push((volume_index, hull_face_index));
            }
        }

        tasks
    }

    /// Build front and back polygon sets of single portal resolve task
    /// # Returns
    /// Tuple that contains front and back polygon sets
    fn build_task_polygon_sets(
        &self,
        task: &[(usize, usize)]
    ) -> (Vec<SplitTaskPolygon>, Vec<SplitTaskPolygon>) {
        /// Polygon cutting descriptor
        pub struct CutInfo {
            /// Plane set
            pub planes: Vec<geom::Plane>,

            /// Cutter boundbox
            pub bound_box: geom::BoundBox,
        }

        let cut_info_set = task
            .iter()
            .copied()
            .flat_map(|(volume_index, face_index)| {
                let volume = self.volumes.get(volume_index).unwrap();
                let face = volume.faces.get(face_index).unwrap();

                face.physical_polygons.iter()
            })
            .filter_map(|physical_polygon| {
                // Do not cut portals by transparent polygons
                if physical_polygon.is_transparent {
                    None
                } else {
                    Some(CutInfo {
                        planes: physical_polygon.polygon.iter_edge_planes().collect(),
                        bound_box: physical_polygon.polygon.build_bound_box(),
                    })
                }
            })
            .collect::<Vec<_>>();

        // Front polygons to resolve
        let mut total_front_polygons = Vec::<SplitTaskPolygon>::new();

        // Back polygons to resolve
        let mut total_back_polygons = Vec::<SplitTaskPolygon>::new();

        'task_loop: for (volume_index, face_index) in task.iter().copied() {
            let volume = self.volumes.get(volume_index).unwrap();
            let face = volume.faces.get(face_index).unwrap();

            // Build potential portal polygon set
            let mut portal_polygons = vec![(
                face.polygon.clone(),
                face.polygon.build_bound_box())
            ];

            for cut_info in &cut_info_set {
                let mut new_portal_polygons = Vec::new();

                // possible improvement:
                // check if portal_polygon is in front of some of
                // cut_edge, and only AFTER check perform splitting.

                'portal_polygon_loop: for (mut portal_polygon, mut portal_polygon_bb) in portal_polygons {
                    if !cut_info.bound_box.is_intersecting(&portal_polygon_bb) {
                        new_portal_polygons.push((portal_polygon, portal_polygon_bb));
                        continue 'portal_polygon_loop;
                    }

                    // cut portal_polygon by cut_polygon
                    for cut_edge_plane in &cut_info.planes {
                        match cut_edge_plane.split_polygon(&portal_polygon) {
                            geom::PolygonSplitResult::Front => {
                                // resolution process finished.
                                new_portal_polygons.push((portal_polygon, portal_polygon_bb));
                                continue 'portal_polygon_loop;
                            }
                            geom::PolygonSplitResult::Back => {
                                // nothing changes
                            }
                            geom::PolygonSplitResult::Intersects { front, back } => {
                                portal_polygon_bb = back.build_bound_box();
                                portal_polygon = back;

                                let bb = front.build_bound_box();

                                new_portal_polygons.push((front, bb));
                            }
                            geom::PolygonSplitResult::Coplanar => {
                                // WTF?
                                // IDK, should I panic here, but...
                                eprintln!("Edgeplane of first of coplanar polygons somehow contains another one...");
                            }
                        }
                    }
                }

                portal_polygons = new_portal_polygons;
            }

            // Check if there's no portal polygons at all
            if portal_polygons.is_empty() {
                continue 'task_loop;
            }

            // Build split task iterator
            let split_task_iter = portal_polygons
                .into_iter()
                .map(|(polygon, polygon_bb)| SplitTaskPolygon {
                    dst_volume_index: volume_index,
                    dst_face_index: face_index,
                    polygon,
                    polygon_bb,
                });

            // Write task polygons to corresponding polygon set
            if face.split_reference.unwrap().edge_is_front {
                total_front_polygons.extend(split_task_iter);
            } else {
                total_back_polygons.extend(split_task_iter);
            }
        }

        (total_front_polygons, total_back_polygons)
    }

    /// Start portal resolve pass
    pub fn start_resolve_portals(&mut self) {

        // Build portal resolution 'tasks'
        let resolve_tasks = self.build_resolve_tasks();

        /*
        for (id, task_contents) in &resolve_tasks {
            let info = &self.split_infos[id.0.get() as usize - 1];
            let split_polygon = &info.split_polygon;
            let plane = split_polygon.plane;
            for (volume_id, face_id) in task_contents.iter().copied() {
                let face_polygon = &self.volumes[volume_id].faces[face_id].polygon;
                let face_plane = face_polygon.plane;
                if face_plane != plane && face_plane.negate_direction() != plane {
                    panic!("Task face coplanarity check failed!\n First plane: {:?}\nSecond plane: {:?}",
                        plane, face_plane
                    );
                }
            }
        }
         */

        // process tasks into polygon set
        'task_loop: for (_, task) in resolve_tasks {
            // Build front and back polygon set
            let (
                front_polygons,
                back_polygons
            ) = self.build_task_polygon_sets(&task);

            // Ignore task if there's no resolved polygons on one of faces
            if front_polygons.is_empty() || back_polygons.is_empty() {
                continue 'task_loop;
            }

            for front_polygon in front_polygons {
                let front_polygon_edge_planes = front_polygon
                    .polygon
                    .iter_edge_planes()
                    .collect::<Vec<_>>();

                'back_polygon_loop: for back_polygon in &back_polygons {
                    if !front_polygon.polygon_bb.is_intersecting(&back_polygon.polygon_bb) {
                        continue 'back_polygon_loop;
                    }

                    let mut portal = back_polygon.polygon.clone();

                    for edge_plane in front_polygon_edge_planes.iter().copied() {
                        match edge_plane.split_polygon(&portal) {
                            geom::PolygonSplitResult::Front => {
                                continue 'back_polygon_loop;
                            }
                            geom::PolygonSplitResult::Back => {
                            }
                            geom::PolygonSplitResult::Intersects { front, back } => {
                                portal = back;
                                _ = front;
                            }
                            // Polygon is degenerate
                            geom::PolygonSplitResult::Coplanar => {
                                // Something strange happened...
                                // eprintln!("Something strange happened because I don't want to use planar geometry...");
                                continue 'back_polygon_loop;
                            }
                        }
                    }

                    // Index of polygon in portal set
                    let portal_polygon_index = self.portal_polygons.len();
                    self.portal_polygons.push(portal);

                    self
                        .volumes[front_polygon.dst_volume_index]
                        .faces[front_polygon.dst_face_index]
                        .portal_polygons
                        .push(PortalPolygon {
                            dst_volume_index: back_polygon.dst_volume_index,
                            is_front: false,
                            polygon_set_index: portal_polygon_index,
                        });

                    self
                        .volumes[back_polygon.dst_volume_index]
                        .faces[back_polygon.dst_face_index]
                        .portal_polygons
                        .push(PortalPolygon {
                            dst_volume_index: front_polygon.dst_volume_index,
                            is_front: true,
                            polygon_set_index: portal_polygon_index,
                        })
                        ;
                }
            }
        }
    }

    /// Build index map based on set of polygons to remove and total set array size
    fn build_index_map(mut remove_set: Vec<usize>, total_count: usize) -> Vec<Option<usize>> {
        remove_set.sort();
        remove_set.reverse();

        let mut delta = 0;

        let mut index_map = Vec::<Option<usize>>::with_capacity(total_count);

        for i in 0..total_count {
            if let Some(removed_index) = remove_set.last().copied() {
                if i < removed_index {
                    index_map.push(Some(i - delta));
                } else {
                    index_map.push(None);
                    delta += 1;
                    remove_set.pop();
                }
            } else {
                index_map.push(Some(i - delta));
            }
        }

        index_map
    }

    /// Remove volumes by index set
    /// 
    /// # Note
    /// If remove set contains at least one volume from a connectivity component, set **must** contain
    /// rest of the connectivity component volumes. If it isn't true, BSP mutation will be incorrect,
    /// and it may cause very strange panics in future.
    fn remove_volumes(&mut self, mut removed_volume_idx: Vec<usize>) {
        removed_volume_idx.sort();
        removed_volume_idx.reverse();

        let portal_index_map = {
            let portal_remove_set = removed_volume_idx
                .iter()
                .flat_map(|id| {
                    let volume = self.volumes.get(*id).unwrap();
                    volume
                        .faces
                        .iter()
                        .flat_map(|face| face
                            .portal_polygons
                            .iter()
                            .map(|portal| portal.polygon_set_index)
                        )
                })
                .collect::<BTreeSet<_>>();

            Self::build_index_map(
                portal_remove_set.into_iter().collect(),
                self.portal_polygons.len()
            )
        };

        // Map portal polygons
        // Meh, vector construction. I hope that rust compiler is 'good enuf' to deal with things like this.
        self.portal_polygons = std::mem::replace(&mut self.portal_polygons, Vec::new())
            .into_iter()
            .enumerate()
            .filter_map(|(index, polygon)| {
                if portal_index_map.get(index).unwrap().is_some() {
                    Some(polygon)
                } else {
                    None
                }
            })
            .collect();

        let volume_index_map = Self::build_index_map(
            removed_volume_idx,
            self.volumes.len()
        );

        // Map volumes
        self.volumes = std::mem::replace(&mut self.volumes, Vec::new())
            .into_iter()
            .enumerate()
            .filter_map(|(volume_index, mut volume)| {
                if volume_index_map.get(volume_index).unwrap().is_none() {
                    return None;
                }

                for face in &mut volume.faces {
                    face.portal_polygons = std::mem::replace(&mut face.portal_polygons, Vec::new())
                        .into_iter()
                        .filter_map(|portal_polygon| {
                            let Some(new_volume_index) = volume_index_map[portal_polygon.dst_volume_index] else {
                                eprintln!("Reachable face somehow have no map...");
                                return None;
                            };
        
                            let Some(new_portal_polygon_index) = portal_index_map[portal_polygon.polygon_set_index] else {
                                eprintln!("Reachable face somehow have no polygon...");
                                return None;
                            };
        
                            Some(PortalPolygon {
                                dst_volume_index: new_volume_index,
                                polygon_set_index: new_portal_polygon_index,
                                is_front: portal_polygon.is_front,
                            })
                        })
                        .collect();
                }

                Some(volume)
            })
            .collect();

        // Map volume BSP

        fn map_volume_bsp(old_volume_bsp: VolumeBsp, index_map: &[Option<usize>]) -> Option<Box<VolumeBsp>> {
            match old_volume_bsp {
                VolumeBsp::Node { plane, front, back } => {
                    let front = map_volume_bsp(*front.unwrap(), index_map);
                    let back = map_volume_bsp(*back.unwrap(), index_map);

                    if front.is_none() && back.is_none() {
                        None
                    } else {
                        Some(Box::new(VolumeBsp::Node { plane, front, back }))
                    }
                },
                VolumeBsp::Leaf(index) => {
                    index_map[index].map(|new_index| Box::new(VolumeBsp::Leaf(new_index)))
                },
            }
        }

        if let Some(bsp) = std::mem::replace(&mut self.volume_bsp, None) {
            self.volume_bsp = map_volume_bsp(*bsp, &volume_index_map);
        }
    }

    /// Build sets of potentially-visible volumes
    fn build_volume_graph(&self) -> Vec<BTreeSet<usize>> {
        // Indices of ALL volumes in graph
        let mut total_volume_idx = BTreeSet::from_iter(0..self.volumes.len());
        let mut total_conn_components = Vec::<BTreeSet<usize>>::new();

        while let Some(first_index) = total_volume_idx.first().copied() {
            let mut conn_component = BTreeSet::<usize>::new();
            let mut component_edge = BTreeSet::<usize>::new();
            let mut new_component_edge = BTreeSet::<usize>::new();

            component_edge.insert(first_index);

            while !component_edge.is_empty() {

                for edge_elt_index in component_edge.iter().copied() {
                    let volume = &self.volumes[edge_elt_index];

                    let portal_idx_iter = volume
                        .faces
                        .iter()
                        .flat_map(|face| face
                            .portal_polygons
                            .iter()
                            .map(|pp| pp.dst_volume_index)
                        );

                    for dst_index in portal_idx_iter {
                        if conn_component.contains(&dst_index) {
                            continue;
                        }

                        new_component_edge.insert(dst_index);
                    }

                    conn_component.insert(edge_elt_index);
                }

                std::mem::swap(&mut component_edge, &mut new_component_edge);
                new_component_edge.clear();
            }

            // Remove current conn component elements from total index span
            for index in conn_component.iter() {
                total_volume_idx.remove(index);
            }

            // Add conn component to index span
            total_conn_components.push(conn_component);
        }

        total_conn_components
    }

    pub fn start_remove_invisible(&mut self, visible_from: Vec<Vec3f>) {
        let removed_index_set = self
            .build_volume_graph()
            .into_iter()
            .filter(|volume_index_set| volume_index_set
                .iter()
                .map(|index| &self.volumes[*index])
                .all(|volume| visible_from
                    .iter()
                    .copied()
                    .all(|pt| !volume.contains_point(pt))
                )
            )
            .flat_map(|set| set.into_iter())
            .collect::<Vec<_>>();

        self.remove_volumes(removed_index_set);
    }
}

/// Compilation context
pub struct CompileContext {
    /// Set of already added polygons
    pub polygon_set: Vec<geom::Polygon>,

    /// Set of material names
    pub material_name_set: Vec<String>,

    /// Material name table
    pub material_name_table: HashMap<String, usize>,

    /// Set of volumes
    pub volume_set: Vec<super::Volume>,

    /// Set of BSP models
    pub bsp_models: Vec<super::BspModel>,

    /// Dynamic model set
    pub dynamic_models: Vec<super::DynamicModel>,
}

impl CompileContext {
    // Build entity physical polygon set
    pub fn build_entity_physical_polygons(&mut self, entity: &map::Entity) -> Vec<PhysicalPolygon> {

        let mut physical_polygons = Vec::<PhysicalPolygon>::new();

        'brush_polygon_building: for brush in &entity.brushes {
            // Don't merge clip brushes into render BSP
            if brush.is_invisible {
                continue 'brush_polygon_building;
            }
    
            let planes = brush.faces
                .iter()
                .map(|face| {
                    let mtlid = self.material_name_table
                        .get(&face.mtl_name)
                        .copied()
                        .unwrap_or_else(|| {
                            let mtlid = self.material_name_set.len();
                            self.material_name_table.insert(face.mtl_name.clone(), mtlid);
                            self.material_name_set.push(face.mtl_name.clone());
    
                            mtlid
                        });
    
                    (face, mtlid)
                })
                .collect::<Vec<_>>();
    
            for (f1, mtlid) in &planes {
    
                let mut points = Vec::<Vec3f>::new();
    
                for (f2, _) in &planes {
                    if std::ptr::eq(f1, f2) {
                        continue;
                    }
    
                    for (f3, _) in &planes {
                        if std::ptr::eq(f1, f3) || std::ptr::eq(f2, f3) {
                            continue;
                        }
    
                        let mat = Mat3f::from_rows(
                            f1.plane.normal,
                            f2.plane.normal,
                            f3.plane.normal
                        );
    
                        let inv = match mat.inversed() {
                            Some(m) => m,
                            None => continue,
                        };
                        let intersection_point = inv * Vec3f::new(
                            f1.plane.distance,
                            f2.plane.distance,
                            f3.plane.distance,
                        );
    
                        points.push(intersection_point);
                    }
                }
    
                points = points
                    .into_iter()
                    .filter(|point| planes
                        .iter()
                        .all(|(face, _)| {
                            face.plane.get_point_relation(*point) != geom::PointRelation::Front
                        })
                    )
                    .collect::<Vec<_>>();
    
                points = geom::deduplicate_points(points);
    
                if points.len() < 3 {
                    // It's not even a polygon, actually
                    continue;
                }
    
                let points = geom::sort_points_by_angle(points, f1.plane.normal);
    
                if f1.is_transparent {
                    physical_polygons.push(PhysicalPolygon {
                        polygon: geom::Polygon {
                            points: {
                                let mut pts = points.clone();
                                pts.reverse();
                                pts
                            },
                            plane: f1.plane.negate_direction(),
                        },
                        material_index: *mtlid,
                        material_u: f1.u,
                        material_v: f1.v,
                        is_transparent: true,
                        is_sky: f1.is_sky,
                    });
                }
    
                // Build physical polygon
                physical_polygons.push(PhysicalPolygon {
                    polygon: geom::Polygon {
                        points: points,
                        plane: f1.plane,
                    },
                    material_index: *mtlid,
                    material_u: f1.u,
                    material_v: f1.v,
                    is_transparent: f1.is_transparent,
                    is_sky: f1.is_sky,
                });
            }
        }
    
        physical_polygons
    }

    /// Add dynamic model to BSP
    pub fn add_dynamic_model(&mut self, origin: Vec3f, rotation: f32, index: usize) -> usize {
        self.dynamic_models.push(super::DynamicModel {
            model_id: super::BspModelId::from_index(index),
            origin,
            rotation,
        });

        self.dynamic_models.len() - 1
    }

    /// Add model to final BSP
    pub fn add_model(&mut self, ctx: BspModelCompileContext) -> usize {
        fn map_bsp(vbsp: Option<Box<VolumeBsp>>, offset: usize) -> super::Bsp {
            let Some(bsp) = vbsp else {
                return super::Bsp::Void;
            };

            match *bsp {
                VolumeBsp::Node {
                    plane,
                    front,
                    back
                } => super::Bsp::Partition {
                    splitter_plane: plane,
                    front: Box::new(map_bsp(front, offset)),
                    back: Box::new(map_bsp(back, offset)),
                },
                VolumeBsp::Leaf(index) =>
                    super::Bsp::Volume(super::VolumeId::from_index(index + offset)),
            }
        }

        // Portal polygon index offset
        let portal_index_offset = self.polygon_set.len();

        // Volume index offset
        let volume_index_offset = self.volume_set.len();

        // Build BSP
        let bsp = Box::new(map_bsp(ctx.volume_bsp, volume_index_offset));

        // Extend polygon set with portals
        self.polygon_set.extend_from_slice(ctx.portal_polygons.as_slice());

        // Extend volume set with iterator
        self.volume_set.extend(ctx
            .volumes
            .into_iter()
            .map(|hull_volume| {
                let mut bound_box = geom::BoundBox::zero();

                for face in &hull_volume.faces {
                    bound_box = bound_box.total(
                        &geom::BoundBox::for_points(face.polygon.points.iter().copied())
                    );
                }

                let mut surfaces = Vec::new();
                let mut portals = Vec::new();

                for face in hull_volume.faces {
                    surfaces.extend(face
                        .physical_polygons
                        .into_iter()
                        .map(|physical_polygon| {

                            let (u_min, u_max, v_min, v_max) = util::calculate_uv_ranges(
                                &physical_polygon.polygon.points,
                                physical_polygon.material_u,
                                physical_polygon.material_v
                            );

                            let polygon_index = self.polygon_set.len();

                            self.polygon_set.push(physical_polygon.polygon);

                            super::Surface {
                                material_id: super::MaterialId::from_index(physical_polygon.material_index),
                                polygon_id: super::PolygonId::from_index(polygon_index),
                                u: physical_polygon.material_u,
                                v: physical_polygon.material_v,
                                is_transparent: physical_polygon.is_transparent,
                                is_sky: physical_polygon.is_sky,
                                u_min,
                                u_max,
                                v_min,
                                v_max,
                            }
                        })
                    );

                    portals.extend(face
                        .portal_polygons
                        .into_iter()
                        .map(|portal_polygon| super::Portal {

                            dst_volume_id: super::VolumeId::from_index(
                                portal_polygon.dst_volume_index + volume_index_offset
                            ),

                            polygon_id: super::PolygonId::from_index(
                                portal_polygon.polygon_set_index + portal_index_offset
                            ),

                            is_facing_front: portal_polygon.is_front,
                        })
                    );
                }

                super::Volume { portals, surfaces, bound_box }
            })
        );

        self.bsp_models.push(super::BspModel { bound_box: ctx.bound_box, bsp });

        self.bsp_models.len() - 1
    }

    /// Finish compilation
    pub fn finish(self, world_model_index: usize) -> super::Map {
        super::Map {
            bsp_models: self.bsp_models,
            material_name_set: self.material_name_set,
            polygon_set: self.polygon_set,
            volume_set: self.volume_set,
            dynamic_models: self.dynamic_models,
            world_model_id: super::BspModelId::from_index(world_model_index),
        }
    }
}

/// World building error
#[derive(Debug)]
pub enum Error {
    /// No worldspawn entity
    NoWorldspawn,
}

/// Build WBSP from map
pub fn compile(map: &map::Map) -> Result<super::Map, Error> {
    let mut context = CompileContext {
        bsp_models: Vec::new(),
        material_name_set: Vec::new(),
        material_name_table: HashMap::new(),
        polygon_set: Vec::new(),
        volume_set: Vec::new(),
        dynamic_models: Vec::new(),
    };

    let mut worldspawn_entity_index_opt: Option<usize> = None;

    // Get all 'origin' properties
    let map_origins = map.get_all_origins();

    'entity_compilation_loop: for entity in map.entities.iter() {
        // Do not compile entities without any geometry
        if entity.brushes.is_empty() {
            continue 'entity_compilation_loop;
        }

        let classname = entity.properties.get("classname")
            .map(|name| name.as_str())
            .unwrap_or("");

        let is_worldspawn = classname == "worldspawn";

        let mut model_compile_context = BspModelCompileContext {
            bound_box: geom::BoundBox::zero(),
            portal_polygons: Vec::new(),
            split_infos: Vec::new(),
            volume_bsp: None,
            volumes: Vec::new(),
        };

        let physical_polygons = context.build_entity_physical_polygons(entity);

        model_compile_context.start_build_volumes(physical_polygons);
        model_compile_context.start_resolve_portals();

        // Invisible removal pass is actual only for external volumes
        if is_worldspawn {
            model_compile_context.start_remove_invisible(map_origins.clone());
        } else {
            // TODO (?): Remove non-external (e.g. unreachable) volumes in non-worldspawn entities
        }

        let index = context.add_model(model_compile_context);

        if is_worldspawn {
            worldspawn_entity_index_opt = Some(index);
        }

        let mut origin: Option<Vec3f> = None;
        let mut angle: Option<f32> = None;

        'parse_origin: {
            if let Some(origin_str) = entity.properties.get("origin") {
                let vs = origin_str.split_whitespace()
                    .map(|v| v.parse::<f32>()).collect::<Result<Vec<_>, _>>().ok();
    
                let Some(vs) = vs else {
                    break 'parse_origin;
                };

                let Some([x, y, z]) = vs.get(..) else {
                    break 'parse_origin;
                };

                origin = Some(Vec3f::new(*x, *y, *z));
            }
        }

        if let Some(angle_str) = entity.properties.get("angle") {
            angle = angle_str.parse::<f32>().ok();
        }

        if origin.is_some() || angle.is_some() {
            let origin = origin.unwrap_or(Vec3f::zero());
            let angle = angle.unwrap_or(0.0);

            _ = context.add_dynamic_model(origin, angle, index);
        }
    }

    let Some(worldspawn_entity_index) = worldspawn_entity_index_opt else {
        return Err(Error::NoWorldspawn);
    };

    Ok(context.finish(worldspawn_entity_index))
}

// builder.rs
