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

use crate::{brush::Brush, geom, math::Vec3f};
use super::Location;

/// Q1 .map brush face binary representation
#[derive(Clone, Debug)]
pub struct MapBrushFace {
    /// First brush point
    pub p0: Vec3f,

    /// Second brush point
    pub p1: Vec3f,

    /// Third brush point
    pub p2: Vec3f,

    /// Brush texture name
    pub texture_name: String,

    /// Texture offset by X (in pixels)
    pub texture_offset_x: f32,

    /// Texture offset by Y (in pixels)
    pub texture_offset_y: f32,

    /// Texture rotation (in degrees)
    pub texture_rotation: f32,

    /// Texture scale by X (in texels per unit (?))
    pub texture_scale_x: f32,

    /// Texture scale by Y (in texels per unit (?))
    pub texture_scale_y: f32,
}

/// Map brush
#[derive(Clone, Debug)]
pub struct MapBrush {
    /// Face set
    pub faces: Vec<MapBrushFace>,
}

/// Quake map entity
#[derive(Clone, Debug)]
pub struct MapEntity {
    /// Entity properties
    pub properties: HashMap<String, String>,

    /// Entity brush set
    pub brushes: Vec<MapBrush>,
}

/// Map (entity collection, actually)
#[derive(Clone, Debug)]
pub struct Map {
    /// All map entities
    pub entities: Vec<MapEntity>,
}

/// Q1 map parsing error
#[derive(Debug, PartialEq, Eq)]
pub enum MapParseError<'t> {
    /// Expected one more token
    NextTokenExpected,

    /// Floating-point number parsing error
    FloatParsingError {
        /// Token floating point number parsed from
        token: &'t str,

        /// Exact error occured during parsing process
        error: std::num::ParseFloatError
    },

    /// Invalid property tokens
    InvalidProperty {
        /// Potential key token
        key: &'t str,

        /// Potential value tokens
        value: &'t str
    },

    /// Unexpected token
    UnexpectedToken {
        /// Actual token
        actual: &'t str,

        /// Expected token
        expected: &'t str,
    },
}

impl Map {
    /// Parse map from string
    pub fn parse<'t>(str: &'t str) -> Result<Map, MapParseError<'t>> {

        /// Parse token from string start
        fn parse_token<'t>(mut str_rest: &'t str) -> Result<(&'t str, &'t str), MapParseError<'t>> {
            // token skipping loop
            'comment_skip_loop: loop {
                str_rest = str_rest.trim_start();
    
                if str_rest.starts_with("//") {
                    let mut off = 0;
    
                    let mut rest_iter = str_rest.chars();
    
                    'comment_skip: while let Some(ch) = rest_iter.next() {
                        if ch == '\n' {
                            break 'comment_skip;
                        }
                        off += ch.len_utf8();
                    }
    
                    str_rest = str_rest[off..].trim_start();
                } else {
                    break 'comment_skip_loop;
                }
            }

            // parse string token
            if str_rest.starts_with('\"') {
                let mut off = 1;

                let mut rest_iter = str_rest[1..].chars();

                while let Some(ch) = rest_iter.next() {
                    if ch == '\"' {
                        let result = &str_rest[0..off + 1];
                        str_rest = &str_rest[off + 1..];
                        return Ok((result, str_rest));
                    }
                    off += ch.len_utf8();
                }

                let result = &str_rest[1..];
                str_rest = "";

                return Ok((result, str_rest));
            }

            if str_rest.is_empty() {
                return Err(MapParseError::NextTokenExpected);
            }

            if let Some((first, last)) = str_rest.split_once(char::is_whitespace) {
                str_rest = last;
                return Ok((first, str_rest));
            } else {
                let result = str_rest;
                str_rest = "";
                return Ok((result, str_rest));
            }
        }

        /// Parse **any** next token
        fn parse_next_token<'t, 'l>(tl: &'l [&'t str]) -> Result<(&'t str, &'l [&'t str]), MapParseError<'t>> {
            if let Some((tok, rest)) = tl.split_first() {
                Ok((*tok, rest))
            } else {
                Err(MapParseError::NextTokenExpected)
            }
        }

        fn parse_literal<'t, 'l>(tl: &'l [&'t str], lit: &'t str) -> Result<((), &'l [&'t str]), MapParseError<'t>> {
            let (tok, tl) = parse_next_token(tl)?;

            if tok == lit {
                Ok(((), tl))
            } else {
                Err(MapParseError::UnexpectedToken {
                    actual: tok,
                    expected: lit,
                })
            }
        }

        fn parse_float<'t, 'l>(tl: &'l [&'t str]) -> Result<(f32, &'l [&'t str]), MapParseError<'t>> {
            let (token, tl) = parse_next_token(tl)?;

            let val = token
                .parse::<f32>()
                .map_err(|error| MapParseError::FloatParsingError { token, error })?;

            return Ok((val, tl));
        }

        fn parse_vector<'t, 'l>(tl: &'l [&'t str]) -> Result<(Vec3f, &'l [&'t str]), MapParseError<'t>> {
            let (_, tl) = parse_literal(tl, "(")?;
            let (x, tl) = parse_float(tl)?;
            let (y, tl) = parse_float(tl)?;
            let (z, tl) = parse_float(tl)?;
            let (_, tl) = parse_literal(tl, ")")?;

            Ok((Vec3f::new(x, y, z), tl))
        }

        fn parse_brush_face<'t, 'l>(mut tl: &'l [&'t str]) -> Result<(MapBrushFace, &'l [&'t str]), MapParseError<'t>> {
            let p0;
            let p1;
            let p2;
            let texture_name;
            let texture_offset_x;
            let texture_offset_y;
            let texture_rotation;
            let texture_scale_x;
            let texture_scale_y;

            // parse plane basis vectors
            (p0, tl) = parse_vector(tl)?;
            (p1, tl) = parse_vector(tl)?;
            (p2, tl) = parse_vector(tl)?;

            // plane texture name
            (texture_name, tl) = parse_next_token(tl)?;
            let texture_name = texture_name.to_string();

            // 
            (texture_offset_x, tl) = parse_float(tl)?;
            (texture_offset_y, tl) = parse_float(tl)?;
            (texture_rotation, tl) = parse_float(tl)?;
            (texture_scale_x, tl) = parse_float(tl)?;
            (texture_scale_y, tl) = parse_float(tl)?;

            Ok((
                MapBrushFace {
                    p0,
                    p1,
                    p2,
                    texture_name,
                    texture_offset_x,
                    texture_offset_y,
                    texture_rotation,
                    texture_scale_x,
                    texture_scale_y
                },
                tl
            ))
        }

        fn parse_brush<'t, 'l>(mut tl: &'l [&'t str]) -> Result<(MapBrush, &'l [&'t str]), MapParseError<'t>> {
            (_, tl) = parse_literal(tl, "{")?;

            let mut faces = Vec::new();

            while let Ok((face, new_tl)) = parse_brush_face(tl) {
                tl = new_tl;
                faces.push(face);
            }

            (_, tl) = parse_literal(tl, "}")?;

            Ok((MapBrush { faces }, tl))
        }

        fn parse_property<'t, 'l>(tl: &'l [&'t str]) -> Result<((String, String), &'l [&'t str]), MapParseError<'t>> {
            let (key, tl) = parse_next_token(tl)?;
            let (value, tl) = parse_next_token(tl)?;

            if true
                && key.starts_with('\"')
                && key.ends_with('\"')
                && value.starts_with('\"')
                && value.ends_with('\"')
            {
                Ok(((key.trim_matches('\"').to_string(), value.trim_matches('\"').to_string()), tl))
            } else {
                Err(MapParseError::InvalidProperty { key, value })
            }
        }

        let tokens = {
            let mut str_rest = str;
            let mut tok_list = Vec::new();

            'parsing_loop: loop {
                match parse_token(str_rest) {
                    Ok((tok, new_str_rest)) => {
                        tok_list.push(tok);
                        str_rest = new_str_rest;
                    }
                    Err(MapParseError::NextTokenExpected) => break 'parsing_loop,
                    Err(err) => return Err(err)
                };
            }

            tok_list
        };

        let mut entities = Vec::<MapEntity>::new();
        let mut tl = tokens.as_slice();

        'parsing_loop: loop {
            match parse_literal(tl, "{") {
                Ok((_, new_tl)) => tl = new_tl,
                Err(MapParseError::NextTokenExpected) => break 'parsing_loop,
                Err(parsing_error) => return Err(parsing_error),
            }

            let mut properties = HashMap::<String, String>::new();
            let mut brushes = Vec::<MapBrush>::new();

            'entity_contents: loop {
                if let Ok((brush, next_tl)) = parse_brush(tl) {
                    tl = next_tl;
                    brushes.push(brush);
                } else if let Ok(((key, value), next_tl)) = parse_property(tl) {
                    tl = next_tl;
                    _ = properties.insert(key, value);
                } else {
                    break 'entity_contents;
                }
            }

            (_, tl) = parse_literal(tl, "}")?;

            entities.push(MapEntity { brushes, properties });
        }

        return Ok(Map { entities });
    }

    /// Find entity by property
    pub fn find_entity(&self, key: &str, value: Option<&str>) -> Option<&MapEntity> {
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
}

/// Polygon that should be drawn
pub struct PhysicalPolygon {
    /// Actual polygon
    pub polygon: geom::Polygon,

    // Material info?
}

/// Reference to another volume
pub struct PortalPolygon {
    /// Index of actual polygon
    pub polygon_set_index: usize,

    /// True denoted polygon polygon should be used 'as-is', False - as check results should be reversed
    pub is_front: bool,

    /// Destination volume
    pub dst_volume_index: usize,
}

/// Hull polygon
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
#[derive(Copy, Clone)]
pub struct SplitReference {
    /// split operation index
    pub split_id: SplitId,

    /// true if portal belongs to front part of split, false - if to back
    pub edge_is_front: bool,
}

/// Hull volume
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

impl VolumeBsp {
    /// Traverse volume BSP
    pub fn traverse(&self, point: Vec3f) -> Option<usize> {
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
                    geom::PolygonRelation::OnPlane => {
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
            eprintln!("Something strange happend...");
            None
        } else {
            Some(geom::Polygon {
                points: geom::sort_points_by_angle(intersection_points, plane.normal),
                plane
            })
        }


        // Build intersection polygon
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
    pub fn split(
        self,
        plane: geom::Plane,
        incident_front_polygons: Vec<PhysicalPolygon>,
        incident_back_polygons: Vec<PhysicalPolygon>,
        split_id: SplitId,
    ) -> Option<(HullVolume, HullVolume, geom::Polygon)> {

        // intersection polygon MUST exist

        let mut intersection_polygon = self.get_intersection_polygon(plane)?;

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
                                front_physical_polygons.push(PhysicalPolygon { polygon: front });
                                back_physical_polygons.push(PhysicalPolygon { polygon: back });
                            }
                            geom::PolygonSplitResult::OnPlane => {
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
                geom::PolygonSplitResult::OnPlane => {
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

        Some((front_hull_volume, back_hull_volume, split_polygon))
    }
}

pub struct SplitInfo {
    /// Polygon used to split
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

/// Map builder structure
pub struct Builder {
    /// Final volumes
    pub volumes: Vec<HullVolume>,

    /// Split infos
    pub split_infos: Vec<SplitInfo>,

    /// Volume
    pub volume_bsp: Option<Box<VolumeBsp>>,

    /// Set of portal polygons (They should be shared to optimize total memory consume)
    pub portal_polygons: Vec<geom::Polygon>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
            split_infos: Vec::new(),
            portal_polygons: Vec::new(),
            volume_bsp: None,
        }
    }

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

        // if physical_polygons.len() == 1 && (physical_polygons[0].polygon.plane.distance + 715.0).abs() < 2.0 {
        //     eprintln!("Nachalosz");
        // }

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
                    geom::PolygonRelation::OnPlane => current_on += 1,
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
    
        // Get current split identifier
        let split_id = self.get_next_split_id();

        // Split MUST be possible, because splitter plane face belongs to hull volume (by definition)
        // If it's not the case, something went totally wrong.
        let (front_volume, back_volume, split_polygon) = volume
            .split(
                splitter_plane,
                front_incident_polygons,
                back_incident_polygons,
                split_id
            )
            .unwrap();

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

    /// Start volume building pass
    pub fn start_build_volumes(&mut self, brushes: &[Brush]) {
        // calculate polygon set hull
        let hull = brushes
            .iter()
            .fold(
                geom::BoundBox::zero(),
                |total, brush| total.total(&brush.bound_box)
            )
            .extend(Vec3f::new(200.0, 200.0, 200.0));

        // calculate hull volume
        let hull_volume = HullVolume::from_bound_box(hull);

        // get brush polygons
        let polygons = brushes
            .iter()
            .flat_map(|v| v.polygons.iter())
            .map(|polygon| PhysicalPolygon { polygon: polygon.clone() })
            .collect::<Vec<_>>();

        // Build volumes
        let volume_bsp = self.build_volume(hull_volume, polygons);

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
            .map(|physical_polygon| CutInfo {
                planes: physical_polygon.polygon.iter_edge_planes().collect(),
                bound_box: physical_polygon.polygon.build_bound_box(),
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
                            geom::PolygonSplitResult::OnPlane => {
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

        for (id, task_contents) in &resolve_tasks {
            let info = &self.split_infos[id.0.get() as usize - 1];
            let split_polygon = &info.split_polygon;
        
            std::hint::black_box(&split_polygon);
        
            let plane = split_polygon.plane;
        
            for (volume_id, face_id) in task_contents.iter().copied() {
                let face_polygon = &self.volumes[volume_id].faces[face_id].polygon;
                let face_plane = face_polygon.plane;
                
                std::hint::black_box(&face_polygon);
                if face_plane != plane && face_plane.negate_direction() != plane {
                    panic!("Task face coplanarity check failed!\n First plane: {:?}\nSecond plane: {:?}",
                        plane, face_plane
                    );
                }
            }
        }

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
                            geom::PolygonSplitResult::OnPlane => {
                                continue 'back_polygon_loop;
                                // Something strange happened...
                                // eprintln!("Something strange happened because I don't want to use planar geometry...");
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

        // todo!("Builder.start_resolve_portals function")
    }

    /// Get indices of invisible volumes
    fn get_invisible_volume_idx(&mut self) -> Vec<usize> {
        let mut invisible_index_set = self.volumes
            .iter()
            .enumerate()
            .filter_map(|(index, volume)| {
                let cond = volume
                    .faces
                    .iter()
                    .any(|face| face.is_external);

                if cond {
                    Some(index)
                } else {
                    None
                }
            })
            .map(|index| (index, true))
            .collect::<HashMap<usize, bool>>();

        'external_propagation_loop: loop {
            let mut new_external_set = BTreeSet::new();

            'volume_loop: for (volume_index, is_new) in invisible_index_set.iter() {
                if !is_new {
                    continue 'volume_loop;
                }

                let volume = self.volumes.get(*volume_index).unwrap();

                for face in &volume.faces {
                    for portal in &face.portal_polygons {
                        if !invisible_index_set.contains_key(&portal.dst_volume_index) {
                            new_external_set.insert(portal.dst_volume_index);
                        }
                    }
                }
            }

            if new_external_set.is_empty() {
                break 'external_propagation_loop;
            }

            for is_new in invisible_index_set.values_mut() {
                *is_new = false;
            }

            invisible_index_set.extend(new_external_set
                .into_iter()
                .map(|index| (index, true))
            );
        }

        let mut invisible_index_vec = invisible_index_set
            .into_iter()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        // remove unreachable volumes
        for (volume_index, volume) in self.volumes.iter().enumerate() {
            let portal_count = volume.faces
                .iter()
                .map(|f| f.portal_polygons.len())
                .sum::<usize>();
            if portal_count == 0 {
                invisible_index_vec.push(volume_index);
            }
        }

        invisible_index_vec.sort();

        invisible_index_vec
    }

    /// Remove invisible volumes
    /// 
    /// # TODO
    /// Filter portal polygon set
    pub fn start_remove_invisible(&mut self) {
        let mut invisible_volume_idx = self.get_invisible_volume_idx();
        invisible_volume_idx.reverse();

        let mut index_map = Vec::<Option<usize>>::with_capacity(self.volumes.len());

        let mut delta = 0;

        for i in 0..self.volumes.len() {
            if let Some(removed_index) = invisible_volume_idx.last().copied() {
                if i < removed_index {
                    index_map.push(Some(i - delta));
                } else {
                    index_map.push(None);
                    delta += 1;
                    invisible_volume_idx.pop();
                }
            } else {
                index_map.push(Some(i - delta));
            }
        }

        let old_volumes = std::mem::replace(&mut self.volumes, Vec::new());

        for (volume_index, mut volume) in old_volumes.into_iter().enumerate() {
            if index_map.get(volume_index).unwrap().is_none() {
                // println!("Removed by {}", volume_index);
                continue;
            }

            for face in &mut volume.faces {
                let old_portal_polygons = std::mem::replace(&mut face.portal_polygons, Vec::new());

                'portal_loop: for mut portal_polygon in old_portal_polygons {
                    let Some(new_volume_index) = index_map[portal_polygon.dst_volume_index] else {
                        eprintln!("Reachable face somehow have not map...");
                        continue 'portal_loop;
                    };

                    portal_polygon.dst_volume_index = new_volume_index;

                    face.portal_polygons.push(portal_polygon);
                }
            }

            self.volumes.push(volume);
        }

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

        let old_volume_bsp = std::mem::replace(&mut self.volume_bsp, None);

        if let Some(bsp) = old_volume_bsp {
            self.volume_bsp = map_volume_bsp(*bsp, &index_map);
        }
    }
}

/// Builder
pub fn build(brushes: Vec<Brush>) -> Location {
    let mut builder = Builder::new();

    // Build volumes & volume BSP
    builder.start_build_volumes(&brushes);

    // Resolve volume portals
    builder.start_resolve_portals();

    // Remove invisible surfaces
    builder.start_remove_invisible();

    unimplemented!()
}

