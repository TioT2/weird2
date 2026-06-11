//! Standard geometry primitive implementation module

use std::{cell::Cell, collections::HashMap, rc::Rc};

use crate::math::{Mat3f, Vec2f, Vec3f, Vec4f};

/// Geometric epsilon (1 cm)
pub const GEOM_EPSILON: f32 = 0.01;

/// (Slow) Geometric equality function
pub fn f32_geom_equal(lhs: f32, rhs: f32) -> bool {
    let abs_diff = (lhs - rhs).abs();

    abs_diff <= GEOM_EPSILON
}

/// Vector equality
pub fn f32_vec_equal(lhs: &Vec3f, rhs: &Vec3f) -> bool {
    f32_geom_equal(lhs.x(), rhs.x()) && f32_geom_equal(lhs.y(), rhs.y()) && f32_geom_equal(lhs.z(), rhs.z())
}

/// Float-point comparison
pub fn f32_relative_equal(lhs: f32, rhs: f32) -> bool {
    if lhs == rhs {
        return true;
    }

    let diff = (lhs - rhs).abs();
    let norm = f32::min(lhs.abs() + rhs.abs(), f32::MAX);

    diff < f32::max(f32::MIN_POSITIVE, 128.0 * f32::EPSILON * norm)
}

/// Plane represetnation structure
/// 
/// # Plane equation
/// Standard plane equation is Ax + By + Cz + D = 0. In this case,
/// * A = point.x
/// * B = point.y
/// * C = point.z
/// * D = -distance
/// 
#[derive(Debug, Copy, Clone)]
pub struct Plane {
    /// Plane normal
    pub normal: Vec3f,

    /// Number to multiply normal to to get base point
    pub distance: f32,
}

impl std::ops::Mul<f32> for Plane {
    type Output = Self;

    fn mul(self, n: f32) -> Self {
        Self {
            normal: self.normal * n.into(),
            distance: self.distance * n,
        }
    }
}

impl std::ops::Div<f32> for Plane {
    type Output = Self;

    fn div(self, n: f32) -> Self {
        Self {
            normal: self.normal / n.into(),
            distance: self.distance / n,
        }
    }
}

/// Line in space
#[derive(Debug, Copy, Clone)]
pub struct Line {
    /// Line direction vector, assumed to be normalized
    pub direction: Vec3f,

    /// Line origin
    pub base: Vec3f,
}

impl Line {
    /// Build line from points
    pub fn from_points(first: Vec3f, second: Vec3f) -> Self {
        Self {
            direction: (second - first).normalized(),
            base: first,
        }
    }
}

/// Relation of plane and polygon
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PolygonRelation {
    /// Polygon located in front of plane
    Front,

    /// Polygon located back of plane
    Back,

    /// Polygon located completely on plane
    Coplanar,

    /// Polygon intersects with plane (splitted by plane)
    Intersects,
}

/// Relation of plane and point
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PointRelation {
    /// Point it located behind plane
    Back,

    /// Point is located on plane
    OnPlane,

    /// Point is located in front of plane
    Front,
}

impl PointRelation {
    /// Get opposite relation
    pub fn opposite(self) -> Self {
        match self {
            Self::Back => Self::Front,
            Self::OnPlane => Self::OnPlane,
            Self::Front => Self::Back,
        }
    }
}

/// Polygon by plane splitting result, resembles `PolygonRelation` by structure
pub enum PolygonSplitResult {
    /// Polygon located in front of plane, so it doesn't require splitting
    Front,

    /// Polygon located back of plane, so it doesn't require splitting
    Back,

    /// Polygon located back of plane, so it doesn't require splitting
    Coplanar,

    /// Polygon intersects with plane
    Intersects {
        /// Front part
        front: Polygon,

        /// Back part
        back: Polygon,
    },
}

impl PartialEq for Plane {
    fn eq(&self, other: &Self) -> bool {
        f32_geom_equal((self.normal % other.normal).length(), 0.0)
            && f32_vec_equal(
                &(self.normal * self.distance.into()),
                &(other.normal * other.distance.into()))
    }
}

impl Plane {
    /// Build plane for point triple (normal is calculated as if triple is CCW-oriented)
    pub fn from_points(p1: Vec3f, p2: Vec3f, p3: Vec3f) -> Self {
        let normal = Vec3f::cross(p3 - p2, p1 - p2).normalized();
        let distance = p2 ^ normal;

        Self { normal, distance }
    }

    /// Build plane from it's point and normal
    pub fn from_point_normal(point: Vec3f, mut normal: Vec3f) -> Self {
        normal = normal.normalized();
        Self { distance: point ^ normal, normal }
    }

    /// Intersect this plane with another one
    pub fn intersect_plane(&self, rhs: Plane) -> Line {
        let direction = (self.normal % rhs.normal).normalized();
        let dot = self.normal ^ rhs.normal;
        let base = (
            (self.normal * self.distance.into() + rhs.normal *  rhs.distance.into()) -
            (self.normal *  rhs.distance.into() + rhs.normal * self.distance.into()) * dot.into()
        ) / (1.0 - dot * dot).into();

        Line { base, direction }
    }

    /// Intersect one plane with two other planes
    pub fn intersect_two_planes(&self, othr: &Plane, thrd: &Plane) -> Option<Vec3f> {
        Mat3f::from_cols([self.normal, othr.normal, thrd.normal])
            .transposed()
            .inversed()
            .map(|m| m * Vec3f::new(self.distance, othr.distance, thrd.distance))
    }

    /// Get intersection line of plane and the polygon
    pub fn intersect_polygon(&self, polygon: &Polygon) -> Option<(Vec3f, Vec3f)> {
        let mut first: Option<Vec3f> = None;
        let mut second: Option<Vec3f> = None;

        // current point relation
        let mut prev_point = *polygon.points.last().unwrap();
        let mut prev_relation = self.get_point_relation(prev_point);

        for index in 0..polygon.points.len() {
            let curr_point = *polygon.points.get(index).unwrap();
            let curr_relation = self.get_point_relation(curr_point);

            if curr_relation == PointRelation::OnPlane {
                first = Some(curr_point);
                std::mem::swap(&mut first, &mut second);
            } else if prev_relation == PointRelation::Front && curr_relation == PointRelation::Back
                   || curr_relation == PointRelation::Front && prev_relation == PointRelation::Back
            {
                first = Some(self.intersect_line(Line::from_points(prev_point, curr_point)));

                std::mem::swap(&mut first, &mut second);
            }

            prev_point = curr_point;
            prev_relation = curr_relation;
        }

        Option::zip(first, second)
    }

    /// Make plane that contains equal point set, but has counter-directional normal
    pub fn negate_direction(self) -> Self {
        Self { normal: -self.normal, distance: -self.distance }
    }

    /// Get point at some signed distance from plane
    pub fn point_at(&self, dist: f32) -> Vec3f {
        self.normal * (dist + self.distance).into()
    }

    /// Project point to plane
    pub fn project(&self, point: Vec3f) -> Vec3f {
        point - self.normal * self.get_signed_distance(point).into()
    }

    /// Project point on plane along axis
    pub fn project_along(&self, point: Vec3f, axis: Vec3f) -> Vec3f {
        let x = (self.distance - (self.normal ^ point)) / (self.normal ^ axis);
        point + axis * x.into()
    }

    /// Get plane signed distance function
    pub fn get_signed_distance(&self, point: Vec3f) -> f32 {
        (point ^ self.normal) - self.distance
    }

    /// Get relation of point and plane
    pub fn get_point_relation(&self, point: Vec3f) -> PointRelation {
        let signed_distance = self.get_signed_distance(point);

        if f32_geom_equal(signed_distance, 0.0) {
            PointRelation::OnPlane
        } else if signed_distance > 0.0 {
            PointRelation::Front
        } else {
            PointRelation::Back
        }
    }

    /// Get relation of plane and polygon
    pub fn get_polygon_relation(&self, polygon: &Polygon) -> PolygonRelation {
        if *self == polygon.plane {
            return PolygonRelation::Coplanar;
        }

        let mut front_occured = false;
        let mut back_occured = false;

        for point in &polygon.points {
            match self.get_point_relation(*point) {
                PointRelation::Front => front_occured    = true,
                PointRelation::Back  => back_occured     = true,
                _ => {}
            }
        }

        match (front_occured, back_occured) {
            (false, false) => PolygonRelation::Coplanar,
            (false, true ) => PolygonRelation::Back,
            (true , false) => PolygonRelation::Front,
            (true , true ) => PolygonRelation::Intersects,
        }
    }

    // Get intersection of the plane and line
    pub fn intersect_line(&self, line: Line) -> Vec3f {
        line.base + line.direction * ((self.distance - (line.base ^ self.normal)) / (line.direction ^ self.normal)).into()
    }

    /// Split polygon by the plane
    pub fn split_polygon(&self, polygon: &Polygon) -> PolygonSplitResult {
        match self.get_polygon_relation(polygon) {
            PolygonRelation::Back => return PolygonSplitResult::Back,
            PolygonRelation::Front => return PolygonSplitResult::Front,
            PolygonRelation::Coplanar => return PolygonSplitResult::Coplanar,
            PolygonRelation::Intersects => {}
        };

        let (first_index, first_point, first_relation) = polygon.points
            .iter()
            .enumerate()
            .map(|(id, vt)| (id, *vt, self.get_point_relation(*vt)))
            .find(|(_, _, relation)| *relation != PointRelation::OnPlane)
            .unwrap();

        let mut prev_point = first_point;
        let mut prev_relation = first_relation;

        let mut first_is_front = first_relation == PointRelation::Front;
        let mut first_point_set = Vec::new();
        let mut second_point_set = Vec::new();

        let index_iter = (0..polygon.points.len())
            .map(|index| (index + first_index + 1) % polygon.points.len());
        for index in index_iter {
            // get current point
            let curr_point = *polygon.points.get(index).unwrap();
            let curr_relation = self.get_point_relation(curr_point);

            // add new point if this line's crossing the polygon.
            if curr_relation == PointRelation::OnPlane {
                first_point_set.push(curr_point);
                second_point_set.push(curr_point);

                first_is_front = !first_is_front;
                std::mem::swap(&mut first_point_set, &mut second_point_set);
            } else if prev_relation == curr_relation.opposite() {
                let intr = self.intersect_line(Line::from_points(prev_point, curr_point));
                first_point_set.push(intr);
                second_point_set.push(intr);

                first_is_front = !first_is_front;
                std::mem::swap(&mut first_point_set, &mut second_point_set);

                first_point_set.push(curr_point);
            } else {
                first_point_set.push(curr_point);
            }

            prev_point = curr_point;
            prev_relation = curr_relation;
        }

        if !first_is_front {
            std::mem::swap(&mut first_point_set, &mut second_point_set);
        }

        PolygonSplitResult::Intersects {
            front: Polygon { plane: polygon.plane, points: first_point_set  },
            back:  Polygon { plane: polygon.plane, points: second_point_set },
        }
    }
}

/// Polygon
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Polygon points
    pub points: Vec<Vec3f>,

    /// Plane
    pub plane: Plane,
}

/// Remove duplicates from point set
pub fn deduplicate_points(points: Vec<Vec3f>) -> Vec<Vec3f> {
    points
        .into_iter()
        .fold(Vec::new(), |mut prev, candidate| {
            for point in prev.iter() {
                if f32_vec_equal(&candidate, point) {
                    return prev;
                }
            }

            prev.push(candidate);
            prev
        })
}

/// Sort points by angle ???
pub fn sort_points_by_angle(mut points: Vec<Vec3f>, normal: Vec3f) -> Vec<Vec3f> {
    let center = points
        .iter()
        .copied()
        .fold(Vec3f::zero(), std::ops::Add::add) / (points.len() as f32).into();

    let mut sorted = vec![points.pop().unwrap()];

    while !points.is_empty() {
        let last = *sorted.last().unwrap() - center;

        let smallest_cotan_opt = points
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, p)| {
                let v = p - center;
                let cross_normal_dot = (last % v) ^ normal;

                (cross_normal_dot < 0.0).then(|| (index, (last ^ v) / cross_normal_dot))
            })
            .min_by(|l, r| f32::total_cmp(&l.1, &r.1));

        let Some((smallest_cotan_index, _)) = smallest_cotan_opt else {
            break;
        };
        sorted.push(points.swap_remove(smallest_cotan_index));
    }

    // fix point set orientation
    // TODO: Fix this sh*tcode
    if sorted.len() >= 3 {
        let point_normal = Vec3f::cross(
            (sorted[2] - sorted[1]).normalized(),
            (sorted[0] - sorted[1]).normalized(),
        ).normalized();
    
        // fix point orientation
        if (point_normal ^ normal) < 0.0 {
            sorted.reverse();
        }
    }

    sorted
}

/// Clip generic polygon by some linear vertex norm function
/// # Note
/// `cmp` must be ordering function, `norm` must be linear in respect of `V` operators
pub fn clip_polygon<V>(
    points: &mut Vec<V>,
    temp: &mut Vec<V>,
    value: f32,
    cmp: impl Fn(f32, f32) -> bool,
    norm: impl Fn(V) -> f32,
) -> bool
where
    V: Copy
        + std::ops::Add<V, Output = V>
        + std::ops::Sub<V, Output = V>
        + std::ops::Mul<V, Output = V>
        + From<f32>
{
    temp.clear();
    for index in 0..points.len() {
        let curr = points[index];
        let next = points[(index + 1) % points.len()];

        if cmp(norm(curr), value) {
            temp.push(curr);

            if cmp(value, norm(next)) {
                let t = (value - norm(curr)) / (norm(next) - norm(curr));
                temp.push((next - curr) * V::from(t) + curr);
           }
        } else if cmp(norm(next), value) {
            let t = (value - norm(curr)) / (norm(next) - norm(curr));
            temp.push((next - curr) * V::from(t) + curr);
        }
    }
    std::mem::swap(points, temp);

    points.len() >= 3
}

impl Polygon {
    /// Negate polygon orientation (e.g. normal is now -normal, point order is reversed to fit normal.)
    pub fn negate_orientation(&mut self) {
        self.plane = self.plane.negate_direction();
        self.points.reverse();
    }

    /// Negate orientation of the polygon
    pub fn negated_orientation(mut self) -> Self {
        self.negate_orientation();
        self
    }

    /// Build polygon bounding box
    pub fn build_bound_box(&self) -> BoundBox {
        BoundBox::for_points(self.points.iter().copied())
    }

    /// Iterator on planes that are parallel to polygon normal and contain corresponding edges.
    /// Plane normals directed out of polygon.
    pub fn iter_edge_planes<'t>(&'t self) -> impl Iterator<Item = Plane> + 't {
        (0..self.points.len())
            .map(|index| {
                let first = *self.points.get(index).unwrap();
                let second = *self.points.get((index + 1) % self.points.len()).unwrap();

                let normal = Vec3f::cross(
                    second - first,
                    self.plane.normal
                ).normalized();

                Plane::from_point_normal(first, normal)
            })
    }

    /// From convex point set, normal is calculated by assuming polygon is counter-clockwise
    pub fn from_ccw(points: Vec<Vec3f>) -> Self {
        // yep, that's all
        assert!(points.len() >= 3);
        Self {
            plane: Plane::from_points(points[0], points[1], points[2]),
            points,
        }
    }

    /// Build polygon from clockwise-going points
    pub fn from_cw(mut points: Vec<Vec3f>) -> Self {
        // yep, that's very ineffective solution, but I don't care (in this case)
        points.reverse();
        Self::from_ccw(points)
    }
}

// Implement common bound volume functions
macro_rules! impl_bb {
    ($BoundBox: ident, $Vec: ty, $Pt: ty, $pt2v: expr) => {
        impl $BoundBox {
            /// Construct new boundbox
            pub fn new(p0: $Pt, p1: $Pt) -> Self {
                let v0 = $pt2v(p0);
                let v1 = $pt2v(p1);

                Self {
                    min: <$Vec>::zip(v0, v1, f32::min),
                    max: <$Vec>::zip(v0, v1, f32::max),
                }
            }

            /// Create bound box containing single point
            pub fn unit(p: $Pt) -> Self {
                let v = $pt2v(p);
                Self { min: v, max: v }
            }

            /// Construct bound box without volume (and location)
            pub fn empty() -> Self {
                Self {
                    min: <$Vec>::broadcast(f32::MAX),
                    max: <$Vec>::broadcast(f32::MIN),
                }
            }

            /// Build bounding volume for points set
            pub fn for_points(points: impl Iterator<Item = $Pt>) -> Self {
                let mut min = <$Vec>::broadcast(f32::MAX);
                let mut max = <$Vec>::broadcast(f32::MIN);

                for pt in points {
                    let v = $pt2v(pt);

                    min = min.zip(v, f32::min);
                    max = max.zip(v, f32::max);
                }

                Self { min, max }
            }

            /// Get clipping octahedron intersection
            pub fn intersection(&self, othr: &Self) -> Option<Self> {
                let max = <$Vec>::zip(self.max, othr.max, f32::min);
                let min = <$Vec>::zip(self.min, othr.min, f32::max);

                <$Vec>::zip(min, max, |l, r| l <= r)
                    .reduce(|x, y| x && y)
                    .then_some(Self { min, max })
            }

            /// Calculate union of bound volumes
            pub fn union(&self, othr: &Self) -> Self {
                Self {
                    min: <$Vec>::zip(self.min, othr.min, f32::min),
                    max: <$Vec>::zip(self.max, othr.max, f32::max),
                }
            }

            /// Check if boundbox intersection isn't empty
            pub fn is_intersecting(&self, another: &$BoundBox) -> bool {
                true
                    && <$Vec>::zip(self.max, another.min, |l, r| l >= r).reduce(|x, y| x && y)
                    && <$Vec>::zip(self.min, another.max, |l, r| l <= r).reduce(|x, y| x && y)
            }

            /// Check if boundbox contains point
            pub fn contains_point(&self, point: &$Pt) -> bool {
                let v = $pt2v(*point);
                true
                    && <$Vec>::zip(self.max, v, |l, r| l >= r).reduce(|x, y| x && y)
                    && <$Vec>::zip(self.min, v, |l, r| l <= r).reduce(|x, y| x && y)
            }
        }
    };
}

/// Bounding box
#[derive(Copy, Clone)]
pub struct BoundBox {
    /// minimal vector
    min: Vec3f,

    /// maximal vector
    max: Vec3f,
}

impl_bb!(BoundBox, Vec3f, Vec3f, std::convert::identity);

impl BoundBox {
    /// Get conservative bounding box of **any** rotation
    pub fn rotate(&self) -> Self {
        let center = (self.min + self.max) / 2.0.into();
        let extent = Vec3f::broadcast((self.min - self.max).length() / 2.0);

        Self {
            min: center - extent,
            max: center + extent,
        }
    }

    /// Get boundbox maximal fitting coordinates
    pub fn max(self) -> Vec3f {
        self.max
    }

    /// Get boundbox minimal fitting coordinates
    pub fn min(self) -> Vec3f {
        self.min
    }

    /// Get boundbox dimensions
    pub fn size(self) -> Vec3f {
        self.max - self.min
    }

    /// Translate boundbox to some extent
    pub fn translate(self, distance: Vec3f) -> Self {
        Self {
            min: self.min + distance,
            max: self.max + distance,
        }
    }

    /// Scale boundbox
    pub fn scale(self, scale: Vec3f) -> Self {
        Self {
            min: self.min * scale,
            max: self.max * scale,
        }
    }

    /// Extend boundbox by some (positive) vector.
    /// In case if delta is negative,
    pub fn extend(self, mut delta: Vec3f) -> Self {
        delta = delta.map(|x| x.max(0.0));

        Self {
            min: self.min - delta,
            max: self.max + delta,
        }
    }
}

/// 2D bounding octagon
#[derive(Copy, Clone, Debug)]
pub struct BoundOct {
    /// Octagonal minimum. `z` and `w` fields are implemented as `y - x` and `y + x` respectively.
    pub min: Vec4f,

    /// Octagonal maximum
    pub max: Vec4f,
}

impl_bb!(BoundOct, Vec4f, Vec2f, Self::vec2to4);

impl BoundOct {
    fn vec2to4(v: Vec2f) -> Vec4f {
        Vec4f::new(v.x(), v.y(), v.y() - v.x(), v.y() + v.x())
    }

    /// Calculate conservative clipping octagon of clipping rectangle
    pub fn from_clip_rect(clip_rect: BoundRect) -> Self {
        Self {
            max: Vec4f::new(
                clip_rect.max.x(),
                clip_rect.max.y(),
                clip_rect.max.y() - clip_rect.min.x(),
                clip_rect.max.y() + clip_rect.max.x(),
            ),
            min: Vec4f::new(
                clip_rect.min.x(),
                clip_rect.min.y(),
                clip_rect.min.y() - clip_rect.max.x(),
                clip_rect.min.y() + clip_rect.min.x(),
            ),
        }
    }

    /// Extend clipping octahedron
    pub fn extend(&self, x: f32, y: f32, y_s_x: f32, y_a_x: f32) -> Self {
        let v = Vec4f::new(x, y, y_s_x, y_a_x);
        Self {
            min: self.min - v,
            max: self.max + v,
        }
    }
}

/// 2D boundbox
#[derive(Copy, Clone, Debug)]
pub struct BoundRect {
    /// Rectangle minimum
    pub min: Vec2f,

    /// Rectangle maximum
    pub max: Vec2f,
}

impl_bb!(BoundRect, Vec2f, Vec2f, std::convert::identity);

impl BoundRect {
    /// Extend boundbox to contain the point
    pub fn extend_to_contain(self, pt: Vec2f) -> Self {
        Self {
            min: Vec2f::zip(self.min, pt, f32::min),
            max: Vec2f::zip(self.max, pt, f32::max),
        }
    }

    /// Extend clipping rectangle for some vector
    pub fn extend(self, v: Vec2f) -> Self {
        Self {
            min: self.min - v,
            max: self.max + v,
        }
    }
}

/// Total-ordered f32
#[derive(Copy, Clone)]
struct TotalF32(f32);

impl TotalF32 {
    /// Compare total-ordered f32
    pub fn cmp(self, othr: TotalF32) -> std::cmp::Ordering {
        f32::total_cmp(&self.0, &othr.0)
    }
}

impl std::cmp::PartialEq for TotalF32 {
    fn eq(&self, othr: &Self) -> bool {
        self.cmp(othr).is_eq()
    }
}

impl std::cmp::Eq for TotalF32 {}

impl std::cmp::PartialOrd for TotalF32 {
    fn partial_cmp(&self, othr: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(othr))
    }
}

impl std::cmp::Ord for TotalF32 {
    fn cmp(&self, othr: &Self) -> std::cmp::Ordering {
        TotalF32::cmp(*self, *othr)
    }
}

/// Convex hull construction error
pub enum ConvexHullError {
    /// Initial set does not form any simplex, e.g. there's no 3d convex hull for this set
    NoInitSimplex,
}

/// Jarvis convex hull edge helper
struct JEdge {
    /// First vertex index
    v0: usize,

    /// Second vertex index
    v1: usize,

    /// First triangle vertex
    _t_init: usize,

    /// Second triangle vertex
    t_next: Cell<Option<usize>>,
}

impl JEdge {
    // /// Get neighbour of some index
    // fn neighbour_of(&self, ind: usize) -> Option<usize> {
    //     if self.t_init != ind {
    //         Some(self.t_init)
    //     } else if let Some(i) = self.t_next.get() && i != ind {
    //         Some(i)
    //     } else {
    //         None
    //     }
    // }

    /// Get vertex indices considering edge swap flag
    fn indices(&self, do_swap: bool) -> (usize, usize) {
        if do_swap {
            (self.v1, self.v0)
        } else {
            (self.v0, self.v1)
        }
    }
}

/// Jarvis convex hull triangle
struct JTriangle {
    /// Array of triangle edges. Contains edge reference and swap flag.
    edges: [(Rc<JEdge>, bool); 3],

    /// Triangle plane
    plane: Plane,
}

/// Jarvis convex hull builder
struct JBuilder<'t> {

    /// Set of points we're building convex hull for
    pts: &'t [Vec3f],

    /// Array of hull triangles
    tris: Vec<JTriangle>,

    /// Map for searching for edges
    edges: HashMap<(usize, usize), Rc<JEdge>>,

    /// Point set average, used for plane orientation validation
    center: Vec3f,

    /// Hull building stack. Contains edge point pair, excluded point, and target normal.
    edge_stack: Vec<(usize, usize, usize, Vec3f)>,
}

impl<'t> JBuilder<'t> {
    /// Create new jarvis hull builder
    pub fn new(pts: &'t [Vec3f]) -> Self {
        Self {
            pts,
            center: pts.iter().copied().sum::<Vec3f>() / (pts.len() as f32).into(),
            tris: Vec::new(),
            edges: HashMap::new(),
            edge_stack: Vec::new(),
        }
    }

    /// Find edge by indices for triangle by index
    fn find_edge(&mut self, tri: usize, mut v0: usize, mut v1: usize) -> (Rc<JEdge>, bool) {
        let do_swap = v0 > v1;
        if do_swap {
            std::mem::swap(&mut v0, &mut v1);
        }

        let edge = match self.edges.entry((v0, v1)) {
            std::collections::hash_map::Entry::Occupied(occ) => {
                let edge = occ.get().clone();
                edge.t_next.set(Some(tri));
                edge
            },
            std::collections::hash_map::Entry::Vacant(vac) => {
                let edge = Rc::new(JEdge {
                    v0,
                    v1,
                    _t_init: tri,
                    t_next: Cell::new(None),
                });
                vac.insert(edge).clone()
            }
        };

        (edge, do_swap)
    }

    /// Find point maximizing certain floating-point parameter
    fn find_max_point(&self, mut cond: impl FnMut(usize, Vec3f) -> Option<f32>) -> usize {
        self.pts.iter().copied().enumerate()
            .max_by_key(|(i, p)| cond(*i, *p).map(TotalF32))
            .unwrap()
            .0
    }

    /// Insert triangle in triangle stack
    fn insert_triangle(&mut self, v0: usize, v1: usize, v2: usize) {
        let plane = Plane::from_points(
            self.pts[v0],
            self.pts[v1],
            self.pts[v2]
        );

        // PARANOID
        if plane.normal.dot(self.pts[v0] - self.center).is_sign_negative() {
            panic!("Invalid triangle orientation");
        }

        let ti = self.tris.len();
        let edges = [
            self.find_edge(ti, v0, v1),
            self.find_edge(ti, v1, v2),
            self.find_edge(ti, v2, v0),
        ];

        let mut push_edge = |i: usize, v_exc: usize| {
            if edges[i].0.t_next.get().is_none() {
                let (e0, e1) = edges[i].0.indices(edges[i].1);
                self.edge_stack.push((e1, e0, v_exc, plane.normal));
            }
        };
        push_edge(0, v2);
        push_edge(1, v0);
        push_edge(2, v1);

        self.tris.push(JTriangle { edges, plane });

    }

    /// Add first triangle and first edges to edge stack
    fn init(&mut self) {
        let v0 = self.find_max_point(|_, p| Some(p.x()));
        let p0 = self.pts[v0];

        let v0_xy = Vec2f::new(self.pts[v0].x(), self.pts[v0].y());
        let v1 = self.find_max_point(|i, p| (i != v0).then(|| (Vec2f::new(p.x(), p.y()) - v0_xy).normalized().x()));
        let p1 = self.pts[v1];

        let part_normal = {
            let d = p1 - p0;
            let mut pn = d.cross(Vec3f::new(1.0, 0.0, 0.0)).cross(d).normalized();
            if pn.dot(p0 - self.center).is_sign_negative() {
                pn = -pn;
            }
            pn
        };
        let v2 = self.find_max_point(|i, p| {
            if i == v0 || i == v1 {
                return None;
            }

            let mut normal = (p0 - p) % (p1 - p);
            if normal.dot(p - self.center).is_sign_negative() {
                normal = -normal;
            }

            Some(part_normal.dot(normal))
        });
        let p2 = self.pts[v2];

        // Fix orientation
        let (v0, v1, v2) = if Plane::from_points(p0, p1, p2).normal.dot(p0 - self.center).is_sign_negative() {
            (v2, v1, v0)
        } else {
            (v0, v1, v2)
        };

        self.insert_triangle(v0, v1, v2);
    }

    /// Perform building step
    fn step(&mut self, v0: usize, v1: usize, v_exclude: usize, normal: Vec3f) {
        // Find new point with minimum angle compared to
        let (p0, p1) = (self.pts[v0], self.pts[v1]);

        let v2 = self.find_max_point(|v2, p2| {
            if v2 == v0 || v2 == v1 || v2 == v_exclude {
                return None;
            }
            let plane = Plane::from_points(p0, p1, p2);

            if plane.normal.dot(p0 - self.center).is_sign_negative() {
                panic!("Something went very wrong");
            }

            Some(plane.normal.dot(normal))
        });

        self.insert_triangle(v0, v1, v2);
    }

    /// Finish building without coplanar polygon merge step
    fn finish_no_merge(&mut self) -> Vec<Polygon> {
        let mut polygons = Vec::new();

        for tri in self.tris.iter() {
            let pt = |i: usize| self.pts[tri.edges[i].0.indices(tri.edges[i].1).0];
            polygons.push(Polygon {
                points: vec![pt(0), pt(1), pt(2)],
                plane: tri.plane,
            })
        }

        polygons
    }

    // fn finish(&mut self) -> Vec<Polygon> {
    //     let mut polygons = Vec::new();
    //     let mut rest_tris = (0..self.tris.len()).collect::<HashSet<_>>();

    //     for tri in 0..self.tris.len() {
    //         if !rest_tris.remove(&tri) {
    //             continue;
    //         }
    //     }

    //     polygons
    // }

    // /// Build final polygon set
    // fn finish(&mut self) -> Vec<Polygon> {
    //     let mut polygons = Vec::new();
    //     let mut tri_inds = BTreeSet::from_iter(0..self.tris.len());

    //     while let Some(tri_ind) = tri_inds.pop_first() {
    //         // Try to merge triangle with all its neighbours
    //         let tri = &self.tris[tri_ind];

    //         // Plane all edges are merged with
    //         let merge_plane = tri.plane;

    //         for (edge, e_do_swap) in tri.edges.iter() {
    //             let n_ind = edge.neighbour_of(tri_ind).unwrap();

    //             // Triangle is already removed from triangle stack and is not needed to be checked
    //             if !tri_inds.contains(&n_ind) {
    //                 continue;
    //             }
    //             let n = &self.tris[n_ind];

    //             // Obviously not neighbour
    //             if n.plane.normal.dot(tri.plane.normal) <= 0.9 {
    //                 continue;
    //             }
    //         }
    //     }

    //     polygons
    // }

    /// Build convex hull
    pub fn build(&mut self) -> Vec<Polygon> {
        self.init();
        while let Some((v0, v1, v_exc, normal)) = self.edge_stack.pop() {
            self.step(v0, v1, v_exc, normal);
        }
        self.finish_no_merge()
    }
}

/// Build convex hull for point set
pub fn convex_hull(pts: &[Vec3f]) -> Result<Vec<Polygon>, ConvexHullError> {
    if pts.len() < 4 {
        return Err(ConvexHullError::NoInitSimplex);
    }

    Ok(JBuilder::new(pts).build())
}
