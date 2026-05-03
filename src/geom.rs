//! Standard geometry primitive implementation module

use crate::math::{Vec2f, Vec3f, Vec4f};

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
        // let other = if self.distance.signum() == other.distance.signum() {
        //     *other
        // } else {
        //     other.negate_direction()
        // };

        f32_geom_equal((self.normal % other.normal).length(), 0.0) && f32_vec_equal(&(self.normal * self.distance.into()), &(other.normal * other.distance.into()))
    }
}

impl Plane {
    /// Build plane for point triple (normal is calculated as if triple is CCW-oriented)
    pub fn from_points(p1: Vec3f, p2: Vec3f, p3: Vec3f) -> Self {
        let normal = Vec3f::cross(
            (p3 - p2).normalized(),
            (p1 - p2).normalized()
        ).normalized();

        let distance = p2 ^ normal;

        Self { normal, distance }
    }

    /// Build plane from it's point and normal
    pub fn from_point_normal(point: Vec3f, mut normal: Vec3f) -> Self {
        normal.normalize();
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
        .fold(Vec3f::zero(), std::ops::Add::add) / (points.len() as f32).into()
    ;

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

impl Polygon {
    /// Negate polygon orientation (e.g. normal is now -normal, point order is reversed to fit normal.)
    pub fn negate_orientation(&mut self) {
        self.plane = self.plane.negate_direction();
        self.points.reverse();
    }

    /// Build polygon bounding box
    pub fn build_bound_box(&self) -> BoundBox {
        BoundBox::for_points(self.points.iter().copied())
    }

    /// Iterator on planes that are parallel to polygon normal and contain corresponding edges
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
                    .fold1(|x, y| x && y)
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
                    && <$Vec>::zip(self.max, another.min, |l, r| l >= r).fold1(|x, y| x && y)
                    && <$Vec>::zip(self.min, another.max, |l, r| l <= r).fold1(|x, y| x && y)
            }

            /// Check if boundbox contains point
            pub fn contains_point(&self, point: &$Pt) -> bool {
                let v = $pt2v(*point);
                true
                    && <$Vec>::zip(self.max, v, |l, r| l >= r).fold1(|x, y| x && y)
                    && <$Vec>::zip(self.min, v, |l, r| l <= r).fold1(|x, y| x && y)
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
            min: self.min - v, // Vec4f::zip(self.min, v, f32::min),
            max: self.max + v, // Vec4f::zip(self.max, v, f32::max),
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
