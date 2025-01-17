/// Standard geometry primitive module

use crate::math::{Mat2f, Vec2f, Vec3f};

/// Geometric epsilon (1 mm)
pub const GEOM_EPSILON: f32 = 0.001;

/// Plane represetnation structure
/// 
/// ## Equation
/// Standard plane equation is Ax + By + Cz + D = 0. In this case,
/// * A = point.x
/// * B = point.y
/// * C = point.z
/// * D = -distance
/// 
#[derive(Debug, Copy, Clone)]
pub struct Plane {
    /// plane normal
    pub normal: Vec3f,

    /// number to multiply normal to to get basic point
    pub distance: f32,
}

/// Bounding box
#[derive(Copy, Clone)]
pub struct BoundBox {
    /// minimal vector
    min: Vec3f,

    /// maximal vector
    max: Vec3f,
}

impl BoundBox {
    /// Build boundbox from min/max vectors
    /// 
    /// # Safety
    /// It's safe to call this function only if for all t min.t <= max.t
    pub unsafe fn from_minmax(min: Vec3f, max: Vec3f) -> Self {
        Self { min, max }
    }

    /// 'Empty' bounding box
    pub fn zero() -> Self {
        Self {
            min: Vec3f::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Vec3f::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Build minimal boundbox that contains this pair of points
    pub fn new(p1: Vec3f, p2: Vec3f) -> Self {
        Self {
            min: Vec3f::new(
                f32::min(p1.x, p2.x),
                f32::min(p1.y, p2.y),
                f32::min(p1.z, p2.z),
            ),
            max: Vec3f::new(
                f32::max(p1.x, p2.x),
                f32::max(p1.y, p2.y),
                f32::max(p1.z, p2.z),
            ),
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
    pub fn extend(self, delta: Vec3f) -> Self {
        if delta.x < 0.0 || delta.y < 0.0 || delta.z < 0.0 {
            self
        } else {
            Self {
                min: self.min - delta,
                max: self.max + delta,
            }
        }

    }

    /// Get minimal boundbox that contains all points from both of `self` and `rhs`
    pub fn total(&self, rhs: &BoundBox) -> Self {
        Self {
            min: Vec3f::new(
                f32::min(self.min.x, rhs.min.x),
                f32::min(self.min.y, rhs.min.y),
                f32::min(self.min.z, rhs.min.z),
            ),
            max: Vec3f::new(
                f32::max(self.max.x, rhs.max.x),
                f32::max(self.max.y, rhs.max.y),
                f32::max(self.max.z, rhs.max.z),
            ),
        }
    }

    /// Calculate common boundbox for some point sef.
    /// Note: Point set must be finite (if you want to get result, of course).
    pub fn for_points(iter: impl Iterator<Item = Vec3f>) -> Self {
        let mut min = Vec3f::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3f::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for point in iter {
            min.x = f32::min(min.x, point.x);
            min.y = f32::min(min.y, point.y);
            min.z = f32::min(min.z, point.z);

            max.x = f32::max(max.x, point.x);
            max.y = f32::max(max.y, point.y);
            max.z = f32::max(max.z, point.z);
        }

        Self { min, max }
    }

    /// Check if boundbox intersection isn't empty
    pub fn is_intersecting(&self, another: &BoundBox) -> bool {
        true
            && self.max.x >= another.min.x && self.min.x <= another.max.x
            && self.max.y >= another.min.y && self.min.y <= another.max.y
            && self.max.z >= another.min.z && self.min.z <= another.max.z
    } // is_intersecting
}

/// Line in space
#[derive(Debug, Copy, Clone)]
pub struct Line {
    /// Line direction vector, assumed to be normalized
    pub direction: Vec3f,

    /// Line base point
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
    OnPlane,

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

/// Polygon by plane splitting result, resembles `PolygonRelation` by structure
pub enum PolygonSplitResult {
    /// Polygon located in front of plane, so it doesn't require splitting
    Front,

    /// Polygon located back of plane, so it doesn't require splitting
    Back,

    /// Polygon located back of plane, so it doesn't require splitting
    OnPlane,

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
        (self.normal % other.normal).length2() <= GEOM_EPSILON && (self.distance - other.distance).abs() <= GEOM_EPSILON
    }
}

impl Plane {
    /// Build plane for point triple (normal is calculated as if triple will be CCW-oriented)
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
        // just for safety)
        normal.normalize();

        Self { distance: point ^ normal, normal }
    }

    // Intersect this plane with another one
    pub fn intersect_plane(&self, rhs: Plane) -> Line {
        let direction = (self.normal % rhs.normal).normalized();

        let npvec = Vec2f::new(
            -self.distance,
            -rhs.distance
        );
        
        let res;

        macro_rules! for_direction {
            ($c1: ident, $c2: ident, $vx: expr, $vy: expr, $vz: expr) => {
                if let Some(inv) = Mat2f::new(
                    self.normal.$c1,
                    self.normal.$c2,
                    rhs.normal.$c1,
                    rhs.normal.$c2,
                ).inversed() {
                    res = inv * npvec;
                    return Line {
                        direction,
                        base: Vec3f::new($vx, $vy, $vz)
                    };
                }
            };
        }

        for_direction!(y, z, 0.0,   res.x, res.y);
        for_direction!(x, z, res.x, 0.0,   res.y);
        for_direction!(x, y, res.x, res.y,   0.0);

        panic!("Maths is broken - line MUST intersect at least one coodrinate plane.");
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
            } else if false
                || prev_relation == PointRelation::Front && curr_relation == PointRelation::Back
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

    /// Get relation of point and plane
    pub fn get_point_relation(&self, point: Vec3f) -> PointRelation {
        let metrics = (point ^ self.normal) - self.distance;

        if metrics > GEOM_EPSILON {
            PointRelation::Front
        } else if metrics < -GEOM_EPSILON {
            PointRelation::Back
        } else {
            PointRelation::OnPlane
        }
    }

    // Get relation of plane and polygon
    pub fn get_polygon_relation(&self, polygon: &Polygon) -> PolygonRelation {
        let mut front_occured = false;
        let mut back_occured = false;

        for point in &polygon.points {
            match self.get_point_relation(*point) {
                PointRelation::Front   => front_occured    = true,
                PointRelation::Back    => back_occured     = true,
                _ => {}
            }
        }

        match (front_occured, back_occured) {
            (false, false) => PolygonRelation::OnPlane,
            (false, true ) => PolygonRelation::Back,
            (true , false) => PolygonRelation::Front,
            (true , true ) => PolygonRelation::Intersects,
        }
    }

    // Get intersection of this polygon and line
    pub fn intersect_line(&self, line: Line) -> Vec3f {
        let t = (self.distance - (line.base ^ self.normal)) / (line.direction ^ self.normal);

        line.base + line.direction * t
    }

    /// Split polygon by the plane
    pub fn split_polygon(&self, polygon: &Polygon) -> PolygonSplitResult {
        match self.get_polygon_relation(polygon) {
            PolygonRelation::Back => return PolygonSplitResult::Back,
            PolygonRelation::Front => return PolygonSplitResult::Front,
            PolygonRelation::OnPlane => return PolygonSplitResult::OnPlane,
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
            } else if false
                || prev_relation == PointRelation::Front && curr_relation == PointRelation::Back
                || curr_relation == PointRelation::Front && prev_relation == PointRelation::Back
            {
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

/// Polygon. Polygon structure contains .
/// TODO: build convexity-safe polygon.
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
            for point in prev.iter().copied() {
                if (candidate - point).length2() < GEOM_EPSILON {
                    return prev;
                }
            }

            prev.push(candidate);
            prev
        })
}

/// Sort point set by angle from pivot point
pub fn sort_plane_points(mut points: Vec<Vec3f>, plane: Plane) -> Vec<Vec3f> {
    let center = points
        .iter()
        .copied()
        .fold(Vec3f::zero(), std::ops::Add::add)
        / (points.len() as f32)
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
                let cross_normal_dot = (last % v) ^ plane.normal;

                // check for direction and calculate cotangent
                if cross_normal_dot < 0.0 {
                    Some((index, (last ^ v) / cross_normal_dot))
                } else {
                    None
                }
            })
            .min_by(|l, r| if l.1 <= r.1 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            })
        ;

        if let Some((smallest_cotan_index, _)) = smallest_cotan_opt {
            sorted.push(points.swap_remove(smallest_cotan_index));
        } else {
            break;
        }
    }

    // fix point set orientation
    // TODO: Fix this sh*tcode
    if sorted.len() >= 3 {
        let point_normal = Vec3f::cross(
            (sorted[2] - sorted[1]).normalized(),
            (sorted[0] - sorted[1]).normalized(),
        ).normalized();
    
        // fix point orientation
        if (point_normal ^ plane.normal) < 0.0 {
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

    /// Build boundbox for polygon
    pub fn build_bound_box(&self) -> BoundBox {
        BoundBox::for_points(self.points.iter().copied())
    } // build_bound_box

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
    } // iter_edge_planes

    /// From convex point set, normal is calculated by assuming polygon is clockwise
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
