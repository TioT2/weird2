use core::f32;

use itertools::{chain, Itertools};

use crate::math::{self, Mat2f, Vec2f, Vec3f};

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

    // Split polygon by self
    pub fn split_polygon(&self, polygon: &Polygon) -> PolygonSplitResult {
        // check if polygon actually intersects and cut cases then not.
        match self.get_polygon_relation(polygon) {
            PolygonRelation::Back => return PolygonSplitResult::Back,
            PolygonRelation::Front => return PolygonSplitResult::Front,
            PolygonRelation::OnPlane => return PolygonSplitResult::OnPlane,
            _ => {}
        };

        let mut new_points = Vec::new();

        let mut curr = *polygon.points.last().unwrap();
        for next in polygon.points.iter().copied() {
            new_points.push(curr);

            let curr_rel = self.get_point_relation(curr);
            let next_rel = self.get_point_relation(next);

            // add new point if this line's crossing the polygon.
            if false
                || curr_rel == PointRelation::Front && next_rel == PointRelation::Back
                || curr_rel == PointRelation::Back && next_rel == PointRelation::Front
            {
                new_points.push(self.intersect_line(Line { base: curr, direction: curr - next }));
            }

            curr = next;
        }

        // TODO Fix this sh*t
        let all = new_points.iter().all(|p| self.get_point_relation(*p) != PointRelation::OnPlane);
        if all {
            println!("WTF?????");
        }

        // get resulting loop and find first root point
        let mut point_loop = new_points
            .iter()
            .copied()
            .cycle()
            .skip_while(|p| self.get_point_relation(*p) != PointRelation::OnPlane);

        let first_plane_point = point_loop.next().unwrap();

        let first = chain!(
            std::iter::once(first_plane_point),
            (&mut point_loop)
                .take_while_inclusive(|p| self.get_point_relation(*p) != PointRelation::OnPlane)
        ).collect::<Vec<_>>();
        
        let second_plane_point = *first.last().unwrap();

        let second = chain!(
            std::iter::once(second_plane_point),
            (&mut point_loop)
                .take_while_inclusive(|p| self.get_point_relation(*p) != PointRelation::OnPlane)
        ).collect::<Vec<_>>();

        let (front, back) = match self.get_point_relation(first[1]) {
            PointRelation::Front   => (first , second),
            PointRelation::Back    => (second, first ),
            PointRelation::OnPlane => panic!("geometry broken during 'split_polygon' function execution"),
        };

        PolygonSplitResult::Intersects {
            front : Polygon { plane: polygon.plane, points: front },
            back  : Polygon { plane: polygon.plane, points: back  },
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

impl Polygon {
    pub fn from_ccw(points: Vec<Vec3f>) -> Self {
        assert!(points.len() >= 3);

        let normal = ((points[2] - points[1]).normalized() % (points[0] - points[1]).normalized()).normalized();

        let point = points
            .iter()
            .fold(vec3f!(0.0, 0.0, 0.0), |s, v| s + *v)
            / points.len() as f32
        ;

        Self {
            plane: Plane {
                normal,
                distance: point ^ normal,
            },
            points,
        }
    }

    /// Build polygon from clockwise-going points
    pub fn from_cw(points: Vec<Vec3f>) -> Self {
        // yep, that's very ineffective solution, but I don't care (in this case)
        Self::from_ccw(points.into_iter().rev().collect())
    }
}
