use core::f32;

use crate::{geom::{BoundBox, Plane, PointRelation, Polygon, PolygonRelation, GEOM_EPSILON}, math::{Mat3f, Vec3f}};

pub struct Brush {
    pub polygons: Vec<Polygon>,
    pub bound_box: BoundBox,
}

/// Points from some plane sorting function
fn sort_plane_points(mut points: Vec<Vec3f>, plane: &Plane) -> Vec<Vec3f> {
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
            .min_by(|l, r| if l.1 < r.1 {
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

    sorted
}

impl Brush {
    pub fn from_planes(planes: &[Plane]) -> Option<Brush> {
        let mut polygons = Vec::<Polygon>::new();

        for p1 in planes {
            let mut points = Vec::new();

            for p2 in planes {
                if std::ptr::eq(p1, p2) {
                    continue;
                }

                for p3 in planes {
                    if std::ptr::eq(p1, p3) || std::ptr::eq(p2, p3) {
                        continue;
                    }

                    let mat = Mat3f::from_rows(p1.normal, p2.normal, p3.normal);
                    let inv = match mat.inversed() {
                        Some(m) => m,
                        None => continue,
                    };
                    let intersection_point = inv * Vec3f::new(p1.distance, p2.distance, p3.distance);

                    points.push(intersection_point);
                }
            }

            // remove points that can't belong to final polygon
            points = points
                .into_iter()
                .filter(|point| planes
                    .iter()
                    .all(|plane| plane.get_point_relation(*point) != PointRelation::Front)
                )
                .collect();

            // deduplicate points
            points = points
                .into_iter()
                .fold(Vec::new(), |mut prev, candidate| {
                    for point in prev.iter().copied() {
                        if (candidate - point).length2() < GEOM_EPSILON {
                            return prev;
                        }
                    }

                    prev.push(candidate);
                    prev
                });
            
            // check if points are even a polygon
            if points.len() < 3 {
                continue;
            }

            polygons.push(Polygon {
                points: sort_plane_points(points, p1),
                plane: *p1,
            })
        }

        let mut bound_box = BoundBox::zero();
        for polygon in &polygons {
            bound_box = bound_box.total(&BoundBox::for_points(&polygon.points));
        }
        Some(Brush {
            polygons,
            bound_box,
        })
    }
}

pub fn get_map_polygons(brushes: &[Brush]) -> Vec<Polygon> {
    let mut result = Vec::new();

    for brush in brushes {
        'polygon_loop: for polygon in &brush.polygons {
            let polygon_bbox = BoundBox::for_points(&polygon.points);

            'second_loop: for second_brush in brushes {
                if false
                    || std::ptr::eq(brush, second_brush)
                    || !polygon_bbox.is_intersecting(&second_brush.bound_box)
                {
                    continue;
                }

                for second_polygon in &second_brush.polygons {
                    if second_polygon.plane.get_polygon_relation(polygon) != PolygonRelation::Back {
                        continue 'second_loop;
                    }
                }

                continue 'polygon_loop;
            }

            result.push(polygon.clone());
        }
    }

    result

    // return brushes
    //     .iter()
    //     .map(|brush| brush.polygons.iter())
    //     .flat_map(|v| v)
    //     .cloned()
    //     .collect::<Vec<_>>()
    // ;

    /*
    let mut result = Vec::new();

    for brush in brushes {
        'polygon_loop: for polygon in &brush.polygons {
            let polygon_bbox = BoundBox::for_points(&polygon.points);

            'second_loop: for second in brushes {
                if std::ptr::eq(brush, second) || !polygon_bbox.is_intersecting(&second.bound_box) {
                    continue;
                }

                for second_polygon in &second.polygons {
                    if second_polygon.plane.get_polygon_relation(polygon) != PolygonRelation::Back {
                        continue 'second_loop;
                    }
                }

                continue 'polygon_loop;
            }

            result.push(polygon.clone());
        }
    }

    result
     */
}