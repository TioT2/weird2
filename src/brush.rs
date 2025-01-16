use crate::{geom, math::{Mat3f, Vec3f}};

pub struct Brush {
    pub polygons: Vec<geom::Polygon>,
    pub bound_box: geom::BoundBox,
}

impl Brush {
    pub fn from_planes(planes: &[geom::Plane]) -> Option<Brush> {
        let mut polygons = Vec::<geom::Polygon>::new();

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
                    .all(|plane| plane.get_point_relation(*point) != geom::PointRelation::Front)
                )
                .collect();

            points = geom::deduplicate_points(points);
            
            // check if points are even a polygon
            if points.len() < 3 {
                continue;
            }

            points = geom::sort_plane_points(points, *p1);

            polygons.push(geom::Polygon { points, plane: *p1 });
        }

        let mut bound_box = geom::BoundBox::zero();
        for polygon in &polygons {
            bound_box = bound_box.total(&geom::BoundBox::for_points(polygon.points.iter().copied()));
        }

        Some(Brush {
            polygons,
            bound_box,
        })
    }
}

pub fn get_map_polygons(brushes: &[Brush], filter: bool) -> Vec<geom::Polygon> {
    if filter {
        let mut result = Vec::new();

        for brush in brushes {
            'polygon_loop: for polygon in &brush.polygons {
                let polygon_bbox = geom::BoundBox::for_points(polygon.points.iter().copied());
    
                'second_loop: for second_brush in brushes {
                    if false
                        || std::ptr::eq(brush, second_brush)
                        || !polygon_bbox.is_intersecting(&second_brush.bound_box)
                    {
                        continue;
                    }
    
                    for second_polygon in &second_brush.polygons {
                        if second_polygon.plane.get_polygon_relation(polygon) != geom::PolygonRelation::Back {
                            continue 'second_loop;
                        }
                    }
    
                    continue 'polygon_loop;
                }
    
                result.push(polygon.clone());
            }
        }
    
        result
    } else {
        brushes
            .iter()
            .map(|brush| brush.polygons.iter())
            .flat_map(|v| v)
            .cloned()
            .collect::<Vec<_>>()
    }
}
