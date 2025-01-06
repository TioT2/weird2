use crate::geom::{Plane, Polygon, PolygonRelation, PolygonSplitResult};

#[derive(Clone)]
struct BspPolygon {
    pub polygon: Polygon,
    pub plane_is_splitter: bool,
}

#[derive(Debug)]
pub enum Bsp {
    Element {
        /// plane that splits current volume into two subvolumes.
        splitter: Plane,

        /// sub-trees front of splitter plane
        front: Box<Bsp>,

        /// sub-trees back of splitter plane
        back: Box<Bsp>,
    },
    Leaf {
        polygons: Vec<Polygon>,
    },
}

impl Bsp {
    fn from_internal(polygons: Vec<BspPolygon>) -> Bsp {
        let mut best_rate = i32::MAX;
        let mut best_splitter_index = Option::<usize>::None;

        for splitter_index in 0..polygons.len() {
            let splitter = &polygons[splitter_index];

            if !splitter.plane_is_splitter {
                continue;
            }

            let splitter_plane = splitter.polygon.plane;

            let mut front_count: i32 = 0;
            let mut back_count: i32 = 0;
            let mut split_count: i32 = 0;
            for index in 0..polygons.len() {
                let polygon = &polygons[index];

                let relation = splitter_plane.get_polygon_relation(&polygon.polygon);

                match relation {
                    PolygonRelation::Front | PolygonRelation::OnPlane => front_count += 1,
                    PolygonRelation::Back => back_count += 1,
                    PolygonRelation::Intersects => split_count += 1,
                }
            }

            let rate = (front_count - back_count).abs();
            _ = split_count;

            // check if splitter is actually splitting anything
            if front_count != 0 && back_count != 0 && rate < best_rate {
                best_rate = rate;
                best_splitter_index = Some(splitter_index);
            }
        }

        // there's no splitters, actually, so this polygon is leaf.
        let Some(splitter_index) = best_splitter_index else {
            return Bsp::Leaf {
                polygons: polygons
                    .into_iter()
                    .map(|p| p.polygon)
                    .collect()
            };
        };

        // split polygon
        let splitter_plane = polygons[splitter_index].polygon.plane;
        let mut front_polygons: Vec<BspPolygon> = Vec::new();
        let mut back_polygons: Vec<BspPolygon> = Vec::new();

        for mut polygon in polygons.into_iter() {
            match splitter_plane.split_polygon(&polygon.polygon) {
                PolygonSplitResult::Front => {
                    front_polygons.push(polygon);
                }
                PolygonSplitResult::Back => {
                    back_polygons.push(polygon);
                }
                PolygonSplitResult::OnPlane => {
                    polygon.plane_is_splitter = false;
                    front_polygons.push(polygon);
                }
                PolygonSplitResult::Intersects { front, back } => {
                    front_polygons.push(BspPolygon {
                        polygon: front,
                        plane_is_splitter: polygon.plane_is_splitter,
                    });
                    back_polygons.push(BspPolygon {
                        polygon: back,
                        plane_is_splitter: polygon.plane_is_splitter,
                    });
                }
            }
        }

        let splitter = splitter_plane;
        let front = Box::new(Bsp::from_internal(front_polygons));
        let back = Box::new(Bsp::from_internal(back_polygons));


        Bsp::Element { splitter, front, back }
    }

    pub fn build(polygons: &[Polygon]) -> Bsp {
        Self::from_internal(polygons 
            .iter()
            .map(|p| BspPolygon {
                polygon: p.clone(),
                plane_is_splitter: true
            })
            .collect()
        )
    }

    pub fn polygon_count(&self) -> usize {
        match self {
            Bsp::Element { front, back, .. }
                => front.polygon_count() + back.polygon_count(),
            Bsp::Leaf { .. }
                => 1,
        }
    }
}
