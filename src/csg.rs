use crate::{geom::{BoundBox, Polygon}, math::Vec3f};

pub struct Brush {
    pub faces: Vec<Polygon>,
    pub boundbox: BoundBox,
}

impl Brush {
    pub fn cube() -> Self {
        let p = vec![
            Vec3f::new(0.0, 0.0, 0.0),
            Vec3f::new(1.0, 0.0, 0.0),
            Vec3f::new(1.0, 1.0, 0.0),
            Vec3f::new(0.0, 1.0, 0.0),
            Vec3f::new(0.0, 0.0, 1.0),
            Vec3f::new(1.0, 0.0, 1.0),
            Vec3f::new(1.0, 1.0, 1.0),
            Vec3f::new(0.0, 1.0, 1.0),
        ];

        Self {
            faces: vec![
                Polygon::from_cw(vec![p[0], p[1], p[2], p[3]]),
                Polygon::from_cw(vec![p[5], p[4], p[7], p[6]]),
                Polygon::from_cw(vec![p[0], p[4], p[5], p[1]]),
                Polygon::from_cw(vec![p[1], p[5], p[6], p[2]]),
                Polygon::from_cw(vec![p[2], p[6], p[7], p[3]]),
                Polygon::from_cw(vec![p[3], p[7], p[4], p[0]]),
            ],
            boundbox: BoundBox::new(
                Vec3f::new(0.0, 0.0, 0.0),
                Vec3f::new(1.0, 1.0, 1.0)
            ),
        }
    }

    pub fn translate(&mut self, offset: Vec3f) -> &mut Self {
        for face in self.faces.iter_mut() {
            face.plane = face.plane.translate(offset);
            for point in face.points.iter_mut() {
                *point += offset;
            }
        }
        self.boundbox = self.boundbox.translate(offset);
        self
    }

    pub fn scale(&mut self, scale: Vec3f) -> &mut Self {
        for face in self.faces.iter_mut() {
            face.plane = face.plane.scale(scale);
            for point in face.points.iter_mut() {
                *point *= scale;
            }
        }
        self.boundbox = self.boundbox.scale(scale);
        self
    }
}

pub enum FaceType {
    Solid,
    Portal(usize),
}

pub struct Face {
    pub polygon: Polygon,
    pub ty: FaceType,
}

pub struct Volume {
    pub faces: Vec<Face>,
    pub boundbox: BoundBox,
}

pub struct Builder {
    pub volumes: Vec<Volume>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            volumes: Vec::new(),
        }
    }

    pub fn add_brush(&mut self, brush: &Brush) {
        if self.volumes.is_empty() {
            self.volumes.push(Volume {
                faces: brush.faces
                    .iter()
                    .map(|v| Face {
                        polygon: v.clone(),
                        ty: FaceType::Solid,
                    })
                    .collect(),
                boundbox: brush.boundbox,
            });

            return;
        }

        // then intersect volume with all another volumes
    }
}

