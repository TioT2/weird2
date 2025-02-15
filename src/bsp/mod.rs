///! Main project BSP structure declaration module

pub use compiler::Error as MapCompilationError;
use std::num::NonZeroU32;
use crate::{geom, map, math::Vec3f};

/// Declare actual map builder module
pub mod compiler;

/// WBSP description module
pub mod wbsp;

/// Id type
pub trait Id: Copy + Clone + Eq + PartialEq + std::hash::Hash + std::fmt::Debug + Ord + PartialOrd {
    fn from_index(index: usize) -> Self;

    fn into_index(self) -> usize;
}

/// Generic id implementation
macro_rules! impl_id {
    ($name: ident) => {
        /// Unique identifier
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
        pub struct $name(NonZeroU32);

        impl Id for $name {
            /// Build id from index
            fn from_index(index: usize) -> Self {
                $name(NonZeroU32::try_from(index as u32 + 1).unwrap())
            }
    
            /// Get index by id
            fn into_index(self) -> usize {
                self.0.get() as usize - 1
            }
        }
    };
}

impl_id!(VolumeId);
impl_id!(PolygonId);
impl_id!(MaterialId);
impl_id!(BspModelId);
impl_id!(DynamicModelId);

// TODO: Rename surface into smth more logical.

/// Visible volume face piece
pub struct Surface {
    /// Polygon material identifier
    pub material_id: MaterialId,

    /// Polygon itself identifier
    pub polygon_id: PolygonId,

    /// Surface transparency flag
    pub is_transparent: bool,

    /// True if sky, false if not
    pub is_sky: bool,

    /// U texture-mapping axis
    pub u: geom::Plane,

    /// V texture-mapping axis
    pub v: geom::Plane,
}

/// Portal (volume-volume connection)
pub struct Portal {
    /// Portal polygon identifier
    pub polygon_id: PolygonId,
    
    /// Is portal polygon facing 'into' volume it belongs to.
    /// This flag is used to share same portal polygons between different volumes
    pub is_facing_front: bool,

    /// Destination volume's identifier
    pub dst_volume_id: VolumeId,
}

/// Convex collection of polygons and portals
pub struct Volume {
    /// Set of visible volume elements
    surfaces: Vec<Surface>,

    /// Set of connections to another volumes
    portals: Vec<Portal>,

    /// Volume bounding box
    bound_box: geom::BoundBox,
}

impl Volume {
    /// Get physical polygon set
    pub fn get_surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    /// Get portal set
    pub fn get_portals(&self) -> &[Portal] {
        &self.portals
    }

    /// Get bounding box
    pub fn get_bound_box(&self) -> &geom::BoundBox {
        &self.bound_box
    }
}

/// Binary Space Partition, used during
pub enum Bsp {
    /// Space partition
    Partition {
        /// Plane that splits front/back volume sets. Front volume set is located in front of plane.
        splitter_plane: geom::Plane,

        /// Pointer to front polygon part
        front: Box<Bsp>,

        /// Pointer to back polygon part
        back: Box<Bsp>,
    },

    /// Final volume
    Volume(VolumeId),

    /// Nothing there
    Void,
}

impl Bsp {
    /// Find BSP cell that contains the point
    pub fn traverse(&self, point: Vec3f) -> Option<VolumeId> {
        let mut curr = self;

        loop {
            match curr {
                Bsp::Partition { splitter_plane, front, back } => {
                    curr = match splitter_plane.get_point_relation(point) {
                        geom::PointRelation::Front | geom::PointRelation::OnPlane => front,
                        geom::PointRelation::Back => back,
                    };
                }
                Bsp::Volume(volume_id) => return Some(*volume_id),
                Bsp::Void => return None,
            }
        }
    }

    /// Calculate BSP tree depth
    pub fn depth(&self) -> usize {
        match self {
            Bsp::Partition { splitter_plane: _, front, back }
                => usize::max(front.depth(), back.depth()) + 1,
            _
                => 1,
        }
    }

    /// Count of BSP elements
    pub fn size(&self) -> usize {
        match self {
            Bsp::Partition { splitter_plane: _, front, back }
                => front.size() + back.size() + 1,
            _
                => 1,
        }
    }
}

/// Static model
pub struct BspModel {
    /// Model BSP
    bsp: Box<Bsp>,

    /// (Simple) Bounding volume, used during splitting process
    bound_box: geom::BoundBox,
}

impl BspModel {
    /// Get BSP 
    pub fn get_bsp(&self) -> &Bsp {
        &self.bsp
    }

    /// Get bounding volume
    pub fn get_bound_box(&self) -> &geom::BoundBox {
        &self.bound_box
    }
}

/// Dynamic BSP element
pub struct DynamicModel {
    /// Model translation
    origin: Vec3f,

    /// Model rotation (along Y axis)
    rotation: f32,

    /// Corresponding BSP model Id
    model_id: BspModelId,
}

impl DynamicModel {
    pub fn get_origin(&self) -> Vec3f {
        self.origin
    }

    pub fn get_rotation(&self) -> f32 {
        self.rotation
    }

    pub fn get_model_id(&self) -> BspModelId {
        self.model_id
    }
}

/// Map
pub struct Map {
    /// Set of map polygons
    polygon_set: Vec<geom::Polygon>,

    /// Set of map materials
    material_name_set: Vec<String>,

    /// Set of map volumes
    volume_set: Vec<Volume>,

    /// Set of BSP models
    bsp_models: Vec<BspModel>,

    /// World BSP model
    world_model_id: BspModelId,

    /// Set of dynamically-rendered objects
    dynamic_models: Vec<DynamicModel>,
}

impl Map {
    /// Get volume by id
    pub fn get_volume(&self, id: VolumeId) -> Option<&Volume> {
        self.volume_set.get(id.into_index())
    }

    /// Get iterator on ids of all volumes
    pub fn all_volume_ids(&self) -> impl Iterator<Item = VolumeId> {
        (0..self.volume_set.len()).map(VolumeId::from_index)
    }

    /// Iterate though dynamic model IDs
    pub fn all_dynamic_model_ids(&self) -> impl Iterator<Item = DynamicModelId> {
        (0..self.dynamic_models.len()).map(DynamicModelId::from_index)
    }

    /// Get material name by it's id
    pub fn get_material_name(&self, id: MaterialId) -> Option<&str> {
        self.material_name_set.get(id.into_index()).map(|s| s.as_str())
    }

    /// Iterate by material names
    pub fn all_material_names(&self) -> impl Iterator<Item = (MaterialId, &str)> {
        self.material_name_set
            .iter()
            .enumerate()
            .map(|(index, name)| (MaterialId::from_index(index), name.as_ref()))
    }

    /// Get dynamic model by id
    pub fn get_dynamic_model(&self, id: DynamicModelId) -> Option<&DynamicModel> {
        self.dynamic_models.get(id.into_index())
    }

    /// Get polygon by id
    pub fn get_polygon(&self, id: PolygonId) -> Option<&geom::Polygon> {
        self.polygon_set.get(id.into_index())
    }

    pub fn get_world_model_id(&self) -> BspModelId {
        self.world_model_id
    }

    pub fn get_world_model(&self) -> &BspModel {
        self.bsp_models.get(self.world_model_id.into_index()).unwrap()
    }

    /// Get BSP model by id
    pub fn get_bsp_model(&self, id: BspModelId) -> Option<&BspModel> {
        self.bsp_models.get(id.into_index())
    }

    /// Test if volume contains point or not
    pub fn volume_contains_point(&self, id: VolumeId, point: Vec3f) -> Option<bool> {
        let volume = self.get_volume(id)?;

        for portal in &volume.portals {
            let mut plane = self.polygon_set[portal.polygon_id.into_index()].plane;

            if !portal.is_facing_front {
                plane = plane.negate_direction();
            }

            if plane.get_point_relation(point) == geom::PointRelation::Back {
                return Some(false);
            }
        }

        for surface in &volume.surfaces {
            let plane = self.polygon_set[surface.polygon_id.into_index()].plane;

            if plane.get_point_relation(point) == geom::PointRelation::Back {
                return Some(false);
            }
        }

        Some(true)
    }
}

/// Map from file loading error
#[derive(Debug)]
pub enum MapLoadingError {
    /// Input/Output error
    IoError(std::io::Error),

    /// Error during building string from UTF-8 byte array
    Utf8Error(std::str::Utf8Error),

    /// Invalid magic value
    InvalidMagic(u32),

    /// Invalid BSP type
    InvalidBspType(u8),

    /// Invalid span occured
    InvalidSpan {
        span_kind: &'static str,
        invalid_reason: &'static str,
        buffer_size: usize,
        span: wbsp::Span,
    },

    /// Invalid index
    InvalidIndex {
        kind: &'static str,
        count: usize,
        index: u32,
    },
}

impl From<std::io::Error> for MapLoadingError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

impl From<std::str::Utf8Error> for MapLoadingError {
    fn from(value: std::str::Utf8Error) -> Self {
        Self::Utf8Error(value)
    }
}

/// Map isn't saved
#[derive(Debug)]
pub enum MapSavingError {
    /// IO error occured
    IoError(std::io::Error),

    /// Incomplete map write
    IncompleteWrite,
}

impl From<std::io::Error> for MapSavingError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

impl Map {
    /// Compile map to WBSP
    pub fn compile(map: &map::Map) -> Result<Self, MapCompilationError> {
        compiler::compile(map)
    }
}

impl Map {
    pub fn load(src: &mut dyn std::io::Read) -> Result<Self, MapLoadingError> {
        // Read typed span
        pub fn load_read_tspan<'t, T: Copy + Clone + bytemuck::AnyBitPattern>(
            data: &'t [u8],
            span: wbsp::Span
        ) -> Result<&'t [T], MapLoadingError> {
            if std::mem::size_of::<T>() == 0 {
                // Empty slice of nothing
                return Ok(&[]);
            }

            let elem_count = span.size as usize / std::mem::size_of::<T>();
            let offset = span.offset as usize;

            let slice = data
                .get(offset..offset + elem_count * std::mem::size_of::<T>())
                .ok_or(MapLoadingError::InvalidSpan {
                    span_kind: "main",
                    invalid_reason: "span overflow",
                    buffer_size: data.len(),
                    span
                })?;

            bytemuck::try_cast_slice::<_, T>(slice)
                .map_err(|err| {
                    let reason = match err {
                        bytemuck::PodCastError::AlignmentMismatch => "alignment mismatch",
                        bytemuck::PodCastError::OutputSliceWouldHaveSlop => "output slice would have slop",
                        bytemuck::PodCastError::SizeMismatch => "size mismatch",
                        bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => "target alignment greater and input not aligned",
                    };

                    MapLoadingError::InvalidSpan {
                        invalid_reason: reason,
                        span_kind: "header",
                        buffer_size: data.len(),
                        span,
                    }
                })
        }

        fn get_id<T: Id, E>(index: u32, arr: &[E], kind: &'static str) -> Result<T, MapLoadingError> {
            if index as usize >= arr.len() {
                return Err(MapLoadingError::InvalidIndex {
                    kind,
                    count: arr.len(),
                    index
                });
            }

            Ok(T::from_index(index as usize))
        }

        /// Build BSP starting from kind of array
        fn bsp_from(elems: &[wbsp::BspElement], volumes: &[wbsp::Volume], start: u32) -> Result<(Box<Bsp>, u32), MapLoadingError> {
            let elem = elems
                .get(start as usize)
                .ok_or(MapLoadingError::InvalidIndex {
                    kind: "bsp element",
                    count: elems.len(),
                    index: start,
                })?;

            let ty = match wbsp::BspType::try_from(elem.ty) {
                Ok(ty) => ty,
                Err(err) => {
                    return Err(MapLoadingError::InvalidBspType(err));
                }
            };

            Ok(match ty {
                wbsp::BspType::Partition => {
                    let partition = unsafe { &elem.data.partition };
                    let splitter_plane: geom::Plane = partition.plane.into();

                    let (front, front_end) = bsp_from(elems, volumes, start + 1)?;
                    let (back, back_end) = bsp_from(elems, volumes, front_end)?;

                    (Box::new(Bsp::Partition { splitter_plane, front, back }), back_end)
                }
                wbsp::BspType::Volume => {
                    let volume = unsafe { &elem.data.volume };
                    let id: VolumeId = get_id(volume.volume_index, volumes, "volume")?;

                    (Box::new(Bsp::Volume(id)), start + 1)
                }
                wbsp::BspType::Void => {
                    (Box::new(Bsp::Void), start + 1)
                }
            })
        }

        fn get_slice<'t, E>(
            span: wbsp::Span,
            arr: &'t [E],
            kind: &'static str,
        ) -> Result<&'t [E], MapLoadingError> {
            arr
                .get(span.range())
                .ok_or(MapLoadingError::InvalidSpan {
                    span_kind: kind,
                    invalid_reason: "span overflow",
                    buffer_size: arr.len(),
                    span,
                })
        }


        let mut data = Vec::new();

        src.read_to_end(&mut data)?;

        let header = load_read_tspan::<wbsp::Header>(
            &data,
            wbsp::Span { offset: 0, size: std::mem::size_of::<wbsp::Header>() as u32 }
        )?.get(0).unwrap();

        // Check file magic
        if header.magic != wbsp::MAGIC {
            return Err(MapLoadingError::InvalidMagic(header.magic));
        }

        macro_rules! load_head_span {
            ($name: ident, $type: ty) => {
                let $name = load_read_tspan::<$type>(&data, header.$name)?;
            };
        }

        // Load spans stored in header
        load_head_span!(chars,           u8                 );
        load_head_span!(points,          wbsp::Vec3         );
        load_head_span!(polygons,        wbsp::Span         );
        load_head_span!(material_names,  wbsp::Span         );
        load_head_span!(volume_surfaces, wbsp::Surface      );
        load_head_span!(volume_portals,  wbsp::Portal       );
        load_head_span!(volumes,         wbsp::Volume       );
        load_head_span!(bsp_elements,    wbsp::BspElement   );
        load_head_span!(bsp_models,      wbsp::BspModel     );
        load_head_span!(dynamic_models,  wbsp::DynamicModel );

        let map = Map {
            polygon_set: {
                let polygons = polygons
                    .iter()
                    .map(|span| {
                        let point_span = get_slice(*span, points, "point")?;

                        Ok(geom::Polygon::from_ccw(point_span
                            .into_iter()
                            .map(|v| Into::<Vec3f>::into(*v))
                            .collect()
                        ))
                    })
                    .collect::<Result<Vec<_>, MapLoadingError>>()?;

                polygons
            },

            material_name_set: {
                let names = material_names
                    .iter()
                    .map(|span| {
                        let chars = get_slice(*span, chars, "material name")?;
                        let str = std::str::from_utf8(chars)?;

                        Ok(str.to_string())
                    })
                    .collect::<Result<Vec<_>, MapLoadingError>>()?;

                names
            },
            volume_set: {
                let volumes = volumes
                    .iter()
                    .map(|volume| {
                        let portal_span = get_slice(volume.portals, volume_portals, "volume portal")?;
                        let surface_span = get_slice(volume.surfaces, volume_surfaces, "volume surface")?;

                        Ok(Volume {
                            portals: portal_span
                                .iter()
                                .map(|portal| {
                                    // validate volume index
                                    Ok(Portal {
                                        dst_volume_id: get_id(portal.dst_volume_index, volumes, "volume")?,
                                        is_facing_front: portal.is_facing_front != 0,
                                        polygon_id: get_id(portal.polygon_index, polygons, "polygon")?,
                                    })
                                })
                                .collect::<Result<Vec<_>, MapLoadingError>>()?,
                            surfaces: surface_span
                                .iter()
                                .map(|surface| {
                                    Ok(Surface {
                                        material_id: get_id(surface.material_index, material_names, "material")?,
                                        polygon_id: get_id(surface.polygon_index, polygons, "polygon")?,
                                        is_sky: surface.is_sky != 0,
                                        is_transparent: surface.is_transparent != 0,
                                        u: surface.u.into(),
                                        v: surface.v.into(),
                                    })
                                })
                                .collect::<Result<Vec<_>, MapLoadingError>>()?,
                            bound_box: geom::BoundBox::new(
                                volume.bound_box_min.into(),
                                volume.bound_box_max.into(),
                            ),
                        })
                    })
                    .collect::<Result<Vec<_>, MapLoadingError>>()?
                    ;

                volumes
            },
            bsp_models: {
                let bsp_models = bsp_models
                    .iter()
                    .map(|model| {
                        Ok(BspModel {
                            bound_box: geom::BoundBox::new(
                                model.bound_box_min.into(),
                                model.bound_box_max.into(),
                            ),
                            bsp: bsp_from(bsp_elements, volumes, model.bsp_root_index)?.0,
                        })
                    })
                    .collect::<Result<Vec<_>, MapLoadingError>>()?;

                bsp_models
            },
            world_model_id: get_id(header.world_bsp_model_index, bsp_models, "bsp model")?,
            dynamic_models: {
                let models = dynamic_models
                    .iter()
                    .map(|model| {
                        Ok(DynamicModel {
                            model_id: get_id(model.bsp_model_index, bsp_models, "bsp model")?,
                            origin: model.origin.into(),
                            rotation: model.rotation
                        })
                    })
                    .collect::<Result<Vec<_>, MapLoadingError>>()?;

                models
            },
        };

        Ok(map)
    }

    /// Save map
    pub fn save(&self, dst: &mut dyn std::io::Write) -> Result<(), MapSavingError> {
        fn write_bsp(dst: &mut Vec<wbsp::BspElement>, bsp: &Bsp) {
            match bsp {
                Bsp::Partition {
                    splitter_plane,
                    front,
                    back
                } => {
                    dst.push(wbsp::BspElement {
                        ty: wbsp::BspType::Partition as u8,
                        data: wbsp::BspElementData {
                            partition: wbsp::BspPartition {
                                plane: (*splitter_plane).into(),
                            },
                        }
                    });
                    write_bsp(dst, front);
                    write_bsp(dst, back);
                }
                Bsp::Volume(id) => {
                    dst.push(wbsp::BspElement {
                        ty: wbsp::BspType::Volume as u8,
                        data: wbsp::BspElementData {
                            volume: wbsp::BspVolume {
                                volume_index: id.into_index() as u32,
                            },
                        },
                    });
                }
                Bsp::Void => {
                    dst.push(wbsp::BspElement {
                        ty: wbsp::BspType::Void as u8,
                        data: wbsp::BspElementData {
                            void: ()
                        },
                    });
                }
            }
        }

        // Precalculate chunks
        let mut chars = Vec::<u8>::new();
        let mut points = Vec::<wbsp::Vec3>::new();
        let mut polygons = Vec::<wbsp::Span>::new();
        let mut material_names = Vec::<wbsp::Span>::new();
        let mut volume_surfaces = Vec::<wbsp::Surface>::new();
        let mut volume_portals = Vec::<wbsp::Portal>::new();
        let mut volumes = Vec::<wbsp::Volume>::new();
        let mut bsp_elements = Vec::<wbsp::BspElement>::new();
        let mut bsp_models = Vec::<wbsp::BspModel>::new();
        let mut dynamic_models = Vec::<wbsp::DynamicModel>::new();

        for name in &self.material_name_set {
            material_names.push(wbsp::Span {
                offset: chars.len() as u32,
                size: name.len() as u32,
            });

            chars.extend_from_slice(name.as_bytes());
        }

        for polygon in &self.polygon_set {
            polygons.push(wbsp::Span {
                offset: points.len() as u32,
                size: polygon.points.len() as u32,
            });

            for pt in &polygon.points {
                points.push((*pt).into());
            }
        }

        for volume in &self.volume_set {
            volumes.push(wbsp::Volume {
                portals: wbsp::Span {
                    offset: volume_portals.len() as u32,
                    size: volume.portals.len() as u32,
                },
                surfaces: wbsp::Span {
                    offset: volume_surfaces.len() as u32,
                    size: volume.surfaces.len() as u32,
                },
                bound_box_min: volume.bound_box.min().into(),
                bound_box_max: volume.bound_box.max().into(),
            });

            for surface in &volume.surfaces {
                volume_surfaces.push(wbsp::Surface {
                    material_index: surface.material_id.into_index() as u32,
                    polygon_index: surface.polygon_id.into_index() as u32,
                    u: surface.u.into(),
                    v: surface.v.into(),
                    is_transparent: surface.is_transparent as u8,
                    is_sky: surface.is_sky as u8,
                });
            }

            for portal in &volume.portals {
                volume_portals.push(wbsp::Portal {
                    polygon_index: portal.polygon_id.into_index() as u32,
                    dst_volume_index: portal.dst_volume_id.into_index() as u32,
                    is_facing_front: portal.is_facing_front as u8,
                });
            }
        }

        for model in &self.bsp_models {
            let bsp_root_index = bsp_elements.len() as u32;

            write_bsp(&mut bsp_elements, &model.bsp);

            bsp_models.push(wbsp::BspModel {
                bsp_root_index,
                bound_box_min: model.bound_box.min().into(),
                bound_box_max: model.bound_box.max().into(),
            });
        }

        for dynamic_model in &self.dynamic_models {
            dynamic_models.push(wbsp::DynamicModel {
                bsp_model_index: dynamic_model.model_id.into_index() as u32,
                origin: dynamic_model.origin.into(),
                rotation: dynamic_model.rotation,
            });
        }

        let mut header = wbsp::Header {
            magic: wbsp::MAGIC,
            world_bsp_model_index: self.world_model_id.into_index() as u32,

            chars: wbsp::Span::zero(),
            points: wbsp::Span::zero(),
            polygons: wbsp::Span::zero(),
            material_names: wbsp::Span::zero(),
            volume_surfaces: wbsp::Span::zero(),
            volume_portals: wbsp::Span::zero(),
            volumes: wbsp::Span::zero(),
            bsp_elements: wbsp::Span::zero(),
            bsp_models: wbsp::Span::zero(),
            dynamic_models: wbsp::Span::zero(),
        };

        let mut infos: [(&mut wbsp::Span, &[u8]); 10] = [
            (&mut header.chars, bytemuck::cast_slice::<_, u8>(&chars)),
            (&mut header.points, bytemuck::cast_slice::<_, u8>(&points)),
            (&mut header.polygons, bytemuck::cast_slice::<_, u8>(&polygons)),
            (&mut header.material_names, bytemuck::cast_slice::<_, u8>(&material_names)),
            (&mut header.volume_surfaces, bytemuck::cast_slice::<_, u8>(&volume_surfaces)),
            (&mut header.volume_portals, bytemuck::cast_slice::<_, u8>(&volume_portals)),
            (&mut header.volumes, bytemuck::cast_slice::<_, u8>(&volumes)),
            (&mut header.bsp_elements, bytemuck::cast_slice::<_, u8>(&bsp_elements)),
            (&mut header.bsp_models, bytemuck::cast_slice::<_, u8>(&bsp_models)),
            (&mut header.dynamic_models, bytemuck::cast_slice::<_, u8>(&dynamic_models)),
        ];

        let mut offset = std::mem::size_of::<wbsp::Header>();
        for (span, vec) in &mut infos {
            span.offset = offset as u32;
            span.size = vec.len() as u32;
            offset += vec.len();
        }

        let total_length = offset - std::mem::size_of::<wbsp::Header>();
        let write_info = infos.map(|(_, v)| std::io::IoSlice::new(v));

        let count = dst.write(bytemuck::bytes_of(&header))?;
        if count != std::mem::size_of::<wbsp::Header>() {
            return Err(MapSavingError::IncompleteWrite);
        }

        let count = dst.write_vectored(&write_info)?;
        if count != total_length {
            return Err(MapSavingError::IncompleteWrite);
        }

        Ok(())
    }
}

// mod.rs
