//! WBSP file format implementation module

use zerocopy::{FromBytes, Immutable, IntoBytes};
use thiserror::Error;
use crate::{bsp::Id, geom, math::Vec2};

/// .WBSP file Magic number
pub const MAGIC: u32 = u32::from_le_bytes(*b"WBSP");

/// Some span
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, Default, FromBytes, IntoBytes, Immutable)]
pub struct Span {
    /// String offset
    pub offset: u32,

    /// String size
    pub size: u32,
}

impl Span {
    /// Construct zero-sized span
    pub const fn zero() -> Self {
        Self { offset: 0, size: 0 }
    }

    /// Convert span to usize range
    pub const fn range(self) -> std::ops::Range<usize> {
        std::ops::Range::<usize> {
            start: self.offset as usize,
            end: self.offset as usize + self.size as usize,
        }
    }

    /// Construct span from range
    pub const fn from_range(range: std::ops::Range<usize>) -> Self {
        Self {
            offset: range.start as u32,
            size: (range.end - range.start) as u32,
        }
    }
}

/// File header structure
#[repr(C)]
#[derive(Copy, Clone, Default, FromBytes, IntoBytes, Immutable)]
pub struct Header {
    /// WBSP file magic
    pub magic: u32,

    /// Index of world (e.g. root) model in BSP model list
    pub world_bsp_model_index: u32,

    /// Span that holds all string characters (String are just indices in the span data)
    pub chars: Span,

    /// Lightmap data span
    pub lightmaps: Span,

    /// Span containing (unique) points referenced by polygons
    pub points: Span,

    /// Span of polygons
    pub polygons: Span,

    /// Material name spans
    pub material_names: Span,

    /// Span containing volume (displayed) surfaces
    pub volume_surfaces: Span,

    /// Span containing volume-to-volume portals
    pub volume_portals: Span,

    /// Span containing volumes (e.g. BSP leafs)
    pub volumes: Span,

    /// BSP element span
    pub bsp_elements: Span,

    /// BSP model span
    pub bsp_models: Span,

    /// Dynamic BSP models
    pub dynamic_models: Span,
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes)]
pub struct Material {
    /// Span of name in chars section
    pub name: Span,
}

/// Descriptor of lightmap texture reference. Lightaps are stored in U64s. Lightmaps with ~0 offset considered not present.
#[repr(C)]
#[derive(Copy, Clone, Default, FromBytes, IntoBytes, Immutable)]
pub struct SurfaceLightmap {
    /// Lightmap array offsef
    pub offset: u32,

    /// Image width (pixels)
    pub width: u32,

    /// Image height (pixels)
    pub height: u32,

    /// Image UV minimum
    pub uv_min: [i32; 2],

    /// Image UV maximum
    pub uv_max: [i32; 2],
}

impl SurfaceLightmap {
    /// Non-present lightmap
    pub fn not_present() -> Self {
        Self {
            offset: !0,
            ..Default::default()
        }
    }

    /// Check lightmap for being present
    pub const fn is_present(&self) -> bool {
        self.offset != !0
    }

    /// Get lightmap data flat span
    pub const fn as_span(&self) -> Option<Span> {
        if !self.is_present() {
            return None;
        }

        Some(Span {
            offset: self.offset,
            size: self.width * self.height,
        })
    }
}

/// Visible volume face piece
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct Surface {
    /// Index of surface material in material set
    pub material_index: u32,

    /// Index of surface polygon in polygon set
    pub polygon_index: u32,

    /// U axis
    pub u: Plane,

    /// V axis
    pub v: Plane,

    /// Surface lightmap descriptor.
    pub lightmap: SurfaceLightmap,

    /// Surface flags (matches [`SurfaceFlags`][super::SurfaceFlags] data)
    pub flags: u32,
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct Plane {
    /// Plane normal vector
    pub normal: [f32; 3],

    /// Distance from point to plane
    pub distance: f32,
}

impl From<Plane> for geom::Plane {
    fn from(p: Plane) -> geom::Plane {
        geom::Plane { distance: p.distance, normal: p.normal.into(), }
    }
}

impl From<geom::Plane> for Plane {
    fn from(value: geom::Plane) -> Self {
        Self { distance: value.distance, normal: value.normal.into() }
    }
}

/// Portal (volume-volume connection)
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct Portal {
    /// Index of portal polygon in polygon set
    pub polygon_index: u32,

    /// Index of destination volume in volume set
    pub dst_volume_index: u32,

    /// If true, portal polygon is faced towards camera
    pub is_facing_front: u8,

    /// Padding bytes
    pub _pad: [u8; 3],
}

/// Volume (BSP leaf) structure
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct Volume {
    /// Subspan of surface span containing volumes that belongs to this volume
    pub surfaces: Span,

    /// Subspan of surface span containing portals that belongs to this volume
    pub portals: Span,

    /// Volume bound box minimum
    pub bound_box_min: [f32; 3],

    /// Volume bound box maximum
    pub bound_box_max: [f32; 3],
}

/// BSP entry type
#[repr(C)]
pub enum BspType {
    /// Partitions, BspPartition strucutre and two Bsp's (left and right children) are written after
    Partition = 1,

    /// Volume, BspVolume strucutre is written after
    Volume = 2,

    /// Void, nothing is written after
    Void = 3,
}

impl TryFrom<u8> for BspType {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Partition,
            2 => Self::Volume,
            3 => Self::Void,
            _ => return Err(value),
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspElement {
    /// Element type byte
    pub ty: u8,

    /// Padding bytes
    pub _pad: [u8; 3],

    /// Element contents
    pub data: BspElementData,
}

/// BSP element data
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub union BspElementData {
    /// Volume contents
    pub volume: BspElementVolume,

    /// Partition contents
    pub partition: BspPartition,

    /// Void contents
    pub void: BspElementVoid,
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspElementVolume {
    /// Exact volume
    pub volume: BspVolume,

    /// Default padding
    pub _pad: [u8; 12],
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspElementVoid {
    /// Void element
    pub _pad: [u8; 16],
}

/// BSP helper structure
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspPartition {
    /// Partition plane's distance
    pub plane: Plane,
}

/// BSP helper structure
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspVolume {
    /// Destination volume index
    pub volume_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct BspModel {
    /// BSP head offset
    pub bsp_root_index: u32,

    /// Bounding box minimum
    pub bound_box_min: [f32; 3],

    /// Bounding box maximum
    pub bound_box_max: [f32; 3],
}

/// Dynamic model (moving stuff, like doors etc.)
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct DynamicModel {
    /// BSP model index
    pub bsp_model_index: u32,

    /// Position
    pub origin: [f32; 3],

    /// Rotation
    pub rotation: f32,
}




// Implementation




/// Map from file loading error
#[derive(Debug, Error)]
pub enum LoadError {
    /// Input/Output error
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error during building string from UTF-8 byte array
    #[error("UTF8 string building error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),

    /// Invalid magic value
    #[error("Invalid WBSP magic: 0x{0:08X}, 0x{a:08X} expected", a=MAGIC)]
    InvalidMagic(u32),

    /// Invalid BSP type
    #[error("0x{0:02X} is not valid BSP type.")]
    InvalidBspType(u8),

    /// Invalid span occured
    #[error("Invalid span")]
    InvalidSpan {
        span_kind: &'static str,
        invalid_reason: &'static str,
        buffer_size: usize,
        span: Span,
    },

    /// Invalid index
    #[error("Invalid index")]
    InvalidIndex {
        kind: &'static str,
        count: usize,
        index: u32,
    },
}

/// Load .WBSP into map
pub fn load(data: &[u8]) -> Result<super::Map, LoadError> {
    // Read typed span
    pub fn load_read_tspan<T: Copy + Clone + FromBytes + Immutable>(
        data: &[u8],
        span: Span
    ) -> Result<&[T], LoadError> {
        if std::mem::size_of::<T>() == 0 {
            // Empty slice of nothing
            return Ok(&[]);
        }

        let elem_count = span.size as usize / std::mem::size_of::<T>();
        let offset = span.offset as usize;

        let slice = data
            .get(offset..offset + elem_count * std::mem::size_of::<T>())
            .ok_or(LoadError::InvalidSpan {
                span_kind: "main",
                invalid_reason: "span overflow",
                buffer_size: data.len(),
                span
            })?;

        <[T]>::ref_from_prefix_with_elems(slice, elem_count)
            .map(|v| v.0)
            .map_err(|e| {
                eprint!("Zerocopy error: {}", e);

                LoadError::InvalidSpan {
                    span_kind: "header",
                    invalid_reason: "zerocopy ref_from_prefix error",
                    buffer_size: data.len(),
                    span,
                }
            })
    }

    fn get_id<T: super::Id, E>(index: u32, arr: &[E], kind: &'static str) -> Result<T, LoadError> {
        if index as usize >= arr.len() {
            return Err(LoadError::InvalidIndex {
                kind,
                count: arr.len(),
                index
            });
        }

        Ok(T::from_index(index as usize))
    }

    /// Build BSP starting from kind of array
    fn bsp_from(elems: &[BspElement], volumes: &[Volume], start: u32) -> Result<(Box<super::Bsp>, u32), LoadError> {
        let elem = elems
            .get(start as usize)
            .ok_or(LoadError::InvalidIndex {
                kind: "bsp element",
                count: elems.len(),
                index: start,
            })?;

        let ty = match BspType::try_from(elem.ty) {
            Ok(ty) => ty,
            Err(err) => {
                return Err(LoadError::InvalidBspType(err));
            }
        };

        Ok(match ty {
            BspType::Partition => {
                let partition = unsafe { &elem.data.partition };
                let splitter_plane: geom::Plane = partition.plane.into();

                let (front, front_end) = bsp_from(elems, volumes, start + 1)?;
                let (back, back_end) = bsp_from(elems, volumes, front_end)?;

                (Box::new(super::Bsp::Partition { splitter_plane, front, back }), back_end)
            }
            BspType::Volume => {
                let volume = unsafe { &elem.data.volume };
                let id: super::VolumeId = get_id(volume.volume.volume_index, volumes, "volume")?;

                (Box::new(super::Bsp::Volume(id)), start + 1)
            }
            BspType::Void => {
                (Box::new(super::Bsp::Void), start + 1)
            }
        })
    }

    fn get_slice<'t, E>(
        span: Span,
        arr: &'t [E],
        kind: &'static str,
    ) -> Result<&'t [E], LoadError> {
        arr
            .get(span.range())
            .ok_or(LoadError::InvalidSpan {
                span_kind: kind,
                invalid_reason: "span overflow",
                buffer_size: arr.len(),
                span,
            })
    }

    let header = load_read_tspan::<Header>(
        data,
        Span { offset: 0, size: std::mem::size_of::<Header>() as u32 }
    )?.first().unwrap();

    // Check file magic
    if header.magic != MAGIC {
        return Err(LoadError::InvalidMagic(header.magic));
    }

    macro_rules! load_head_span {
        ($name: ident, $type: ty) => {
            let $name = load_read_tspan::<$type>(&data, header.$name)?;
        };
    }

    // Load spans stored in header
    load_head_span!(chars,           u8           );
    load_head_span!(lightmaps,       u64          );
    load_head_span!(points,          [f32; 3]     );
    load_head_span!(polygons,        Span         );
    load_head_span!(material_names,  Span         );
    load_head_span!(volume_surfaces, Surface      );
    load_head_span!(volume_portals,  Portal       );
    load_head_span!(volumes,         Volume       );
    load_head_span!(bsp_elements,    BspElement   );
    load_head_span!(bsp_models,      BspModel     );
    load_head_span!(dynamic_models,  DynamicModel );

    let map_polygon_set = polygons
        .iter()
        .map(|span| {
            let point_span = get_slice(*span, points, "point")?;

            Ok(geom::Polygon::from_ccw(point_span
                .iter()
                .map(|v| Into::<crate::math::Vec3f>::into(*v))
                .collect()
            ))
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let map_material_name_set = material_names
        .iter()
        .map(|span| {
            let chars = get_slice(*span, chars, "material name")?;
            let str = std::str::from_utf8(chars)?;

            Ok(str.to_string())
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let map_volume_set = volumes
        .iter()
        .map(|volume| {
            let portal_span = get_slice(volume.portals, volume_portals, "volume portal")?;
            let surface_span = get_slice(volume.surfaces, volume_surfaces, "volume surface")?;

            Ok(super::Volume {
                portals: portal_span
                    .iter()
                    .map(|portal| {
                        // validate volume index
                        Ok(super::Portal {
                            dst_volume_id: get_id(portal.dst_volume_index, volumes, "volume")?,
                            is_facing_front: portal.is_facing_front != 0,
                            polygon_id: get_id(portal.polygon_index, polygons, "polygon")?,
                        })
                    })
                    .collect::<Result<Vec<_>, LoadError>>()?,
                surfaces: surface_span
                    .iter()
                    .map(|surface| {
                        // Calculate volume set
                        let u = surface.u.into();
                        let v = surface.v.into();
                        let polygon_id: super::PolygonId = get_id(surface.polygon_index, polygons, "polygon")?;

                        // Extract lightmap from map data
                        let lightmap = if let Some(span) = surface.lightmap.as_span() {
                            let data = get_slice(span, lightmaps, "lightmap")?;

                            Some(super::SurfaceLightmap {
                                width: surface.lightmap.width as usize,
                                height: surface.lightmap.height as usize,
                                data: data.to_vec().into_boxed_slice(),
                                uv_min: Vec2::from_array(surface.lightmap.uv_min).map(|i| i as isize),
                                uv_max: Vec2::from_array(surface.lightmap.uv_max).map(|i| i as isize),
                            })
                        } else {
                            None
                        };

                        Ok(super::Surface {
                            material_id: get_id(surface.material_index, material_names, "material")?,
                            polygon_id,
                            flags: super::SurfaceFlags(surface.flags as u8),
                            lightmap,
                            u,
                            v,
                        })
                    })
                    .collect::<Result<Vec<_>, LoadError>>()?,
                bound_box: geom::BoundBox::new(
                    volume.bound_box_min.into(),
                    volume.bound_box_max.into(),
                ),
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let map = super::Map {
        polygon_set: map_polygon_set,
        material_name_set: map_material_name_set,
        volume_set: map_volume_set,

        bsp_models: bsp_models
            .iter()
            .map(|model| {
                Ok(super::BspModel {
                    bound_box: geom::BoundBox::new(
                        model.bound_box_min.into(),
                        model.bound_box_max.into(),
                    ),
                    bsp: bsp_from(bsp_elements, volumes, model.bsp_root_index)?.0,
                })
            })
            .collect::<Result<Vec<_>, LoadError>>()?,
        world_model_id: get_id(header.world_bsp_model_index, bsp_models, "bsp model")?,
        dynamic_models: dynamic_models
            .iter()
            .map(|model| {
                Ok(super::DynamicModel {
                    model_id: get_id(model.bsp_model_index, bsp_models, "bsp model")?,
                    origin: model.origin.into(),
                    rotation: model.rotation
                })
            })
            .collect::<Result<Vec<_>, LoadError>>()?,
    };

    Ok(map)
}

/// Error occured during map saving
#[derive(Debug, Error)]
pub enum SaveError {
    /// IO error occured
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    /// Incomplete map write
    #[error("Incomplete write")]
    IncompleteWrite,
}

/// Save .WBSP to map
pub fn save(map: &super::Map, dst: &mut dyn std::io::Write) -> Result<(), SaveError> {
    fn write_bsp(dst: &mut Vec<BspElement>, bsp: &super::Bsp) {
        match bsp {
            super::Bsp::Partition {
                splitter_plane,
                front,
                back
            } => {
                dst.push(BspElement {
                    ty: BspType::Partition as u8,
                    _pad: [0; _],
                    data: BspElementData {
                        partition: BspPartition {
                            plane: (*splitter_plane).into(),
                        },
                    }
                });
                write_bsp(dst, front);
                write_bsp(dst, back);
            }
            super::Bsp::Volume(id) => {
                dst.push(BspElement {
                    ty: BspType::Volume as u8,
                    _pad: [0; _],
                    data: BspElementData {
                        volume: BspElementVolume {
                            volume: BspVolume {
                                volume_index: id.into_index() as u32,
                            },
                            _pad: [0; _],
                        }
                    },
                });
            }
            super::Bsp::Void => {
                dst.push(BspElement {
                    ty: BspType::Void as u8,
                    _pad: [0; _],
                    data: BspElementData {
                        void: BspElementVoid {
                            _pad: [0; _],
                        }
                    },
                });
            }
        }
    }

    // Precalculate chunks
    let mut chars = Vec::<u8>::new();
    let mut lightmaps = Vec::<u64>::new();
    let mut points = Vec::<[f32; 3]>::new();
    let mut polygons = Vec::<Span>::new();
    let mut material_names = Vec::<Span>::new();
    let mut volume_surfaces = Vec::<Surface>::new();
    let mut volume_portals = Vec::<Portal>::new();
    let mut volumes = Vec::<Volume>::new();
    let mut bsp_elements = Vec::<BspElement>::new();
    let mut bsp_models = Vec::<BspModel>::new();
    let mut dynamic_models = Vec::<DynamicModel>::new();

    for name in &map.material_name_set {
        material_names.push(Span {
            offset: chars.len() as u32,
            size: name.len() as u32,
        });

        chars.extend_from_slice(name.as_bytes());
    }

    for polygon in &map.polygon_set {
        polygons.push(Span {
            offset: points.len() as u32,
            size: polygon.points.len() as u32,
        });

        for pt in &polygon.points {
            points.push((*pt).into());
        }
    }

    for volume in &map.volume_set {
        volumes.push(Volume {
            portals: Span {
                offset: volume_portals.len() as u32,
                size: volume.portals.len() as u32,
            },
            surfaces: Span {
                offset: volume_surfaces.len() as u32,
                size: volume.surfaces.len() as u32,
            },
            bound_box_min: volume.bound_box.min().into(),
            bound_box_max: volume.bound_box.max().into(),
        });

        for surface in &volume.surfaces {

            // Save surface lightmap
            let lightmap = if let Some(lightmap) = surface.lightmap.as_ref() {
                let offset = lightmaps.len();
                lightmaps.extend_from_slice(&lightmap.data);

                // Build lightmap strcture
                SurfaceLightmap {
                    offset: offset as u32,
                    width: lightmap.width as u32,
                    height: lightmap.height as u32,
                    uv_min: lightmap.uv_min.map(|i| i as i32).into_array(),
                    uv_max: lightmap.uv_max.map(|i| i as i32).into_array(),
                }
            } else {
                SurfaceLightmap::not_present()
            };

            volume_surfaces.push(Surface {
                material_index: surface.material_id.into_index() as u32,
                polygon_index: surface.polygon_id.into_index() as u32,
                u: surface.u.into(),
                v: surface.v.into(),
                lightmap,
                flags: surface.flags.0 as u32,
            });
        }

        for portal in &volume.portals {
            volume_portals.push(Portal {
                polygon_index: portal.polygon_id.into_index() as u32,
                dst_volume_index: portal.dst_volume_id.into_index() as u32,
                is_facing_front: portal.is_facing_front as u8,
                _pad: [0; _],
            });
        }
    }

    for model in &map.bsp_models {
        let bsp_root_index = bsp_elements.len() as u32;

        write_bsp(&mut bsp_elements, &model.bsp);

        bsp_models.push(BspModel {
            bsp_root_index,
            bound_box_min: model.bound_box.min().into(),
            bound_box_max: model.bound_box.max().into(),
        });
    }

    for dynamic_model in &map.dynamic_models {
        dynamic_models.push(DynamicModel {
            bsp_model_index: dynamic_model.model_id.into_index() as u32,
            origin: dynamic_model.origin.into(),
            rotation: dynamic_model.rotation,
        });
    }

    // Construct file header
    let mut header = Header {
        magic: MAGIC,
        world_bsp_model_index: map.world_model_id.into_index() as u32,
        ..Default::default()
    };

    let mut infos: [(&mut Span, &[u8]); 11] = [
        (&mut header.chars,           chars.as_bytes()),
        (&mut header.lightmaps,       lightmaps.as_bytes()),
        (&mut header.points,          points.as_bytes()),
        (&mut header.polygons,        polygons.as_bytes()),
        (&mut header.material_names,  material_names.as_bytes()),
        (&mut header.volume_surfaces, volume_surfaces.as_bytes()),
        (&mut header.volume_portals,  volume_portals.as_bytes()),
        (&mut header.volumes,         volumes.as_bytes()),
        (&mut header.bsp_elements,    bsp_elements.as_bytes()),
        (&mut header.bsp_models,      bsp_models.as_bytes()),
        (&mut header.dynamic_models,  dynamic_models.as_bytes()),
    ];

    // Bytes used for padding writes
    let pad_bytes = [0u8; 16];

    fn align16(n: usize) -> usize { (n / 16 + !n.is_multiple_of(16) as usize) * 16 }
    let pad_for = |n| &pad_bytes[..align16(n) - n];

    let mut write_infos = Vec::with_capacity(2 + infos.len() * 2);

    // Insert two dummy write infos
    write_infos.push(std::io::IoSlice::new(&[]));
    write_infos.push(std::io::IoSlice::new(&[]));

    // Calculate sizes and paddings
    let mut offset = align16(std::mem::size_of::<Header>());
    for (span, vec) in &mut infos {
        span.offset = offset as u32;
        span.size = vec.len() as u32;
        offset += align16(vec.len());

        write_infos.push(std::io::IoSlice::new(vec));
        write_infos.push(std::io::IoSlice::new(pad_for(vec.len())));
    }

    // Update header and header padding slices
    write_infos[0] = std::io::IoSlice::new(header.as_bytes());
    write_infos[1] = std::io::IoSlice::new(pad_for(std::mem::size_of::<Header>()));

    // Perform write!
    if dst.write_vectored(&write_infos)? != offset {
        return Err(SaveError::IncompleteWrite);
    }

    Ok(())
}
