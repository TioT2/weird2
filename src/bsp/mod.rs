use bytemuck::Zeroable;
///! Main project BSP structure declaration module

pub use compiler::Error as MapCompilationError;
use std::num::NonZeroU32;
use crate::{geom, map, math::Vec3f};

/// Declare actual map builder module
pub mod compiler;

/// WBSP description module
pub mod wbsp;

/// Generic id implementation
macro_rules! impl_id {
    ($name: ident) => {
        /// Unique identifier
        #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Ord, PartialOrd)]
        pub struct $name(NonZeroU32);

        impl $name {
            /// Build id from index
            pub fn from_index(index: usize) -> Self {
                $name(NonZeroU32::try_from(index as u32 + 1).unwrap())
            }

            /// Get index by id
            pub fn into_index(self) -> usize {
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
    pub origin: Vec3f,

    /// Model rotation (along Y axis)
    pub rotation: f32,

    /// Corresponding BSP model Id
    pub model_id: BspModelId,
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
        (0..self.volume_set.len()).map(DynamicModelId::from_index)
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
    StringFromUtf8Error(std::string::FromUtf8Error),

    /// Invalid magic value
    InvalidMagic(u32),

    /// Invalid span occured
    InvalidSpan {
        kind: &'static str,
        buffer_size: u32,
        offset: u32,
        size: u32,
    },

    /// 
    PolygonLengthsSumMoreThanPolygonCount {
        polygon_point_count: u32,
        polygon_length_sum: u32,
    },

    ShortMaterialNameByteSet {
        name_length_sum: u32,
        name_bytes_count: u32,
    },

    ShortVolumePortalSet {
        volume_portal_length_sum: u32,
        volume_portal_count: u32,
    },

    ShortVolumeSurfaceSet {
        volume_surface_length_sum: u32,
        volume_surface_count: u32,
    },

    /// Invalid BSP type
    InvalidBspType(u32),

    /// Invalid volume index
    InvalidVolumeIndex {
        /// Volume set length
        volume_count: u32,

        /// Volume index
        volume_index: u32,
    },

    /// Invalid index of polygon occured in structure
    InvalidPolygonIndex {
        /// Polygon count
        polygon_count: u32,

        /// Polygon index
        polygon_index: u32,
    },

    /// Invalid index of material
    InvalidMaterialIndex {
        /// Material count
        material_count: u32,

        /// Material index
        material_index: u32,
    },
}

impl From<std::io::Error> for MapLoadingError {
    fn from(value: std::io::Error) -> Self {
        MapLoadingError::IoError(value)
    }
}

impl Map {
    /// Compile map to WBSP
    pub fn compile(map: &map::Map) -> Result<Self, MapCompilationError> {
        compiler::compile(map)
    }
}

// Load/Save implementation
#[cfg(not)]
impl Map {
    /// Load map from WBSP file
    pub fn load(src: &mut dyn std::io::Read) -> Result<Self, MapLoadingError> {
        let mut header = wbsp::Header::zeroed();

        src.read(bytemuck::bytes_of_mut(&mut header))?;

        if header.magic != wbsp::MAGIC {
            return Err(MapLoadingError::InvalidMagic(header.magic));
        }

        /*
        /// Volume total count
        pub volume_count: u32,
        */

        fn read_vec<T: bytemuck::AnyBitPattern + bytemuck::Zeroable + bytemuck::NoUninit + Clone>(
            src: &mut dyn std::io::Read,
            count: u32
        ) -> Result<Vec<T>, std::io::Error> {
            let mut result = Vec::<T>::new();

            result.resize(count as usize, T::zeroed());
            src.read(bytemuck::cast_slice_mut(result.as_mut_slice()))?;

            Ok(result)
        }

        let polygon_lengths = read_vec::<u32>(src, header.polygon_length_count)?;
        let polygon_points = read_vec::<wbsp::Vec3>(src, header.polygon_point_count)?;
        let material_name_chars = read_vec::<u8>(src, header.string_buffer_length)?;
        let material_name_spans = read_vec::<wbsp::Span>(src, header.material_name_span_count)?;
        let volume_portals = read_vec::<wbsp::Portal>(src, header.volume_portal_count)?;
        let volume_surfaces = read_vec::<wbsp::Surface>(src, header.volume_surface_count)?;
        let volumes = read_vec::<wbsp::Volume>(src, header.volume_count)?;
        let bsp_elements = read_vec::<wbsp::BspElement>(src, header.bsp_element_count)?;

        let bsp = {
            fn read_bsp(
                src: &mut dyn std::io::Read,
                volume_count: usize
            ) -> Result<Box<Bsp>, MapLoadingError> {
                let mut bsp_type_u32 = 0u32;
                src.read(bytemuck::bytes_of_mut(&mut bsp_type_u32))?;

                let bsp_type = match wbsp::BspType::try_from(bsp_type_u32) {
                    Ok(t) => t,
                    Err(_) => return Err(MapLoadingError::InvalidBspType(bsp_type_u32))
                };

                Ok(Box::new(match bsp_type {
                    wbsp::BspType::Partition => {
                        let mut partition = wbsp::BspPartition::zeroed();

                        src.read(bytemuck::bytes_of_mut(&mut partition))?;

                        let front = read_bsp(src, volume_count)?;
                        let back = read_bsp(src, volume_count)?;

                        Bsp::Partition {
                            splitter_plane: geom::Plane {
                                distance: partition.plane_distance,
                                normal: Vec3f::new(
                                    partition.plane_normal.x,
                                    partition.plane_normal.y,
                                    partition.plane_normal.z,
                                ),
                            },
                            front,
                            back
                        }
                    }
                    wbsp::BspType::Volume => {
                        let mut volume_index = 0u32;
                        src.read(bytemuck::bytes_of_mut(&mut volume_index))?;

                        if volume_index as usize >= volume_count {
                            return Err(MapLoadingError::InvalidVolumeIndex {
                                volume_count: volume_count as u32,
                                volume_index,
                            });
                        }

                        Bsp::Volume(VolumeId::from_index(volume_index as usize))
                    }
                    wbsp::BspType::Void => {
                        Bsp::Void
                    }
                }))
            }

            read_bsp(src, volumes.len())?
        };

        // Build polygon set
        let polygon_set = {
            // Ensure that polygon_length.sum == polygon_points.len
            let polygon_length_sum = polygon_lengths.iter().sum::<u32>();
    
            if polygon_length_sum as usize > polygon_points.len() {
                return Err(MapLoadingError::PolygonLengthsSumMoreThanPolygonCount {
                    polygon_point_count: polygon_points.len() as u32,
                    polygon_length_sum,
                });
            }

            let mut polygons = Vec::<geom::Polygon>::with_capacity(polygon_lengths.len());
    
            let mut offset = 0usize;
            for length in polygon_lengths.iter().copied() {
                let points = polygon_points
                    .get(offset..offset + length as usize)
                    .unwrap()
                    .iter()
                    .map(|v| Vec3f { x: v.x, y: v.y, z: v.z })
                    .collect::<Vec<_>>();
                offset += length as usize;

                polygons.push(geom::Polygon::from_ccw(points));
            }

            polygons
        };

        let material_name_set = {
            material_name_spans
                .iter()
                .map(|span| {
                    let name = material_name_chars
                        .get(span.offset as usize..(span.offset + span.size) as usize)
                        .ok_or(MapLoadingError::InvalidSpan {
                            kind: "material name span",
                            buffer_size: material_name_chars.len() as u32,
                            offset: span.offset,
                            size: span.size,
                        })?;

                    String::from_utf8(name.to_vec())
                        .map_err(MapLoadingError::StringFromUtf8Error)
                })
                .collect::<Result<Vec<_>, _>>()
                ?
        };

        let volume_set = {
            // Check correctness of portal lengths
            let volume_portal_length_sum = volumes
                .iter()
                .map(|volume| volume.portal_count)
                .sum::<u32>();

            if volume_portal_length_sum > header.volume_portal_count {
                return Err(MapLoadingError::ShortVolumePortalSet {
                    volume_portal_length_sum,
                    volume_portal_count: header.volume_portal_count
                });
            }

            let volume_surface_length_sum = volumes
                .iter()
                .map(|volume| volume.surface_count)
                .sum::<u32>();

            if volume_surface_length_sum > header.volume_surface_count {
                return Err(MapLoadingError::ShortVolumeSurfaceSet {
                    volume_surface_length_sum,
                    volume_surface_count: header.volume_surface_count
                });
            }

            let mut result = Vec::<Volume>::new();

            let mut portal_offset = 0usize;
            let mut surface_offset = 0usize;

            for volume in &volumes {
                let mut portals = Vec::new();
                let mut surfaces = Vec::new();

                for portal in &volume_portals[portal_offset..portal_offset + volume.portal_count as usize] {
                    if portal.dst_volume_index >= header.volume_count {
                        return Err(MapLoadingError::InvalidVolumeIndex {
                            volume_count: header.volume_count,
                            volume_index: portal.dst_volume_index,
                        });
                    }

                    if portal.polygon_index >= header.polygon_length_count {
                        return Err(MapLoadingError::InvalidPolygonIndex {
                            polygon_count: header.polygon_length_count,
                            polygon_index: portal.polygon_index
                        });
                    }

                    portals.push(Portal {
                        dst_volume_id: VolumeId::from_index(portal.dst_volume_index as usize),
                        polygon_id: PolygonId::from_index(portal.polygon_index as usize),
                        is_facing_front: portal.is_facing_front != 0,
                    });
                }
                portal_offset += volume.portal_count as usize;

                for surface in &volume_surfaces[surface_offset..surface_offset + volume.surface_count as usize] {
                    if surface.material_index >= header.material_name_span_count {
                        return Err(MapLoadingError::InvalidMaterialIndex {
                            material_count: header.material_name_span_count,
                            material_index: surface.material_index,
                        });
                    }

                    if surface.polygon_index >= header.polygon_length_count {
                        return Err(MapLoadingError::InvalidPolygonIndex {
                            polygon_count: header.polygon_length_count,
                            polygon_index: surface.polygon_index
                        });
                    }

                    surfaces.push(Surface {
                        material_id: MaterialId::from_index(surface.material_index as usize),
                        polygon_id: PolygonId::from_index(surface.polygon_index as usize),
                        is_transparent: surface.is_transparent != 0,
                        is_sky: surface.is_sky != 0,

                        u: surface.u.into(),
                        v: surface.v.into(),
                    });
                }
                surface_offset += volume.surface_count as usize;

                result.push(Volume { portals, surfaces });
            }

            result
        };

        Ok(Self {
            bsp,
            material_name_set,
            polygon_set,
            volume_set,
        })
    }

    /// Save map in WBSP format
    pub fn save(&self, dst: &mut dyn std::io::Write) -> Result<(), std::io::Error> {
        // Construct WBSP file header
        let header = wbsp::Header {
            magic: wbsp::MAGIC,

            string_buffer_length: self
                .material_name_set
                .iter()
                .map(|name| name.len() as u32)
                .sum(),

            polygon_length_count: self.polygon_set.len() as u32,
            polygon_point_count: self
                .polygon_set
                .iter()
                .map(|polygon| polygon.points.len() as u32)
                .sum(),
            material_name_span_count: self.material_name_set.len() as u32,
            volume_portal_count: self.volume_set
                .iter()
                .map(|volume| volume.get_portals().len() as u32)
                .sum(),
            volume_surface_count: self.volume_set
                .iter()
                .map(|volume| volume.get_surfaces().len() as u32)
                .sum(),
            volume_count: self.volume_set.len() as u32,

            bsp_element_count: self.bsp_models
                .iter()
                .map(|m| m.get_bsp().size())
                .sum::<usize>() as u32,
        };

        // Write header
        dst.write(bytemuck::bytes_of(&header))?;

        fn write_by_iter<'t, T: bytemuck::NoUninit>(
            dst: &mut dyn std::io::Write,
            iter: impl Iterator<Item = T>
        ) -> Result<(), std::io::Error> {


            
            for val in iter {
                dst.write(bytemuck::bytes_of(&val))?;
            }

            Ok(())
        }

        // Write polygon lengths
        write_by_iter(dst, self
            .polygon_set
            .iter()
            .map(|polygon| polygon.points.len() as u32)
        )?;

        // Write polygon contents
        write_by_iter(dst, self
            .polygon_set
            .iter()
            .flat_map(|polygon| polygon.points.iter())
            .map(|point| wbsp::Vec3 {
                x: point.x,
                y: point.y,
                z: point.z,
            })
        )?;


        // Write material name lengths
        write_by_iter(dst, self
            .material_name_set
            .iter()
            .map(|name| name.len() as u32)
        )?;

        // Write material name bytes
        // TODO: UUGH, it's TOO slow
        write_by_iter(dst, self
            .material_name_set
            .iter()
            .flat_map(|name| name.as_bytes().iter().copied())
        )?;

        // Write volume portals
        write_by_iter(dst, self
            .volume_set
            .iter()
            .flat_map(|volume| volume.portals.iter())
            .map(|portal| wbsp::Portal {
                dst_volume_index: portal.dst_volume_id.into_index() as u32,
                polygon_index: portal.polygon_id.into_index() as u32,
                is_facing_front: portal.is_facing_front as u32,
            })
        )?;

        // Write volume surfaces
        write_by_iter(dst, self
            .volume_set
            .iter()
            .flat_map(|volume| volume.surfaces.iter())
            .map(|surface| wbsp::Surface {
                material_index: surface.material_id.into_index() as u32,
                polygon_index: surface.polygon_id.into_index() as u32,
                u: surface.u.into(),
                v: surface.v.into(),
                is_transparent: surface.is_transparent as u8,
                is_sky: surface.is_sky as u8,
            })
        )?;

        // Write volumes
        write_by_iter(dst, self
            .volume_set
            .iter()
            .map(|volume| wbsp::Volume {
                portal_count: volume.portals.len() as u32,
                surface_count: volume.surfaces.len() as u32,
            })
        )?;

        // BSP write function
        fn write_bsp(bsp: &Bsp, dst: &mut dyn std::io::Write) -> Result<(), std::io::Error> {
            match bsp {
                Bsp::Partition { splitter_plane, front, back } => {
                    dst.write(bytemuck::bytes_of(&wbsp::BspElement {
                        ty: wbsp::BspType::Partition as u8,
                        data: wbsp::BspElementData {
                            partition: wbsp::BspPartition {
                                plane_distance: splitter_plane.distance,
                                plane_normal: splitter_plane.normal.into(),
                            }
                        },
                    }));

                    write_bsp(front.as_ref(), dst)?;
                    write_bsp(back.as_ref(), dst)?;
                }
                Bsp::Volume(id) => {
                    dst.write(bytemuck::bytes_of(&wbsp::BspElement {
                        ty: wbsp::BspType::Partition as u8,
                        data: wbsp::BspElementData {
                            volume: wbsp::BspVolume {
                                volume_index: id.into_index() as u32
                            },
                        },
                    }));
                }
                Bsp::Void => {
                    dst.write(bytemuck::bytes_of(&wbsp::BspElement {
                        ty: wbsp::BspType::Partition as u8,
                        data: wbsp::BspElementData::zeroed(),
                    }));
                }
            }

            Ok(())
        }

        Ok(())
    }
}

// mod.rs
