///! Main project BSP structure declaration module

use std::num::NonZeroU32;
use bytemuck::Zeroable;
use crate::{geom, math::Vec3f};

/// Declare actual map builder module
pub mod builder;

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

// TODO: Rename surface into smth more logical.

/// Visible volume face piece
pub struct Surface {
    /// Polygon material identifier
    pub material_id: MaterialId,

    /// Polygon itself identifier
    pub polygon_id: PolygonId,

    /// Surface transparency flag
    pub is_transparent: bool,

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

#[repr(C, packed)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Rgb8 {
    /// Red color component
    pub r: u8,

    /// Gree color component
    pub g: u8,

    /// Blue color component
    pub b: u8,
}

impl From<u32> for Rgb8 {
    fn from(value: u32) -> Self {
        let [r, g, b, _] = value.to_le_bytes();

        Self { r, g, b }
    }
}

impl Into<u32> for Rgb8 {
    fn into(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
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
            Bsp::Partition { splitter_plane: _, front, back } =>
                usize::max(front.depth(), back.depth()) + 1,
            _ =>
                1,
        }
    }
}

/// Map
pub struct Map {
    /// Map (visible) BSP.
    bsp: Box<Bsp>,

    /// Set of map polygons
    polygon_set: Vec<geom::Polygon>,

    /// Set of map materials
    material_name_set: Vec<String>,

    /// Set of map volumes
    volume_set: Vec<Volume>,
}

impl Map {
    /// Build map from BSP and sets.
    pub fn new(
        bsp: Box<Bsp>,
        polygon_set: Vec<geom::Polygon>,
        material_name_set: Vec<String>,
        volume_set: Vec<Volume>
    ) -> Self {

        Self {
            bsp,
            polygon_set,
            material_name_set,
            volume_set
        }
    }

    /// Get volume by id
    pub fn get_volume(&self, id: VolumeId) -> Option<&Volume> {
        self.volume_set.get(id.into_index())
    }

    /// Get iterator on ids of all volumes
    pub fn all_volume_ids(&self) -> impl Iterator<Item = VolumeId> {
        (0..self.volume_set.len())
            .map(VolumeId::from_index)
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

    /// Get polygon by id
    pub fn get_polygon(&self, id: PolygonId) -> Option<&geom::Polygon> {
        self.polygon_set.get(id.into_index())
    }

    /// Get location BSP
    pub fn get_bsp(&self) -> &Bsp {
        &self.bsp
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

    /// Remove voids from BSP
    pub fn _remove_bsp_voids(&mut self) {
        fn remove_voids(bsp: Bsp) -> Option<Bsp> {
            match bsp {
                Bsp::Partition { splitter_plane, front, back } => {
                    let front = remove_voids(*front);
                    let back = remove_voids(*back);

                    if let Some(front) = front {
                        if let Some(back) = back {
                            Some(Bsp::Partition {
                                splitter_plane,
                                front: Box::new(front),
                                back: Box::new(back)
                            })
                        } else {
                            Some(front)
                        }
                    } else {
                        if let Some(back) = back {
                            Some(back)
                        } else {
                            None
                        }
                    }
                }
                Bsp::Volume(id) => Some(Bsp::Volume(id)),
                Bsp::Void => None,
            }
        }

        let old_bsp = std::mem::replace(&mut self.bsp, Box::new(Bsp::Void));

        if let Some(bsp) = remove_voids(*old_bsp) {
            self.bsp = Box::new(bsp);
        }
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
        let material_name_lengths = read_vec::<u32>(src, header.material_name_length_count)?;
        let material_name_bytes = read_vec::<u8>(src, header.material_name_chars_length)?;
        let volume_portals = read_vec::<wbsp::Portal>(src, header.volume_portal_count)?;
        let volume_surfaces = read_vec::<wbsp::Surface>(src, header.volume_surface_count)?;
        let volumes = read_vec::<wbsp::Volume>(src, header.volume_count)?;

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
            let length_sum = material_name_lengths.iter().sum();

            if length_sum as usize > material_name_bytes.len() {
                return Err(MapLoadingError::ShortMaterialNameByteSet {
                    name_length_sum: length_sum,
                    name_bytes_count: material_name_bytes.len() as u32
                });
            }

            let mut material_name_bytes_rest = material_name_bytes.as_slice();

            material_name_lengths
                .iter()
                .map(|len| {
                    let bytes;
                    (bytes, material_name_bytes_rest) =
                        material_name_bytes_rest.split_at(*len as usize);

                    String::from_utf8(bytes.to_vec())
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(MapLoadingError::StringFromUtf8Error)
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
                    // Validate material index
                    if surface.material_index >= header.material_name_length_count {
                        return Err(MapLoadingError::InvalidMaterialIndex {
                            material_count: header.material_name_length_count,
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
            polygon_length_count: self.polygon_set.len() as u32,
            polygon_point_count: self
                .polygon_set
                .iter()
                .map(|polygon| polygon.points.len() as u32)
                .sum(),
            material_name_chars_length: self
                .material_name_set
                .iter()
                .map(|name| name.len() as u32)
                .sum(),
            material_name_length_count: self.material_name_set.len() as u32,
            volume_portal_count: self.volume_set
                .iter()
                .map(|volume| volume.get_portals().len() as u32)
                .sum(),
            volume_surface_count: self.volume_set
                .iter()
                .map(|volume| volume.get_surfaces().len() as u32)
                .sum(),
            volume_count: self.volume_set.len() as u32,
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
                    dst.write(bytemuck::bytes_of(&(wbsp::BspType::Partition as u32)))?;

                    dst.write(bytemuck::bytes_of(&wbsp::BspPartition {
                        plane_distance: splitter_plane.distance,
                        plane_normal: wbsp::Vec3 {
                            x: splitter_plane.normal.x,
                            y: splitter_plane.normal.y,
                            z: splitter_plane.normal.z,
                        }
                    }))?;
                    write_bsp(front.as_ref(), dst)?;
                    write_bsp(back.as_ref(), dst)?;
                }
                Bsp::Volume(id) => {
                    dst.write(bytemuck::bytes_of(&(wbsp::BspType::Volume as u32)))?;

                    dst.write(bytemuck::bytes_of(&wbsp::BspVolume {
                        volume_index: id.into_index() as u32,
                    }))?;
                }
                Bsp::Void => {
                    dst.write(bytemuck::bytes_of(&(wbsp::BspType::Void as u32)))?;
                }
            }

            Ok(())
        }

        // Write BSP
        write_bsp(self.bsp.as_ref(), dst)?;

        Ok(())
    }
}

// mod.rs
