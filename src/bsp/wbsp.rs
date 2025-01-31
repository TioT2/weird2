///! WBSP file format description module.

use bytemuck::{AnyBitPattern, NoUninit, Zeroable};

/// Magic number
pub const MAGIC: u32 = u32::from_le_bytes(*b"WBSP");

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct Header {
    /// WBSP file magic
    pub magic: u32,

    /// Count of polygon lengths
    pub polygon_length_count: u32,

    /// Count of points
    pub polygon_point_count: u32,

    /// Count of polygons
    pub material_count: u32,

    /// Surface count
    pub volume_surface_count: u32,

    /// Portal count
    pub volume_portal_count: u32,

    /// Volume total count
    pub volume_count: u32,
}


unsafe impl Zeroable for Header {}
unsafe impl AnyBitPattern for Header {}
unsafe impl NoUninit for Header {}

/// Visible volume face piece
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Surface {
    /// Index of surface material in material set
    pub material_index: u32,

    /// Index of surface polygon in polygon set
    pub polygon_index: u32,
}

unsafe impl Zeroable for Surface {}
unsafe impl AnyBitPattern for Surface {}
unsafe impl NoUninit for Surface {}

/// Portal (volume-volume connection)
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Portal {
    /// Index of portal polygon in polygon set
    pub polygon_index: u32,

    /// Index of destination volume in volume set
    pub dst_volume_index: u32,

    /// If true, portal polygon is faced towards camera
    pub is_facing_front: u32,
}

unsafe impl Zeroable for Portal {}
unsafe impl AnyBitPattern for Portal {}
unsafe impl NoUninit for Portal {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Volume {
    /// Count of volume surfaces
    pub surface_count: u32,

    /// Count of polygon portals
    pub portal_count: u32,
}

unsafe impl Zeroable for Volume {}
unsafe impl AnyBitPattern for Volume {}
unsafe impl NoUninit for Volume {}

/// Material
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Material {
    /// Material color
    pub color: u32,
}

unsafe impl Zeroable for Material {}
unsafe impl AnyBitPattern for Material {}
unsafe impl NoUninit for Material {}

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

impl TryFrom<u32> for BspType {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => Self::Partition,
            2 => Self::Volume,
            3 => Self::Void,
            _ => return Err(value),
        })
    }
}

/// Stable (e.g. with certain field order) 3-component float vector
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Vec3 {
    /// X coordinate
    pub x: f32,

    /// Y coordinate
    pub y: f32,

    /// Z coordinate
    pub z: f32,
}

unsafe impl Zeroable for Vec3 {}
unsafe impl AnyBitPattern for Vec3 {}
unsafe impl NoUninit for Vec3 {}

/// BSP helper structure
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspPartition {
    /// Partition plane's distance
    pub plane_distance: f32,

    /// Partition plane's normal
    pub plane_normal: Vec3,
}

unsafe impl Zeroable for BspPartition {}
unsafe impl AnyBitPattern for BspPartition {}
unsafe impl NoUninit for BspPartition {}

/// BSP helper structure
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspVolume {
    /// Destination volume index
    pub volume_index: u32,
}

unsafe impl Zeroable for BspVolume {}
unsafe impl AnyBitPattern for BspVolume {}
unsafe impl NoUninit for BspVolume {}

// wbsp.rs
