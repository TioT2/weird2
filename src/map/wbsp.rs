use bytemuck::{AnyBitPattern, NoUninit, Zeroable};

///! WBSP file format description module.

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
    pub material_index: u32,
    pub polygon_index: u32,
}

unsafe impl Zeroable for Surface {}
unsafe impl AnyBitPattern for Surface {}
unsafe impl NoUninit for Surface {}

/// Portal (volume-volume connection)
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Portal {
    /// Polygon index
    pub polygon_index: u32,
    pub dst_volume_index: u32,
    pub is_facing_front: u32,
}

unsafe impl Zeroable for Portal {}
unsafe impl AnyBitPattern for Portal {}
unsafe impl NoUninit for Portal {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Volume {
    pub surface_count: u32,
    pub portal_count: u32,
}

unsafe impl Zeroable for Volume {}
unsafe impl AnyBitPattern for Volume {}
unsafe impl NoUninit for Volume {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Material {
    pub color: u32,
}

unsafe impl Zeroable for Material {}
unsafe impl AnyBitPattern for Material {}
unsafe impl NoUninit for Material {}

#[repr(C)]
pub enum BspType {
    Partition = 1,
    Volume = 2,
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

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

unsafe impl Zeroable for Vec3 {}
unsafe impl AnyBitPattern for Vec3 {}
unsafe impl NoUninit for Vec3 {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspPartition {
    pub plane_distance: f32,
    pub plane_normal: Vec3,
}

unsafe impl Zeroable for BspPartition {}
unsafe impl AnyBitPattern for BspPartition {}
unsafe impl NoUninit for BspPartition {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspVolume {
    pub volume_index: u32,
}

unsafe impl Zeroable for BspVolume {}
unsafe impl AnyBitPattern for BspVolume {}
unsafe impl NoUninit for BspVolume {}
