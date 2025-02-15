///! WBSP file format description module.

use bytemuck::{AnyBitPattern, NoUninit, Zeroable};

use crate::geom;

/// .WBSP file Magic number
pub const MAGIC: u32 = u32::from_le_bytes(*b"WBSP");

// Binary format
macro_rules! bin_format {
    ($name: ident) => {
        unsafe impl Zeroable for $name {}
        unsafe impl AnyBitPattern for $name {}
        unsafe impl NoUninit for $name {}
    };

    ($head: ident, $($tail: ident),* $(,)?) => {
        bin_format!($head);

        bin_format!($($tail),*);
    };
}

/// Span in any kind of set
#[repr(C, packed)]
#[derive(Copy, Clone)]
#[derive(Debug)]
pub struct Span {
    /// String offset
    pub offset: u32,

    /// String size
    pub size: u32,
}

impl Span {
    pub const fn zero() -> Self {
        Self { offset: 0, size: 0 }
    }

    /// Range
    pub fn range(self) -> std::ops::Range<usize> {
        std::ops::Range::<usize> {
            start: self.offset as usize,
            end: self.offset as usize + self.size as usize,
        }
    }

    // TODO: TryFrom
    pub fn from_range(range: std::ops::Range<usize>) -> Self {
        Self {
            offset: range.start as u32,
            size: (range.end - range.start) as u32,
        }
    }
}

/// Header
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Header {
    /// WBSP file magic
    pub magic: u32,

    /// World BSP model index
    pub world_bsp_model_index: u32,

    /// Length of material string set
    pub chars: Span,

    /// Count of points
    pub points: Span,

    /// Count of polygon lengths
    pub polygons: Span,

    /// Material name spans
    pub material_names: Span,

    /// Surface count
    pub volume_surfaces: Span,

    /// Portal count
    pub volume_portals: Span,

    /// Total volume count
    pub volumes: Span,

    /// BSP element span
    pub bsp_elements: Span,

    /// BSP model span
    pub bsp_models: Span,

    /// Dynamic BSP models
    pub dynamic_models: Span,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Material {
    /// In 'chars' span
    pub name: Span,
}

/// Visible volume face piece
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Surface {
    /// Index of surface material in material set
    pub material_index: u32,

    /// Index of surface polygon in polygon set
    pub polygon_index: u32,

    /// U axis
    pub u: Plane,

    /// V axis
    pub v: Plane,

    /// True if transparent, false if not
    pub is_transparent: u8,

    /// Sky flag
    pub is_sky: u8,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Plane {
    /// Plane normal vector
    pub normal: Vec3,

    /// Distance from point to plane
    pub distance: f32,
}

impl Into<geom::Plane> for Plane {
    fn into(self) -> geom::Plane {
        geom::Plane { distance: self.distance, normal: self.normal.into() }
    }
}

impl From<geom::Plane> for Plane {
    fn from(value: geom::Plane) -> Self {
        Self { distance: value.distance, normal: value.normal.into() }
    }
}

/// Portal (volume-volume connection)
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Portal {
    /// Index of portal polygon in polygon set
    pub polygon_index: u32,

    /// Index of destination volume in volume set
    pub dst_volume_index: u32,

    /// If true, portal polygon is faced towards camera
    pub is_facing_front: u8,
}

/// Volume structure
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Volume {
    /// Surface span
    pub surfaces: Span,

    /// Portal span
    pub portals: Span,

    /// Boundbox min
    pub bound_box_min: Vec3,

    /// Boundbox max
    pub bound_box_max: Vec3,
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

impl Into<super::Vec3f> for Vec3 {
    fn into(self) -> super::Vec3f {
        super::Vec3f { x: self.x, y: self.y, z: self.z }
    }
}

impl From<super::Vec3f> for Vec3 {
    fn from(value: super::Vec3f) -> Self {
        Self { x: value.x, y: value.y, z: value.z }
    }
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

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspElement {
    /// Element type byte
    pub ty: u8,

    /// Element contents
    pub data: BspElementData,
}

/// BSP element data
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub union BspElementData {
    /// Volume contents
    pub volume: BspVolume,

    /// Partition contents
    pub partition: BspPartition,

    /// Void case
    pub void: (),
}

/// BSP helper structure
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspPartition {
    /// Partition plane's distance
    pub plane: Plane,
}

/// BSP helper structure
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspVolume {
    /// Destination volume index
    pub volume_index: u32,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct BspModel {
    /// BSP head offset
    pub bsp_root_index: u32,

    /// Bounding box minimum
    pub bound_box_min: Vec3,

    /// Bounding box maximum
    pub bound_box_max: Vec3,
}

/// Dynamic model (moving stuff, like doors etc.)
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct DynamicModel {
    /// BSP model index
    pub bsp_model_index: u32,

    /// Position
    pub origin: Vec3,

    /// Rotation
    pub rotation: f32,
}

// Binary format
bin_format!(
    BspElement,
    BspElementData,
    BspVolume,
    BspPartition,
    BspModel,
    DynamicModel,
    Material,
    Span,
    Header,
    Vec3,
    Volume,
    Portal,
    Surface,
    Plane,
);

// wbsp.rs
