///! Module with WBSP raw structure descriptions

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::geom;

/// .WBSP file Magic number
pub const MAGIC: u32 = u32::from_le_bytes(*b"WBSP");

/// Some span
#[repr(C, packed)]
#[derive(Copy, Clone, Debug, FromBytes, IntoBytes, Immutable)]
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
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct Header {
    /// WBSP file magic
    pub magic: u32,

    /// Index of world (e.g. root) model in BSP model list
    pub world_bsp_model_index: u32,

    /// Span that holds all string characters (String are just indices in the span data)
    pub chars: Span,

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
    /// In 'chars' span
    pub name: Span,
}

/// Wad2 file flags
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct SurfaceFlags(pub u32);

impl SurfaceFlags {
    /// Transparency flag
    pub const TRANSPARENT : u32 = 0b01;

    /// Sky flag
    pub const SKY         : u32 = 0b10;
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

    /// Surface flags
    pub flags: SurfaceFlags,
}

#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
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
    pub bound_box_min: Vec3,

    /// Volume bound box maximum
    pub bound_box_max: Vec3,
}

/// 3-component f32 vector vector
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
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
        super::Vec3f::new(self.x, self.y, self.z)
    }
}

impl From<super::Vec3f> for Vec3 {
    fn from(value: super::Vec3f) -> Self {
        Self { x: value.x(), y: value.y(), z: value.z() }
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

    /// Void case
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
    pub bound_box_min: Vec3,

    /// Bounding box maximum
    pub bound_box_max: Vec3,
}

/// Dynamic model (moving stuff, like doors etc.)
#[repr(C)]
#[derive(Copy, Clone, FromBytes, IntoBytes, Immutable)]
pub struct DynamicModel {
    /// BSP model index
    pub bsp_model_index: u32,

    /// Position
    pub origin: Vec3,

    /// Rotation
    pub rotation: f32,
}

// wbsp.rs
