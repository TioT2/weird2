///! Quake 1 WAD2 resource format

/// WAD2 file magic
pub const MAGIC: [u8; 4] = *b"WAD2";

/// WAD2 file header
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Header {
    /// Magic number
    pub magic: [u8; 4],

    /// Count WAD entries
    pub entry_count: i32,

    /// Entry array offset
    pub entry_array_offset: i32,
}

unsafe impl bytemuck::Zeroable for Header {}
unsafe impl bytemuck::AnyBitPattern for Header {}
unsafe impl bytemuck::NoUninit for Header {}


/// WAD entry
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Entry {
    /// Offset of data block
    pub data_offset: i32,

    /// Count of bytes
    pub data_size: i32,

    /// Size of decompressed entry
    pub data_decompressed_size: i32,

    /// Entry type
    pub ty: i8,

    /// Compression flag
    pub compression: i8,

    /// Just unused bytes
    pub _dummy: u16,

    /// Entry name
    pub name: [u8; 16],
}


unsafe impl bytemuck::Zeroable for Entry {}
unsafe impl bytemuck::AnyBitPattern for Entry {}
unsafe impl bytemuck::NoUninit for Entry {}

/// Entry type
#[repr(C)]
#[derive(Copy, Clone)]
pub enum EntryType {
    /// Color palette ([Color; 256] array)
    ColorPalette = 0x40,

    /// StatusBar picture (PictureHeader + contents)
    StatusBarPictures = 0x42,

    /// MipMapped texture
    MipTexture = 0x44,

    /// 320x200 picture (?)
    ConsolePicture = 0x45,
}

impl EntryType {
    pub fn from_i8(v: i8) -> Option<Self> {
        Some(match v {
            0x40 => Self::ColorPalette,
            0x42 => Self::StatusBarPictures,
            0x44 => Self::MipTexture,
            0x45 => Self::ConsolePicture,
            _ => return None,
        })
    }
}

/// Header of standard picture
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct PictureHeader {
    /// Picture width
    pub width: i32,

    /// Picture height
    pub height: i32,
    // Pixels: array of i8 with width x height size
}

unsafe impl bytemuck::Zeroable for PictureHeader {}
unsafe impl bytemuck::AnyBitPattern for PictureHeader {}
unsafe impl bytemuck::NoUninit for PictureHeader {}

/// Mipmap texture header
#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct MipTextureHeader {
    /// Texture name
    pub name: [u8; 16],

    /// Texture width
    pub width: i32,

    /// Texture height
    pub height: i32,

    /// Offsets of pre-generated texture mipmaps
    pub offsets: [i32; 4],
}

unsafe impl bytemuck::Zeroable for MipTextureHeader {}
unsafe impl bytemuck::AnyBitPattern for MipTextureHeader {}
unsafe impl bytemuck::NoUninit for MipTextureHeader {}

#[repr(C, packed)]
#[derive(Copy, Clone)]
pub struct Color {
    /// Red component
    pub r: u8,

    /// Green component
    pub g: u8,

    /// Blue component
    pub b: u8,
}

impl Color {
    /// Get zeor color
    pub const fn zero() -> Self {
        Self { r: 0, g: 0, b: 0 }
    }

    /// Transform color into u32
    pub const fn rgb_u32(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
    }
}

unsafe impl bytemuck::Zeroable for Color {}
unsafe impl bytemuck::AnyBitPattern for Color {}
unsafe impl bytemuck::NoUninit for Color {}
