//! Resource format

/// Quake 1 WAD2 file format
pub mod q1_wad2;

use std::{collections::BTreeMap, ffi::CStr};
use zerocopy::FromBytes;

use crate::{bsp::{self, Id}, rand};

/// Image reference
/// # Note
/// Image data should have exactly stride * height elements
#[derive(Copy, Clone)]
pub struct ImageRef<'t> {
    /// Width (in pixels)
    pub width: usize,

    /// Height (in pixels)
    pub height: usize,

    /// Image line stride, in dwords
    pub stride: usize,

    /// RGBX Data
    pub data: &'t [u32],
}

/// Mipmapped texture
pub struct Texture {
    /// Base UV scale
    uv_scale: f32,

    /// Texture width
    width: usize,

    /// Texture height
    height: usize,

    /// Offsets (in u32s) to mipmap datas
    offsets: Vec<usize>,

    /// Image data
    data: Vec<u32>,
}

impl Texture {
    /// Get default 'not found' texture
    pub fn default() -> Self {
        Self {
            uv_scale: 32.0,
            width: 2,
            height: 2,
            offsets: vec![0],
            data: vec![0x000000, 0xFF00FF, 0xFF00FF, 0x000000],
        }
    }

    /// Get texture mipmap count
    pub fn get_mipmap_count(&self) -> usize {
        self.offsets.len()
    }

    /// Mip image reference
    /// Returns (image, uv_scale) pair with UV account
    pub fn get_mipmap<'t>(&'t self, index: usize) -> (ImageRef<'t>, f32) {
        // Clamp mipmap index
        let index = usize::min(index, self.offsets.len() - 1);

        let width = self.width >> index;
        let height = self.height >> index;
        let offset = self.offsets[index];

        (
            ImageRef {
                data: &self.data[offset..offset + width * height],
                stride: width,
                width,
                height,
            },

            // Calculate texture UV scale
            (1 << index) as f32 * self.uv_scale
        )
    }
}

/// Material table imported from WAD files
pub struct MaterialTable {
    /// Index table
    pub texture_index_map: BTreeMap<String, usize>,

    /// Zero index stands for default texture
    pub textures: Vec<Texture>,
}

/// WAD2 file loading error
#[derive(Debug)]
pub enum Wad2LoadingError {
    /// Invali WAD file magic
    InvalidWadMagic([u8; 4]),

    /// Unexpected WAD file end
    UnexpectedFileEnd,

    /// Unexpected WAD entry end
    UnexpectedEntryEnd,

    /// File interaction error
    IoError(std::io::Error),

    /// Error of building UTF-8 from bytes that (technically) should be string
    Utf8Error(std::str::Utf8Error),

    /// Invalid CStr
    CStrReadingError(std::ffi::FromBytesUntilNulError),

    /// Slice is smaller, than required
    TooSmallSlice(std::array::TryFromSliceError),

    /// Compressed data occured
    CompressedData,

    /// Entry type isn't documented.
    UnknownEntryType(i8),
}

impl From<std::io::Error> for Wad2LoadingError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

impl From<std::str::Utf8Error> for Wad2LoadingError {
    fn from(value: std::str::Utf8Error) -> Self {
        Self::Utf8Error(value)
    }
}

impl From<std::array::TryFromSliceError> for Wad2LoadingError {
    fn from(value: std::array::TryFromSliceError) -> Self {
        Self::TooSmallSlice(value)
    }
}

impl From<std::ffi::FromBytesUntilNulError> for Wad2LoadingError {
    fn from(value: std::ffi::FromBytesUntilNulError) -> Self {
        Self::CStrReadingError(value)
    }
}

/// Table of references to certain WAD
pub struct MaterialReferenceTable<'t> {
    /// id.index -> (&texture, 'canonical' random color)
    ref_table: Vec<(&'t Texture, u32)>,
}

impl<'t> MaterialReferenceTable<'t> {
    /// Get texture by index
    pub fn get_texture(&self, id: bsp::MaterialId) -> Option<&Texture> {
        self.ref_table.get(id.into_index()).map(|(t, _)| t).copied()
    }

    pub fn get_color(&self, id: bsp::MaterialId) -> Option<u32> {
        self.ref_table.get(id.into_index()).map(|(_, c)| c).copied()
    }
}

impl MaterialTable {
    /// Load material table from Quake WAD2 file
    pub fn load_wad2(data: &[u8]) -> Result<Self, Wad2LoadingError> {
        let header = q1_wad2::Header::read_from_prefix(data).map_err(|_| Wad2LoadingError::UnexpectedFileEnd)?.0;

        // Read WAD2 header
        if header.magic != q1_wad2::MAGIC {
            return Err(Wad2LoadingError::InvalidWadMagic(header.magic));
        }

        // Entry array
        let entries = {
            let off = header.entry_array_offset as usize;
            let n = header.entry_count as usize;
            let edata = data
                .get(off..off + n * std::mem::size_of::<q1_wad2::Entry>())
                .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

            <[q1_wad2::Entry]>::ref_from_prefix(edata)
                .map_err(|_| Wad2LoadingError::UnexpectedFileEnd)?.0
        };
        let mut color_palette: [u32; 256] = q1_wad2::DEFAULT_COLOR_PALETTE;

        // Find palette entry and parse it

        let mut textures =  Vec::<Texture>::new();
        let mut texture_index_map = BTreeMap::<String, usize>::new();

        // Default texture isn't mapped to any name
        textures.push(Texture::default());

        // Read WAD entries
        for entry in entries {
            // I Don't know how to decode compressed data...
            if entry.compression != 0 {
                continue;
                // return Err(Wad2LoadingError::CompressedData);
            }

            let ty = q1_wad2::EntryType::from_i8(entry.ty)
                .ok_or(Wad2LoadingError::UnknownEntryType(entry.ty))?;

            let name = CStr::from_bytes_until_nul(&entry.name)?.to_str()?;

            let offset = entry.data_offset as usize;

            let entry_data = data
                .get(offset..offset + entry.data_size as usize)
                .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

            match ty {
                q1_wad2::EntryType::ColorPalette => {
                    let wad_color_slice: &[q1_wad2::Color; 256] = <[q1_wad2::Color; 256]>::ref_from_prefix(entry_data)
                        .map_err(|_| Wad2LoadingError::UnexpectedEntryEnd)?
                        .0;

                    color_palette = wad_color_slice.map(|v| v.rgb_u32());
                }
                q1_wad2::EntryType::ConsolePicture => {
                    // ignore
                }
                q1_wad2::EntryType::StatusBarPictures => {
                    // ignore
                }
                q1_wad2::EntryType::MipTexture => {
                    let header = q1_wad2::MipTextureHeader::read_from_prefix(entry_data)
                        .map_err(|_| Wad2LoadingError::UnexpectedEntryEnd)?.0;

                    let bytes = entry_data;

                    let width = header.width as usize;
                    let height = header.height as usize;

                    let mut data = Vec::<u32>::new();
                    let mut offsets = Vec::<usize>::new();

                    offsets.push(0);

                    for index in 0..4 {
                        let src_offset = header.offsets[index] as usize;

                        let width = width >> index;
                        let height = height >> index;

                        if bytes.len() < width * height + src_offset {
                            return Err(Wad2LoadingError::UnexpectedEntryEnd);
                        }

                        // Reserve texture
                        data.reserve(width * height);

                        for y in 0..height {
                            for x in 0..width {
                                let index = bytes[src_offset + y * width + x];
                                let color = color_palette[index as usize];

                                data.push(color);
                            }
                        }

                        offsets.push(data.len());
                    }

                    offsets.pop();
                    offsets.shrink_to_fit();
                    data.shrink_to_fit();

                    let name = name.to_string().to_ascii_uppercase();
                    texture_index_map.insert(name, textures.len());
                    textures.push(Texture {
                        uv_scale: 1.0,
                        width,
                        height,
                        offsets,
                        data,
                    });
                }
            }
        }

        Ok(Self {
            textures,
            texture_index_map,
        })
    }

    /// Build WAD reference table
    pub fn build_reference_table<'t>(&'t self, bsp: &bsp::Map) -> MaterialReferenceTable<'t> {
        // Quite bad solution, though...
        let mut ref_table = Vec::new();
        let mut rand_device = rand::Xorshift128p::new(304780.try_into().unwrap());

        let default_texture = self.textures.get(0).unwrap();

        for (mtl_id, mtl_name) in bsp.all_material_names() {
            let index = mtl_id.into_index();
            let color = (rand_device.next() & 0xFFFF_FFFF) as u32;

            ref_table.resize(index + 1, (default_texture, 0xFF00FF));

            let texture_index_opt = self.texture_index_map.get(mtl_name).copied();
            let texture_index = texture_index_opt.unwrap_or(0);

            ref_table[index] = (self.textures.get(texture_index).unwrap(), color);
        }

        MaterialReferenceTable { ref_table }
    }
}

// mod.rs
