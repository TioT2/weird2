//! Resource format

/// Quake 1 WAD2 file format
pub mod q1_wad2;

use std::{collections::BTreeMap, ffi::CStr};
use thiserror::Error;
use zerocopy::FromBytes;

use crate::{bsp::{self, Id}, frame_slice::FrameSlice};

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

impl Default for Texture {
    fn default() -> Self {
        Self::new_checker(0, 0xFF00FF)
    }
}

impl Texture {
    /// Get 2x2 checker texture ([`even`] is color at even corodinate sum)
    pub fn new_checker(even: u32, odd: u32) -> Self {
        Self {
            uv_scale: 32.0,
            width: 2,
            height: 2,
            offsets: vec![0],
            data: vec![even, odd, odd, even],
        }
    }

    /// Get base texture width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get base texture height
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get number of texture mip levels
    pub fn mip_levels(&self) -> usize {
        self.offsets.len()
    }

    /// Access mip level contents (0 corresponds to **largest** image).
    pub fn get_mip_level<'t>(&'t self, level: usize) -> Option<(FrameSlice<'t, u32>, f32)> {
        let offset = self.offsets.get(level).copied()?;
        let width = self.width >> level;
        let height = self.height >> level;

        Some((
            FrameSlice::new(width, height, width, &self.data[offset..offset + width * height]),
            (1 << level) as f32 * self.uv_scale
        ))
    }
}

/// Material table structure
pub struct MaterialTable {
    /// Index table
    pub texture_index_map: BTreeMap<String, usize>,

    /// Zero index stands for default texture
    pub textures: Vec<Texture>,
}

/// WAD2 file loading error
#[derive(Debug, Error)]
pub enum Wad2LoadingError {
    /// Invali WAD file magic
    #[error("Invalid WAD2 file magic: 0x{a:08X}, 0x{t:08X}", a = u32::from_ne_bytes(*.0), t = u32::from_ne_bytes(q1_wad2::MAGIC))]
    InvalidWadMagic([u8; 4]),

    /// Unexpected WAD file end
    #[error("Unexpected end of file contents")]
    UnexpectedFileEnd,

    /// Unexpected WAD entry end
    #[error("Unexpected end of entry contents")]
    UnexpectedEntryEnd,

    /// File interaction error
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error of building UTF-8 from bytes that (technically) should be string
    #[error("UTF8 reading error")]
    Utf8Error(#[from] std::str::Utf8Error),

    /// Invalid CStr
    #[error("C string reading error")]
    CStrReadingError(#[from] std::ffi::FromBytesUntilNulError),

    /// Slice is smaller, than required
    #[error("Slice is smaller, than required")]
    TooSmallSlice(#[from] std::array::TryFromSliceError),

    /// Compressed data occured
    #[error("Compressed WAD2 data is not supported")]
    CompressedData,

    /// Entry type isn't documented.
    #[error("Entry type 0x{0:02X} is not known.")]
    UnknownEntryType(i8),
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

    /// Get some default material color
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
        let mut color_palette: [u32; 256] = *q1_wad2::DEFAULT_COLOR_PALETTE;

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

    /// Get default material
    pub fn get_default_material<'t>(&'t self) -> &'t Texture {
        self.textures.first().unwrap()
    }

    /// Get material by name
    pub fn get_material<'t>(&'t self, str: &str) -> Option<&'t Texture> {
        Some(&self.textures[*self.texture_index_map.get(str)?])
    }

    /// Build reference table
    pub fn build_reference_table<'t>(&'t self, map: &bsp::Map) -> MaterialReferenceTable<'t> {
        let mut ref_table = Vec::new();

        let default_texture = self.textures.get(0).unwrap();

        for (mtl_id, mtl_name) in map.all_material_names() {
            let index = mtl_id.into_index();

            // Use name DJB2 hash as color
            let color = mtl_name.as_bytes().iter().fold(5381u32, |h, b| h.wrapping_mul(33) ^ (*b as u32));

            ref_table.resize(index + 1, (default_texture, 0xFF00FF));

            let texture_index_opt = self.texture_index_map.get(mtl_name).copied();
            let texture_index = texture_index_opt.unwrap_or(0);

            ref_table[index] = (self.textures.get(texture_index).unwrap(), color);
        }

        MaterialReferenceTable { ref_table }
    }
}

// mod.rs
