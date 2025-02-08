///! Resource format

/// Quake 1 WAD2 file format
pub mod q1_wad2;

use std::{collections::BTreeMap, ffi::CStr};
use crate::{bsp, rand};

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
    /// id.index -> (&texture, some random color) mapping
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
    /// Load Quake WAD2 file in material table
    pub fn load_wad2(file: &mut dyn std::io::Read) -> Result<Self, Wad2LoadingError> {
        let mut data = Vec::<u8>::new();

        file.read_to_end(&mut data)?;

        let header_data = data
            .get(0..std::mem::size_of::<q1_wad2::Header>())
            .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;
        let header = bytemuck::pod_read_unaligned::<q1_wad2::Header>(header_data);

        // Read WAD2 header
        if header.magic != q1_wad2::MAGIC {
            return Err(Wad2LoadingError::InvalidWadMagic(header.magic));
        }

        let entry_bytes = data
            .get(header.entry_array_offset as usize..)
            .ok_or(Wad2LoadingError::UnexpectedFileEnd)?;

        let entries = bytemuck::cast_slice::<_, q1_wad2::Entry>(entry_bytes);

        // Standard quake color palette
        let mut color_palette: [u32; 256] = [
            0x000000, 0x0f0f0f, 0x1f1f1f, 0x2f2f2f, 0x3f3f3f, 0x4b4b4b, 0x5b5b5b, 0x6b6b6b,
            0x7b7b7b, 0x8b8b8b, 0x9b9b9b, 0xababab, 0xbbbbbb, 0xcbcbcb, 0xdbdbdb, 0xebebeb,
            0x070b0f, 0x0b0f17, 0x0b171f, 0x0f1b27, 0x13232f, 0x172b37, 0x172f3f, 0x1b374b,
            0x1b3b53, 0x1f435b, 0x1f4b63, 0x1f536b, 0x1f5773, 0x235f7b, 0x236783, 0x236f8f,
            0x0f0b0b, 0x1b1313, 0x271b1b, 0x332727, 0x3f2f2f, 0x4b3737, 0x573f3f, 0x674747,
            0x734f4f, 0x7f5b5b, 0x8b6363, 0x976b6b, 0xa37373, 0xaf7b7b, 0xbb8383, 0xcb8b8b,
            0x000000, 0x000707, 0x000b0b, 0x001313, 0x001b1b, 0x002323, 0x072b2b, 0x072f2f,
            0x073737, 0x073f3f, 0x074747, 0x0b4b4b, 0x0b5353, 0x0b5b5b, 0x0b6363, 0x0f6b6b,
            0x000007, 0x00000f, 0x000017, 0x00001f, 0x000027, 0x00002f, 0x000037, 0x00003f,
            0x000047, 0x00004f, 0x000057, 0x00005f, 0x000067, 0x00006f, 0x000077, 0x00007f,
            0x001313, 0x001b1b, 0x002323, 0x002b2f, 0x002f37, 0x003743, 0x073b4b, 0x074357,
            0x07475f, 0x0b4b6b, 0x0f5377, 0x135783, 0x135b8b, 0x1b5f97, 0x1f63a3, 0x2367af,
            0x071323, 0x0b172f, 0x0f1f3b, 0x13234b, 0x172b57, 0x1f2f63, 0x233773, 0x2b3b7f,
            0x33438f, 0x334f9f, 0x2f63af, 0x2f77bf, 0x2b8fcf, 0x27abdf, 0x1fcbef, 0x1bf3ff,
            0x00070b, 0x00131b, 0x0f232b, 0x132b37, 0x1b3347, 0x233753, 0x2b3f63, 0x33476f,
            0x3f537f, 0x475f8b, 0x536b9b, 0x5f7ba7, 0x6b87b7, 0x7b93c3, 0x8ba3d3, 0x97b3e3,
            0xa38bab, 0x977f9f, 0x877393, 0x7b678b, 0x6f5b7f, 0x635377, 0x574b6b, 0x4b3f5f,
            0x433757, 0x372f4b, 0x2f2743, 0x231f37, 0x1b172b, 0x131323, 0x0b0b17, 0x07070f,
            0x9f73bb, 0x8f6baf, 0x835fa3, 0x775797, 0x6b4f8b, 0x5f4b7f, 0x534373, 0x4b3b6b,
            0x3f335f, 0x372b53, 0x2b2347, 0x231f3b, 0x1b172f, 0x131323, 0x0b0b17, 0x07070f,
            0xbbc3db, 0xa7b3cb, 0x9ba3bf, 0x8b97af, 0x7b87a3, 0x6f7b97, 0x5f6f87, 0x53637b,
            0x47576b, 0x3b4b5f, 0x333f53, 0x273343, 0x1f2b37, 0x171f27, 0x0f131b, 0x070b0f,
            0x7b836f, 0x6f7b67, 0x67735f, 0x5f6b57, 0x57634f, 0x4f5b47, 0x47533f, 0x3f4b37,
            0x37432f, 0x2f3b2b, 0x273323, 0x1f2b1f, 0x172317, 0x131b0f, 0x0b130b, 0x070b07,
            0x1bf3ff, 0x17dfef, 0x13cbdb, 0x0fb7cb, 0x0fa7bb, 0x0b97ab, 0x07839b, 0x07738b,
            0x07637b, 0x00536b, 0x00475b, 0x00374b, 0x002b3b, 0x001f2b, 0x000f1b, 0x00070b,
            0xff0000, 0xef0b0b, 0xdf1313, 0xcf1b1b, 0xbf2323, 0xaf2b2b, 0x9f2f2f, 0x8f2f2f,
            0x7f2f2f, 0x6f2f2f, 0x5f2f2f, 0x4f2b2b, 0x3f2323, 0x2f1b1b, 0x1f1313, 0x0f0b0b,
            0x00002b, 0x00003b, 0x00074b, 0x00075f, 0x000f6f, 0x07177f, 0x071f93, 0x0b27a3,
            0x0f33b7, 0x1b4bc3, 0x2b63cf, 0x3b7fdb, 0x4f97e3, 0x5fabe7, 0x77bfef, 0x8bd3f7,
            0x3b7ba7, 0x379bb7, 0x37c3c7, 0x57e3e7, 0xffbf7f, 0xffe7ab, 0xffffd7, 0x000067,
            0x00008b, 0x0000b3, 0x0000d7, 0x0000ff, 0x93f3ff, 0xc7f7ff, 0xffffff, 0x535b9f,
        ];

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
                    let wad_color_slice: &[q1_wad2::Color; 256] = bytemuck::cast_slice::<_, q1_wad2::Color>(entry_data)
                        .try_into()
                        .map_err(|_| Wad2LoadingError::UnexpectedEntryEnd)?;

                    color_palette = wad_color_slice.map(|v| v.rgb_u32());
                }
                q1_wad2::EntryType::ConsolePicture => {
                    // ignore
                }
                q1_wad2::EntryType::StatusBarPictures => {
                    // ignore
                }
                q1_wad2::EntryType::MipTexture => {
                    // Try to parse texture
                    let header_data = entry_data
                        .get(..std::mem::size_of::<q1_wad2::MipTextureHeader>())
                        .ok_or(Wad2LoadingError::UnexpectedEntryEnd)?;
                    let header = bytemuck::pod_read_unaligned::<q1_wad2::MipTextureHeader>(header_data);

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
