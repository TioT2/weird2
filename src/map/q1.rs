///! Quake 1 map format support

use std::collections::HashMap;
use crate::{geom, math::{Mat4f, Vec3f}};

/// Q1 .map brush face binary representation
#[derive(Clone, Debug)]
pub struct MapBrushFace {
    /// First brush point
    pub p0: Vec3f,

    /// Second brush point
    pub p1: Vec3f,

    /// Third brush point
    pub p2: Vec3f,

    /// Brush texture name
    pub texture_name: String,

    /// Texture offset by X (in texels)
    pub texture_offset_x: f32,

    /// Texture offset by Y (in texels)
    pub texture_offset_y: f32,

    /// Texture rotation (in degrees)
    pub texture_rotation: f32,

    /// Texture scale by X (in texels per unit (?))
    pub texture_scale_x: f32,

    /// Texture scale by Y (in texels per unit (?))
    pub texture_scale_y: f32,
}

/// Map brush
#[derive(Clone, Debug)]
pub struct MapBrush {
    /// Face set
    pub faces: Vec<MapBrushFace>,
}

/// Map entity (brush and string-string property collection)
#[derive(Clone, Debug)]
pub struct MapEntity {
    /// Entity properties
    pub properties: HashMap<String, String>,

    /// Entity brush set
    pub brushes: Vec<MapBrush>,
}

/// Map (entity collection, actually)
#[derive(Clone, Debug)]
pub struct Map {
    /// All map entities
    pub entities: Vec<MapEntity>,
}

/// Q1 map parsing error
#[derive(Debug, PartialEq, Eq)]
pub enum MapParseError<'t> {
    /// Expected one more token
    NextTokenExpected,

    /// Floating-point number parsing error
    FloatParsingError {
        /// Token floating point number parsed from
        token: &'t str,

        /// Exact error occured during parsing process
        error: std::num::ParseFloatError
    },

    /// Invalid property tokens
    InvalidProperty {
        /// Potential key token
        key: &'t str,

        /// Potential value tokens
        value: &'t str
    },

    /// Unexpected token
    UnexpectedToken {
        /// Actual token
        actual: &'t str,

        /// Expected token
        expected: &'t str,
    },
}

impl Map {
    /// Parse map from string
    pub fn parse<'t>(str: &'t str) -> Result<Map, MapParseError<'t>> {

        /// Parse token from string start
        fn parse_token<'t>(mut str_rest: &'t str) -> Result<(&'t str, &'t str), MapParseError<'t>> {
            // token skipping loop
            'comment_skip_loop: loop {
                str_rest = str_rest.trim_start();

                if str_rest.starts_with("//") {
                    let mut off = 0;

                    let mut rest_iter = str_rest.chars();

                    'comment_skip: while let Some(ch) = rest_iter.next() {
                        if ch == '\n' {
                            break 'comment_skip;
                        }
                        off += ch.len_utf8();
                    }
    
                    str_rest = str_rest[off..].trim_start();
                } else {
                    break 'comment_skip_loop;
                }
            }

            // parse string token
            if str_rest.starts_with('\"') {
                let mut off = 1;

                let mut rest_iter = str_rest[1..].chars();

                while let Some(ch) = rest_iter.next() {
                    if ch == '\"' {
                        let result = &str_rest[0..off + 1];
                        str_rest = &str_rest[off + 1..];
                        return Ok((result, str_rest));
                    }
                    off += ch.len_utf8();
                }

                let result = &str_rest[1..];
                str_rest = "";

                return Ok((result, str_rest));
            }

            if str_rest.is_empty() {
                return Err(MapParseError::NextTokenExpected);
            }

            if let Some((first, last)) = str_rest.split_once(char::is_whitespace) {
                str_rest = last;
                return Ok((first, str_rest));
            } else {
                let result = str_rest;
                str_rest = "";
                return Ok((result, str_rest));
            }
        }

        /// Parse **any** next token
        fn parse_next_token<'t, 'l>(tl: &'l [&'t str]) -> Result<(&'t str, &'l [&'t str]), MapParseError<'t>> {
            if let Some((tok, rest)) = tl.split_first() {
                Ok((*tok, rest))
            } else {
                Err(MapParseError::NextTokenExpected)
            }
        }

        fn parse_literal<'t, 'l>(tl: &'l [&'t str], lit: &'t str) -> Result<((), &'l [&'t str]), MapParseError<'t>> {
            let (tok, tl) = parse_next_token(tl)?;

            if tok == lit {
                Ok(((), tl))
            } else {
                Err(MapParseError::UnexpectedToken {
                    actual: tok,
                    expected: lit,
                })
            }
        }

        fn parse_float<'t, 'l>(tl: &'l [&'t str]) -> Result<(f32, &'l [&'t str]), MapParseError<'t>> {
            let (token, tl) = parse_next_token(tl)?;

            let val = token
                .parse::<f32>()
                .map_err(|error| MapParseError::FloatParsingError { token, error })?;

            return Ok((val, tl));
        }

        fn parse_vector<'t, 'l>(tl: &'l [&'t str]) -> Result<(Vec3f, &'l [&'t str]), MapParseError<'t>> {
            let (_, tl) = parse_literal(tl, "(")?;
            let (x, tl) = parse_float(tl)?;
            let (y, tl) = parse_float(tl)?;
            let (z, tl) = parse_float(tl)?;
            let (_, tl) = parse_literal(tl, ")")?;

            Ok((Vec3f::new(x, y, z), tl))
        }

        fn parse_brush_face<'t, 'l>(mut tl: &'l [&'t str]) -> Result<(MapBrushFace, &'l [&'t str]), MapParseError<'t>> {
            let p0;
            let p1;
            let p2;
            let texture_name;
            let texture_offset_x;
            let texture_offset_y;
            let texture_rotation;
            let texture_scale_x;
            let texture_scale_y;

            // parse plane basis vectors
            (p0, tl) = parse_vector(tl)?;
            (p1, tl) = parse_vector(tl)?;
            (p2, tl) = parse_vector(tl)?;

            // plane texture name
            (texture_name, tl) = parse_next_token(tl)?;
            let texture_name = texture_name.to_string();

            // 
            (texture_offset_x, tl) = parse_float(tl)?;
            (texture_offset_y, tl) = parse_float(tl)?;
            (texture_rotation, tl) = parse_float(tl)?;
            (texture_scale_x, tl) = parse_float(tl)?;
            (texture_scale_y, tl) = parse_float(tl)?;

            Ok((
                MapBrushFace {
                    p0,
                    p1,
                    p2,
                    texture_name,
                    texture_offset_x,
                    texture_offset_y,
                    texture_rotation,
                    texture_scale_x,
                    texture_scale_y
                },
                tl
            ))
        }

        fn parse_brush<'t, 'l>(mut tl: &'l [&'t str]) -> Result<(MapBrush, &'l [&'t str]), MapParseError<'t>> {
            (_, tl) = parse_literal(tl, "{")?;

            let mut faces = Vec::new();

            while let Ok((face, new_tl)) = parse_brush_face(tl) {
                tl = new_tl;
                faces.push(face);
            }

            (_, tl) = parse_literal(tl, "}")?;

            Ok((MapBrush { faces }, tl))
        }

        fn parse_property<'t, 'l>(tl: &'l [&'t str]) -> Result<((String, String), &'l [&'t str]), MapParseError<'t>> {
            let (key, tl) = parse_next_token(tl)?;
            let (value, tl) = parse_next_token(tl)?;

            if true
                && key.starts_with('\"')
                && key.ends_with('\"')
                && value.starts_with('\"')
                && value.ends_with('\"')
            {
                Ok(((key.trim_matches('\"').to_string(), value.trim_matches('\"').to_string()), tl))
            } else {
                Err(MapParseError::InvalidProperty { key, value })
            }
        }

        let tokens = {
            let mut str_rest = str;
            let mut tok_list = Vec::new();

            'parsing_loop: loop {
                match parse_token(str_rest) {
                    Ok((tok, new_str_rest)) => {
                        tok_list.push(tok);
                        str_rest = new_str_rest;
                    }
                    Err(MapParseError::NextTokenExpected) => break 'parsing_loop,
                    Err(err) => return Err(err)
                };
            }

            tok_list
        };

        let mut entities = Vec::<MapEntity>::new();
        let mut tl = tokens.as_slice();

        'parsing_loop: loop {
            match parse_literal(tl, "{") {
                Ok((_, new_tl)) => tl = new_tl,
                Err(MapParseError::NextTokenExpected) => break 'parsing_loop,
                Err(parsing_error) => return Err(parsing_error),
            }

            let mut properties = HashMap::<String, String>::new();
            let mut brushes = Vec::<MapBrush>::new();

            'entity_contents: loop {
                if let Ok((brush, next_tl)) = parse_brush(tl) {
                    tl = next_tl;
                    brushes.push(brush);
                } else if let Ok(((key, value), next_tl)) = parse_property(tl) {
                    tl = next_tl;
                    _ = properties.insert(key, value);
                } else {
                    break 'entity_contents;
                }
            }

            (_, tl) = parse_literal(tl, "}")?;

            entities.push(MapEntity { brushes, properties });
        }

        return Ok(Map { entities });
    }

    /// Find best fitting UV from pre-defined candidate set
    pub fn find_texture_uv(normal: Vec3f) -> (Vec3f, Vec3f) {
        // Candidate array contains [normal, u, v] triplets.
        const UV_CANDIDADTES: [[Vec3f; 3]; 6] = [
            [
                Vec3f::new(0.0, 0.0, 1.0),
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, -1.0, 0.0),
            ], // floor
            [
                Vec3f::new(0.0, 0.0, -1.0),
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, -1.0, 0.0),
            ], // ceiling
            [
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, 1.0, 0.0),
                Vec3f::new(0.0, 0.0, -1.0),
            ], // west wall
            [
                Vec3f::new(-1.0, 0.0, 0.0),
                Vec3f::new(0.0, 1.0, 0.0),
                Vec3f::new(0.0, 0.0, -1.0),
            ], // east wall
            [
                Vec3f::new(0.0, 1.0, 0.0),
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, 0.0, -1.0),
            ], // south wall
            [
                Vec3f::new(0.0, -1.0, 0.0),
                Vec3f::new(1.0, 0.0, 0.0),
                Vec3f::new(0.0, 0.0, -1.0),
            ], // north wall
        ];

        let mut best_dot = f32::MIN;
        let mut best_candidate = &UV_CANDIDADTES[0];

        for candidate in &UV_CANDIDADTES {
            let dot = candidate[0] ^ normal;

            if dot >= best_dot {
                best_dot = dot;
                best_candidate = candidate;
            }
        }

        (best_candidate[1], best_candidate[2])
    }

    /// Build Q1 map into WMAP format
    pub fn build_wmap(&self) -> super::Map {
        let mut entities = Vec::<super::Entity>::new();

        for entity in &self.entities {
            let mut brushes = Vec::<super::Brush>::new();
            let properties = entity.properties.clone();

            for brush in &entity.brushes {
                let mut faces = Vec::<super::BrushFace>::new();

                for face in &brush.faces {
                    let normal = ((face.p0 - face.p1) % (face.p2 - face.p1)).normalized();

                    let (u, v) = Self::find_texture_uv(normal);

                    let rcos = face.texture_rotation.to_radians().cos();
                    let rsin = face.texture_rotation.to_radians().sin();

                    let (u, v) = (
                        u * rcos - v * rsin,
                        u * rsin + v * rcos,
                    );

                    faces.push(super::BrushFace {
                        plane: geom::Plane::from_points(face.p1, face.p0, face.p2),
                        u: geom::Plane {
                            normal: u / face.texture_scale_x,
                            distance: face.texture_offset_x,
                        },
                        v: geom::Plane {
                            normal: v / face.texture_scale_y,
                            distance: face.texture_offset_y,
                        },
                        mtl_name: face.texture_name.clone(),
                    });
                }

                brushes.push(super::Brush { faces });
            }

            entities.push(super::Entity { brushes, properties });
        }

        super::Map { entities }
    }
}

// q1.rs
