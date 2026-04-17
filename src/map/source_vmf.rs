//! Source engine VMF format parser

use std::collections::{BTreeMap, HashMap};

use crate::{geom, math::Vec3f};

/// VMF entry
#[derive(Debug)]
pub struct Entry {
    /// Class of the entry
    pub class: String,

    /// Set of entry properties
    pub properties: BTreeMap<String, String>,

    /// Sub-entry list
    pub entries: Vec<Entry>,
}

/// Token iterator structure
pub struct TokenIterator<'t> {
    rest: &'t str,
}

/// Token
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Token<'t> {
    /// Some bracket-enclosed string
    String(&'t str),

    /// Identifier
    Ident(&'t str),

    /// Opening curly brace
    BrOpen,

    /// Closing curly brace
    BrClose,
}

impl<'t> Iterator for TokenIterator<'t> {
    type Item = Token<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        // Trim heading whitespace
        self.rest = self.rest.trim_start();

        if self.rest.starts_with('{') {
            self.rest = &self.rest[1..];
            return Some(Token::BrOpen);
        }

        if self.rest.starts_with('}') {
            self.rest = &self.rest[1..];
            return Some(Token::BrClose);
        }

        if self.rest.starts_with('\"') {
            let rest = &self.rest[1..];
            let end_index = rest.find('\"')?;

            let str = &rest[..end_index];

            self.rest = &rest[end_index + 1..];

            return Some(Token::String(str));
        }

        let end = self.rest.find(|c: char| c.is_whitespace())?;
        let res = &self.rest[..end];
        self.rest = &self.rest[end..];

        Some(Token::Ident(res))
    }
}

impl Entry {
    /// Parse .VMF file contents
    pub fn parse_vmf(source: &str) -> Option<Entry> {
        let mut stack = Vec::new();

        let mut token_list = TokenIterator { rest: source };

        stack.push(Entry {
            class: "".to_string(),
            entries: Vec::new(),
            properties: BTreeMap::new(),
        });

        'parsing_loop: loop {
            let Some(entry) = stack.last_mut() else {
                break 'parsing_loop;
            };

            let Some(token) = token_list.next() else {
                break 'parsing_loop;
            };

            if token == Token::BrClose {
                let entry = stack.pop().unwrap();
                if let Some(top) = stack.last_mut() {
                    top.entries.push(entry);
                } else {
                    return None;
                }

                continue 'parsing_loop;
            }

            if let Token::String(token) = token {
                let Token::String(value) = token_list.next()? else {
                    return None;
                };

                entry.properties.insert(token.to_string(), value.to_string());
                continue 'parsing_loop;
            }

            let start = token_list.next()?;

            if start != Token::BrOpen {
                return None;
            }

            let Token::Ident(class) = token else {
                return None;
            };

            stack.push(Entry {
                class: class.to_string(),
                entries: Vec::new(),
                properties: BTreeMap::new(),
            });
        }

        Some(stack.swap_remove(0))
    }

    /// WMAP building function
    pub fn build_wmap(&self) -> Option<super::Map> {

        /// Split by whitespace
        fn num_iter(str: &str, left_br: char, right_br: char) -> impl Iterator<Item = &str> {
            str
                .split(move |ch: char| ch == left_br || ch == right_br || ch.is_whitespace())
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
        }

        /// Parse plane from String slice
        fn parse_plane(str: &str) -> Option<geom::Plane> {
            let mut num_iter = num_iter(str, '(', ')');

            let mut parse_vec = || {
                let x1 = num_iter.next()?.parse::<f32>().ok()?;
                let y1 = num_iter.next()?.parse::<f32>().ok()?;
                let z1 = num_iter.next()?.parse::<f32>().ok()?;

                Some(Vec3f::new(x1, y1, z1))
            };

            let v1 = parse_vec()?;
            let v2 = parse_vec()?;
            let v3 = parse_vec()?;

            Some(geom::Plane::from_points(v2, v1, v3))
        }

        /// Parse axis from string slice
        fn parse_axis(str: &str) -> Option<geom::Plane> {
            let mut num_iter = num_iter(str, '[', ']');

            let x = num_iter.next()?.parse::<f32>().ok()?;
            let y = num_iter.next()?.parse::<f32>().ok()?;
            let z = num_iter.next()?.parse::<f32>().ok()?;
            let d = num_iter.next()?.parse::<f32>().ok()?;
            let s = num_iter.next()?.parse::<f32>().ok()?;

            Some(geom::Plane {
                normal: Vec3f::new(x, y, z) * s.into(),
                distance: d * s,
            })
        }

        /// Parse brush face from side node
        fn parse_face(side: &Entry) -> Option<super::BrushFace> {
            let u = side.properties.get("uaxis").map(String::as_str).and_then(parse_axis)?;
            let v = side.properties.get("vaxis").map(String::as_str).and_then(parse_axis)?;
            let mtl_name = side.properties.get("material")?.to_string();
            let plane = side.properties.get("plane").map(String::as_str).and_then(parse_plane)?;

            // Set flags based on surface properties
            let flags = super::BrushFaceFlags::empty()
                ;

            Some(super::BrushFace { plane, u, v, mtl_name, flags })
        }

        // Find world node
        let world = self.entries.iter().find(|entry| entry.class == "world")?;

        // For all entries collect their sides converted to faces to vec and build brushes and collect them to vec
        let brushes = world.entries
            .iter()
            .filter(|e| e.class == "solid")
            .map(|s| s.entries.iter()
                .filter(|e| e.class == "side")
                .filter_map(parse_face)
                .collect::<Vec<_>>())
            .map(|faces| super::Brush { faces, flags: super::BrushFlags::empty() })
            .collect::<Vec<_>>();

        Some(super::Map {
            entities: vec![super::Entity {
                brushes,
                properties: HashMap::from_iter(vec![
                    ("classname".to_string(), "worldspawn".to_string())
                ].into_iter()),
            }],
        })
    }
}
