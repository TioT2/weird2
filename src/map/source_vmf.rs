///! Source engine VMF file format parsing module

use std::collections::{BTreeMap, HashMap};

use crate::{geom, math::Vec3f};

/// VMF entry
#[derive(Debug)]
pub struct Entry {
    pub class: String,
    pub properties: BTreeMap<String, String>,
    pub entires: Vec<Entry>,
}

pub struct TokenIterator<'t> {
    rest: &'t str,
}

/// Token
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Token<'t> {
    String(&'t str),
    Ident(&'t str),
    BrOpen,
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
    /// Parse .VMF file
    pub fn parse_vmf(source: &str) -> Option<Entry> {
        let mut stack = Vec::new();

        let mut token_list = TokenIterator { rest: &source };

        stack.push(Entry {
            class: "".to_string(),
            entires: Vec::new(),
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
                    top.entires.push(entry);
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
                entires: Vec::new(),
                properties: BTreeMap::new(),
            });
        }

        return Some(stack.swap_remove(0));
    }

    /// WMAP building function
    pub fn build_wmap(&self) -> Option<super::Map> {
        let world =self.entires.iter().find(|entry| entry.class == "world")?;

        let solids = world.entires
            .iter()
            .filter(|entry| entry.class == "solid");

        fn parse_plane(str: &str) -> Option<geom::Plane> {
            let mut num_iter = str
                .split(|ch: char| ch == '(' || ch == ')' || ch.is_whitespace())
                .filter_map(|v| {
                    let t = v.trim();

                    if t.is_empty() {
                        return None;
                    }

                    return Some(t);
                });

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

        fn parse_axis(str: &str) -> Option<geom::Plane> {
            let mut num_iter = str
                .split(|ch: char| ch == '[' || ch == ']' || ch.is_whitespace())
                .filter_map(|v| {
                    let t = v.trim();

                    if !t.is_empty() {
                        Some(t)
                    } else {
                        None
                    }
                });

            let x = num_iter.next()?.parse::<f32>().ok()?;
            let y = num_iter.next()?.parse::<f32>().ok()?;
            let z = num_iter.next()?.parse::<f32>().ok()?;
            let d = num_iter.next()?.parse::<f32>().ok()?;
            let s = num_iter.next()?.parse::<f32>().ok()?;

            Some(geom::Plane {
                normal: Vec3f::new(x, y, z) * s,
                distance: d * s,
            })
        }

        let mut brushes = Vec::new();

        for solid in solids {
            let sides = solid.entires
                .iter()
                .filter(|entry| entry.class == "side");

            let mut faces = Vec::new();

            'side_loop: for side in sides {
                let Some(u) = side.properties.get("uaxis") else {
                    continue 'side_loop;
                };

                let Some(v) = side.properties.get("vaxis") else {
                    continue 'side_loop;
                };

                let Some(material) = side.properties.get("material") else {
                    continue 'side_loop;
                };

                let Some(plane) = side.properties.get("plane") else {
                    continue 'side_loop;
                };

                let u = parse_axis(u)?;
                let v = parse_axis(v)?;
                let plane = parse_plane(plane)?;

                faces.push(super::BrushFace {
                    plane,
                    u,
                    v,
                    mtl_name: material.to_string(),
                    is_transparent: false,
                    is_sky: false,
                });
            }

            brushes.push(super::Brush { faces, is_invisible: false });
        }

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

