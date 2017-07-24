#![allow(unused_variables,dead_code,unused_mut,unused_imports)]
extern crate geo;
extern crate protobuf;
extern crate flate2;

use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;

use std::collections::HashMap;
use geo::{Geometry, Point, MultiPoint};
use flate2::FlateReadExt;

mod vector_tile;

#[derive(Debug)]
pub struct Feature {
    // The geometry is really integers
    geometry: Geometry<f32>,
    properties: HashMap<String, Value>,
}

impl Feature {
    fn get_point<'a>(&'a self) -> Option<Point<f32>> {
        match self.geometry {
            Geometry::Point(p) => Some(p),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    name: String,
    features: Vec<Feature>,
    extent: u32,
}

#[derive(Debug)]
pub struct Tile {
    layers: Vec<Layer>,
}

#[derive(Debug,Clone)]
pub enum Value {
    String(String),
    Float(f32),
    Double(f64),
    Int(i64),
    UInt(u64),
    SInt(i64),
    Boolean(bool),
    Unknown,
}

// TODO add TryFrom for all the base types.

impl Value {
    fn to_string(self) -> Option<String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    // TODO add more getters
}

impl Tile {
    pub fn from_file(filename: &str) -> Tile {
        let mut file = File::open(filename).unwrap();
        let mut contents: Vec<u8> = Vec::new();
        file.read_to_end(&mut contents).unwrap();

        // FIXME if the gzip file is empty then return something sensible
        
        // unzip
        let mut bytes: Vec<u8> = Vec::with_capacity(contents.len());
        let cursor = Cursor::new(contents);
        let mut contents: Vec<u8> = Vec::new();
        cursor.gz_decode().unwrap().read_to_end(&mut contents).unwrap();

        let mut tile: vector_tile::Tile = protobuf::parse_from_bytes(&contents).unwrap();

        pbftile_to_tile(tile)
        //tile.into()

    }

}

impl From<vector_tile::Tile_Value> for Value {
    fn from(val: vector_tile::Tile_Value) -> Value {
        if val.has_string_value() {
            Value::String(val.get_string_value().into())
        } else if val.has_float_value() {
            Value::Float(val.get_float_value())
        } else if val.has_double_value() {
            Value::Double(val.get_double_value())
        } else if val.has_int_value() {
            Value::Int(val.get_int_value())
        } else if val.has_uint_value() {
            Value::UInt(val.get_uint_value())
        } else if val.has_sint_value() {
            Value::SInt(val.get_sint_value())
        } else if val.has_bool_value() {
            Value::Boolean(val.get_bool_value())
        } else {
            Value::Unknown
        }

    }

}

// FIXME use std::convert types, but it needs the mut
fn pbftile_to_tile(mut tile: vector_tile::Tile) -> Tile {
    Tile{ layers: tile.take_layers().into_iter().map(|l| pbflayer_to_layer(l)).collect() }
}

fn pbflayer_to_layer(mut layer: vector_tile::Tile_Layer) -> Layer {
    let name = layer.take_name();
    let extent = layer.get_extent();
    let features = layer.take_features();
    let keys = layer.take_keys();
    let values = layer.take_values();

    let features: Vec<Feature> = features.into_iter().map(|mut f| {
        // TODO do we need the clone on values? I get a 'cannot move out of indexed context'
        // otherwise
        let properties: HashMap<String, Value> = f.take_tags().chunks(2).map(|kv: &[u32]| (keys[kv[0] as usize].clone(), values[kv[1] as usize].clone().into())).collect();

        Feature { properties: properties, geometry: decode_geom(f.get_geometry(), &f.get_field_type()) }
    }).collect();
    
    Layer{ name: name, extent: extent, features: features }

}

enum DrawingCommand {
    MoveTo(Vec<(i32, i32)>),
    LineTo(Vec<(i32, i32)>),
    ClosePath,
}

fn decode_drawing_commands(data: &[u32]) -> Vec<DrawingCommand> {
    let mut res = Vec::new();

    let mut idx = 0;
    loop {
        if idx >= data.len() {
            break;
        }

        let command = data[idx];
        idx += 1;
        let id = command & 0x7;
        let count = command >> 3;

        // Only MoveTo and LineTo take params
        let mut params = Vec::with_capacity(count as usize);
        if id == 1 || id == 2 {
            for _ in 0..count {
                // XXX what about overflows here??
                let value = data[idx] as i32;
                idx += 1;
                let d_x = (value >> 1) ^ (-(value & 1));

                // XXX what about overflows here??
                let value = data[idx] as i32;
                idx += 1;
                let d_y = (value >> 1) ^ (-(value & 1));

                params.push((d_x, d_y));
            }
        }

        res.push(match id {
            1 => DrawingCommand::MoveTo(params),
            2 => DrawingCommand::LineTo(params),
            7 => DrawingCommand::ClosePath,
            _ => unreachable!(),
        });
    }

    res
}

fn decode_geom(data: &[u32], geom_type: &vector_tile::Tile_GeomType) -> Geometry<f32> {
    let drawing_commands = decode_drawing_commands(data);

    match *geom_type {
        vector_tile::Tile_GeomType::POINT => {
            let commands = decode_drawing_commands(data);
            // Specs define this, but maybe we should make this return a Result instead of an
            // assertion
            assert_eq!(commands.len(), 1);
            match commands[0] {
                DrawingCommand::MoveTo(ref points) => {
                    if points.len() == 0 {
                        unreachable!()
                    } else if points.len() == 1 {
                        Geometry::Point(Point::new(points[0].0 as f32, points[0].1 as f32))
                    } else if points.len() > 1 {
                        let mut cx = 0;
                        let mut cy = 0;
                        let mut new_points = Vec::with_capacity(points.len());
                        for p in points.into_iter() {
                            cx = cx + p.0;
                            cy = cy + p.1;
                            new_points.push(Point::new(cx as f32, cy as f32));
                        }
                        Geometry::MultiPoint(MultiPoint(new_points))
                    } else {
                        unreachable!();
                    } 
                }
                _ => { unreachable!() },
            }
        },
        vector_tile::Tile_GeomType::LINESTRING => {
            unimplemented!()
        },
        vector_tile::Tile_GeomType::POLYGON => {
            unimplemented!()
        },
        vector_tile::Tile_GeomType::UNKNOWN => unreachable!(),
    }
}
