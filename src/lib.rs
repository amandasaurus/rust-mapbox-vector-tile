#![allow(unused_variables,dead_code,unused_mut,unused_imports)]
extern crate geo;
extern crate protobuf;
extern crate flate2;
extern crate slippy_map_tiles;

use std::fs::File;
use std::io::prelude::*;
use std::io::Cursor;
use protobuf::Message;

use std::collections::{HashMap, HashSet, BTreeMap};
use geo::{Geometry, Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon};
use flate2::FlateReadExt;

mod vector_tile;

#[derive(Debug)]
pub struct Properties(pub HashMap<String, Value>);

//impl From<HashMap<String, Value>> for Properties {
//    fn from(x: HashMap<String, Value>) -> Properties { Properties(x) }
//}


impl<V> From<HashMap<String, V>> for Properties where V: Into<Value> {
    fn from(x: HashMap<String, V>) -> Properties {
        // FIXME is this the most effcient with doesn't mess with duplicating memory
        let x: HashMap<String, Value>  = x.into_iter().map(|(k, v)| (k, v.into())).collect();
        Properties(x)
    }
}

impl Properties {
    pub fn new() -> Self {
        Properties(HashMap::new())
    }

    pub fn insert<K: Into<String>, V: Into<Value>>(&mut self, k: K, v: V)  {
        self.0.insert(k.into(), v.into());
    }

    pub fn set<K: Into<String>, V: Into<Value>>(mut self, k: K, v: V) -> Self {
        self.insert(k, v);
        self
    }
}

#[derive(Debug)]
pub struct Feature {
    // The geometry is really integers
    pub geometry: Geometry<f32>,
    pub properties: Properties,
}

impl Feature {
    pub fn new(geometry: Geometry<f32>, properties: Properties) -> Self {
        Feature{ geometry: geometry, properties: properties }
    }

    pub fn from_geometry(geometry: Geometry<f32>) -> Self {
        Feature::new(geometry, Properties::new())
    }

    pub fn get_point<'a>(&'a self) -> Option<Point<f32>> {
        match self.geometry {
            Geometry::Point(p) => Some(p),
            _ => None,
        }
    }

    pub fn translate_geometry(&mut self, x_func: &Fn(f32)->f32, y_func: &Fn(f32) -> f32) {
        // FIXME why the back and forth and not just set it
        self.geometry = match self.geometry {
            Geometry::Point(mut p) => {
                let x = x_func(p.x());
                let y = y_func(p.y());
                p.set_x(x).set_y(y);
                Geometry::Point(p)
            },
            _ => unimplemented!(),
        };
    }

    pub fn set<K: Into<String>, V: Into<Value>>(mut self, k: K, v: V) -> Self {
        self.properties.insert(k, v);
        self
    }

}

impl From<Geometry<f32>> for Feature {
    fn from(geom: Geometry<f32>) -> Feature {
        Feature::from_geometry(geom)
    }
}


#[derive(Debug)]
pub struct Layer {
    name: String,
    pub features: Vec<Feature>,
    extent: u32,
}

impl Layer {
    pub fn new(name: String) -> Self {
        Layer{ name: name, features: Vec::new(), extent: 4096 }
    }

    pub fn new_and_extent(name: String, extent: u32) -> Self {
        Layer{ name: name, features: Vec::new(), extent: extent }
    }

    pub fn set_locations(&mut self, geometry_tile: &slippy_map_tiles::Tile) {
        let extent = self.extent as f32;
        let top = geometry_tile.top();
        let bottom = geometry_tile.bottom();
        let left = geometry_tile.left();
        let right = geometry_tile.right();
        let width = right - left;
        let height = bottom - top;

        let x_func = |x: f32| { left + (x / extent) * width };
        let y_func = |y: f32| { top + (y / extent) * height };

        for f in self.features.iter_mut() {
            f.translate_geometry(&x_func, &y_func);
        }
        
    }

    pub fn add_feature(&mut self, f: Feature) {
        // TODO error checking for non-integer geom?
        self.features.push(f);
    }
}


// TODO probably some way to merge these
impl From<String> for Layer {
    fn from(name: String) -> Layer {
        Layer::new(name)
    }
}
impl<'a> From<&'a str> for Layer {
    fn from(name: &str) -> Layer {
        Layer::new(name.to_string())
    }
}

impl<'a> From<&'a Layer> for vector_tile::Tile_Layer {
    fn from(l: &Layer) -> vector_tile::Tile_Layer {
        let mut res = vector_tile::Tile_Layer::new();

        res.set_name(l.name.clone());
        res.set_version(2);
        res.set_extent(l.extent);

        // make key index
        let all_keys: Vec<&String> = l.features.iter().flat_map(|f| f.properties.0.keys()).collect();
        let (keys_in_order, number_for_key) = make_index(&all_keys);

        for k in keys_in_order {
            res.mut_keys().push(k.clone());
        }

        let all_values: Vec<&Value> = l.features.iter().flat_map(|f| f.properties.0.values()).collect();
        let (values_in_order, number_for_value) = make_index(&all_values);
        for v in values_in_order {
            res.mut_values().push(v.into());
        }
        //println!("values_in_order {:?} number_for_value {:?}", values_in_order, number_for_value);

        // There's lots of 'cannot move out of borrowed context' errors here
        for f in l.features.iter() {
            let mut pbf_feature = vector_tile::Tile_Feature::new();
            for (k, v) in f.properties.0.iter() {
                pbf_feature.mut_tags().push(number_for_key[&k]);
                pbf_feature.mut_tags().push(number_for_value[&v]);
            }
            // FIXME remove clone
            let geom: DrawingCommands = f.geometry.clone().into();
            pbf_feature.set_geometry(geom.into());

            res.mut_features().push(pbf_feature);
        }

        res
    }
}

fn make_index<'a, T: Ord>(input: &[&'a T]) -> (Vec<&'a T>, BTreeMap<&'a T, u32>) {
    let mut counter = BTreeMap::new();
    for key in input {
        let val = counter.entry(key).or_insert(0);
        *val += 1;
    }
    let mut popular_items: Vec<_> = counter.into_iter().map(|(k, v)| (v, k)).collect();
    popular_items.sort();
    let popular_items = popular_items;
    let number_for_key: BTreeMap<&T, u32> = popular_items.iter().enumerate().map(|(i, &(v, k))| (*k, i as u32)).collect();
    let keys_in_order: Vec<&T> = popular_items.iter().map(|&(v, k)| *k).collect();

    (keys_in_order, number_for_key)
}


#[derive(Debug)]
pub struct Tile {
    pub layers: Vec<Layer>,
}

impl Tile {
    pub fn new() -> Self {
        Tile{ layers: Vec::new() }
    }

    pub fn add_layer<L: Into<Layer>>(&mut self, l: L) {
        let l = l.into();
        // TODO check for duplicate layer name
        self.layers.push(l);

    }

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

    pub fn set_locations(&mut self, geometry_tile: &slippy_map_tiles::Tile) {
        for l in self.layers.iter_mut() {
            l.set_locations(&geometry_tile);
        }
    }

    pub fn add_feature(&mut self, layer_name: &str, f: Feature) {
        // TODO proper error for layer name not found
        let layer = self.layers.iter_mut().filter(|l| l.name == layer_name).nth(0).unwrap();
        layer.add_feature(f);
    }

    pub fn write_to_file(&self, filename: &str)  {
        let converted: vector_tile::Tile = self.into();
        let mut file = File::create(filename).unwrap();
        let mut cos = protobuf::CodedOutputStream::new(&mut file);
        converted.write_to(&mut cos).unwrap();
        cos.flush().unwrap();
        //converted.write_to_bytes().unwrap()
    }

}

impl<'a> From<&'a Tile> for vector_tile::Tile {
    fn from(tile: &Tile) -> vector_tile::Tile {
        let mut result = vector_tile::Tile::new();

        for layer in tile.layers.iter() {
            result.mut_layers().push(layer.into());
        }

        result
    }
}

#[derive(Debug,Clone,PartialEq,PartialOrd)]
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
    pub fn to_string(self) -> Option<String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
    // TODO add more getters
    
    fn repr(&self) -> String {
        match self {
            &Value::String(ref s) => format!("String({:?})", s),
            &Value::Float(ref s) => format!("Float({:?})", s),
            &Value::Double(ref s) => format!("Double({:?})", s),
            &Value::Int(ref s) => format!("Int({:?})", s),
            &Value::UInt(ref s) => format!("UInt({:?})", s),
            &Value::SInt(ref s) => format!("SInt({:?})", s),
            &Value::Boolean(ref s) => format!("Boolean({:?})", s),
            &Value::Unknown => format!("Unknown"),
        }
    }
}

impl Eq for Value {}
impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.repr().cmp(&other.repr())
    }

}

impl From<String> for Value { fn from(x: String) -> Value { Value::String(x) } }
impl<'a> From<&'a str> for Value { fn from(x: &str) -> Value { Value::String(x.to_owned()) } }

impl From<f32> for Value { fn from(x: f32) -> Value { Value::Float(x) } }
impl From<f64> for Value { fn from(x: f64) -> Value { Value::Double(x) } }
impl From<i64> for Value { fn from(x: i64) -> Value { Value::SInt(x) } }
impl From<u64> for Value { fn from(x: u64) -> Value { Value::UInt(x) } }
impl From<bool> for Value { fn from(x: bool) -> Value { Value::Boolean(x) } }


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

impl<'a> From<&'a Value> for vector_tile::Tile_Value {
    fn from(val: &Value) -> vector_tile::Tile_Value {
        let mut res = vector_tile::Tile_Value::new();
        match val {
            &Value::String(ref s) => res.set_string_value(s.clone()),
            &Value::Float(s) => res.set_float_value(s),
            &Value::Double(s) => res.set_double_value(s),
            &Value::Int(s) => res.set_int_value(s),
            &Value::UInt(s) => res.set_uint_value(s),
            &Value::SInt(s) => res.set_sint_value(s),
            &Value::Boolean(s) => res.set_bool_value(s),
            &Value::Unknown => {},
        }

        res
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

        Feature { properties: properties.into(), geometry: decode_geom(f.get_geometry(), &f.get_field_type()) }
    }).collect();
    
    Layer{ name: name, extent: extent, features: features }

}


#[derive(Debug,Clone)]
pub enum DrawingCommand {
    MoveTo(Vec<(i32, i32)>),
    LineTo(Vec<(i32, i32)>),
    ClosePath,
}

#[derive(Debug,Clone)]
pub struct DrawingCommands(pub Vec<DrawingCommand>);

impl<'a> From<&'a [u32]> for DrawingCommands {
    fn from(data: &[u32]) -> DrawingCommands {
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

        DrawingCommands(res)
    }
}

#[derive(Debug,PartialEq,Eq,Clone)]
enum WindingOrder { Clockwise, AntiClockwise }

/// Returns the winding order of the ring. presumes a valid ring
fn winding_order(ring: &LineString<f32>) -> WindingOrder {
    // FIXME can we use .windows here instead?
    let res: f32 = (0..ring.0.len()-1).map(|i| ( ring.0[i].x()*ring.0[i+1].y()  - ring.0[i+1].x()*ring.0[i].y() ) ).sum();
    if res < 0. {
        WindingOrder::AntiClockwise
    } else {
        WindingOrder::Clockwise
    }
}


fn decode_geom(data: &[u32], geom_type: &vector_tile::Tile_GeomType) -> Geometry<f32> {

    let commands: DrawingCommands = data.into();
    match *geom_type {
        vector_tile::Tile_GeomType::POINT => {
            // Specs define this, but maybe we should make this return a Result instead of an
            // assertion
            assert_eq!(commands.0.len(), 1);
            match commands.0[0] {
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
            let mut cx = 0;
            let mut cy = 0;
            assert_eq!(commands.0.len() % 2, 0);
            let mut lines: Vec<LineString<f32>> = Vec::with_capacity(commands.0.len()/2);
            for cmds in commands.0.chunks(2) {
                assert_eq!(cmds.len(), 2);

                let mut linestring_points = Vec::new();

                if let DrawingCommand::MoveTo(ref points) = cmds[0] {
                    assert_eq!(points.len(), 1);
                    let (dx, dy) = points[0];
                    cx += dx;
                    cy += dy;
                    linestring_points.push(Point::new(cx as f32, cy as f32));
                } else {
                    assert!(false);
                }
                if let DrawingCommand::LineTo(ref points) = cmds[1] {
                    assert!(points.len() > 0);
                    linestring_points.reserve(points.len());
                    for &(dx, dy) in points.into_iter() {
                        cx += dx;
                        cy += dy;
                        linestring_points.push(Point::new(cx as f32, cy as f32));
                    }
                } else {
                    assert!(false);
                }

                lines.push(LineString(linestring_points));
            }

            assert!(lines.len() > 0);
            if lines.len() == 1 {
                Geometry::LineString(lines.remove(0))
            } else {
                Geometry::MultiLineString(MultiLineString(lines))
            }

        },
        vector_tile::Tile_GeomType::POLYGON => {
            let mut cx = 0;
            let mut cy = 0;
            assert_eq!(commands.0.len() % 3, 0);
            let mut rings  = Vec::with_capacity(commands.0.len()/3);
            //println!("\nNew\nCommands: {:?}", commands.0);
            for cmds in commands.0.chunks(3) {
                assert_eq!(cmds.len(), 3);
                let mut linestring_points = Vec::new();

                if let DrawingCommand::MoveTo(ref points) = cmds[0] {
                    assert_eq!(points.len(), 1);
                    let (dx, dy) = points[0];
                    cx += dx;
                    cy += dy;
                    linestring_points.push(Point::new(cx as f32, cy as f32));
                } else {
                    assert!(false);
                }
                if let DrawingCommand::LineTo(ref points) = cmds[1] {
                    assert!(points.len() > 1);
                    linestring_points.reserve(points.len());
                    for &(dx, dy) in points.into_iter() {
                        cx += dx;
                        cy += dy;
                        linestring_points.push(Point::new(cx as f32, cy as f32));
                    }
                } else {
                    assert!(false);
                }
                if let DrawingCommand::ClosePath = cmds[2] {
                    let line = LineString(linestring_points);
                    let winding_order = winding_order(&line);
                    rings.push((line, winding_order));
                } else {
                    assert!(false);
                }
            }

            assert!(rings.len() > 0);
            let mut polygons = Vec::new();
            loop {
                if rings.len() == 0 {
                    break;
                }
                //println!("rings {:?}", rings.iter().map(|ref x| x.1.clone()).collect::<Vec<WindingOrder>>());
                let (exterior_ring, winding_order) = rings.remove(0);
                // FIXME put this back in
                //assert_eq!(winding_order, WindingOrder::Clockwise);
                let mut inner_rings = Vec::new();
                if rings.len() > 0 {
                    loop {
                        if rings.len() == 0 || rings[0].1 == WindingOrder::Clockwise  {
                            break;
                        }
                        assert!(rings.len() > 0);
                        inner_rings.push(rings.remove(0).0);
                    }
                }
                polygons.push(Polygon::new(exterior_ring, inner_rings));
            }
            assert!(polygons.len() > 0);

            if polygons.len() == 1 {
                // FIXME spec diff?
                Geometry::Polygon(polygons.remove(0))
            } else {
                Geometry::MultiPolygon(MultiPolygon(polygons))
            }
        },
        vector_tile::Tile_GeomType::UNKNOWN => unreachable!(),
    }
}

// FIXME these 2 conversions should be TryFrom which throw an error for non-integer coords
impl From<Point<f32>> for DrawingCommands {
    fn from(point: Point<f32>) -> DrawingCommands {
        DrawingCommands(vec![DrawingCommand::MoveTo(vec![(point.x().trunc() as i32, point.y().trunc() as i32)])])
    }
}

impl From<MultiPoint<f32>> for DrawingCommands {
    fn from(mp: MultiPoint<f32>) -> DrawingCommands {
        let mut offsets = Vec::with_capacity(mp.0.len());
        let (mut cx, mut cy) = (0, 0);
        for p in mp.0 {
            let x = p.x().trunc() as i32;
            let y = p.y().trunc() as i32;
            let dx = x - cx;
            let dy = y - cy;
            cx = x;
            cy = y;
            offsets.push((dx, dy));
        }

        DrawingCommands(vec![DrawingCommand::MoveTo(offsets)])
    }
}

impl From<Geometry<f32>> for DrawingCommands {
    fn from(g: Geometry<f32>) -> DrawingCommands {
        match g {
            Geometry::Point(x) => x.into(),
            Geometry::MultiPoint(x) => x.into(),
            _ => unimplemented!(),
        }
    }

}

impl From<DrawingCommands> for Vec<u32> {
    fn from(dc: DrawingCommands) -> Vec<u32> {
        let mut res = Vec::with_capacity(dc.0.len());
        for cmd in dc.0 {
            match cmd {
                DrawingCommand::MoveTo(points) => {
                    res.reserve(1 + points.len()*2);
                    let count = points.len() as u32;
                    let id: u32 = 1;
                    let cmd_int = (id & 0x7) | (count << 3);
                    res.push(cmd_int);
                    for (dx, dy) in points {
                        let dx = ((dx << 1) ^ (dx >> 31)) as u32;
                        let dy = ((dy << 1) ^ (dy >> 31)) as u32;
                        res.push(dx);
                        res.push(dy);
                    }
                },
                _ => unimplemented!(),
            }
        }
        res
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn test_creation() {
        use super::*;

        let mut t = Tile::new();
        let l: Layer = Layer::new("fart".to_string());
        let l: Layer = "hello".into();

        assert_eq!(t.layers.len(), 0);
        t.add_layer("poop");
        assert_eq!(t.layers.len(), 1);
        assert_eq!(t.layers[0].name, "poop");
        assert_eq!(t.layers[0].extent, 4096);

        t.add_feature("poop", Feature::new(Geometry::Point(Point::new(10., 10.)), Properties::new().set("name", "fart")));
    }

    #[test]
    fn test_properties() {
        use super::*;
        let mut p = Properties::new();
        p.insert("name", "bar");
        p.insert("visible", false);

        let p = Properties::new().set("foo", "bar").set("num", 10i64).set("visible", false);
    }

    #[test]
    fn test_geom_encoding() {
        use super::*;

        let p = Point::new(25., 17.);
        let dc: DrawingCommands = p.into();
        let bytes: Vec<u32> = dc.into();
        assert_eq!(bytes, vec![9, 50, 34]);

        let mp = MultiPoint(vec![Point::new(5., 7.), Point::new(3., 2.)]);
        let dc: DrawingCommands = mp.into();
        let bytes: Vec<u32> = dc.into();
        assert_eq!(bytes, vec![17, 10, 14, 3, 9]);
    }

}
