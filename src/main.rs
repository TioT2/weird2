use std::{cell::Cell, collections::{BTreeMap, BTreeSet, HashMap}};

use math::Vec3f;
use sdl2::{event::Event, keyboard::Scancode};

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

/// Binary space partition implementation
pub mod bsp;

/// Basic geometry
pub mod geom;

/// New map implementation
pub mod map;

#[derive(Copy, Clone)]
pub struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new() -> Self {
        Self { state: 1 }
    }

    pub fn next(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state <<  5;
        self.state
    }
}

pub struct Timer {
    start: std::time::Instant,
    now: std::time::Instant,
    dt: std::time::Duration,

    fps_duration: std::time::Duration,
    fps_last_measure: std::time::Instant,
    fps_frame_counter: usize,
    fps: Option<f32>,
}

impl Timer {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            now,
            dt: std::time::Duration::from_millis(60),

            fps_duration: std::time::Duration::from_millis(1000),
            fps_last_measure: now,
            fps_frame_counter: 0,
            fps: None,
        }
    }

    pub fn response(&mut self) {
        let new_now = std::time::Instant::now();
        self.dt = new_now.duration_since(self.now);
        self.now = new_now;
        
        self.fps_frame_counter += 1;

        let fps_delta = self.now - self.fps_last_measure;
        if fps_delta > self.fps_duration {
            self.fps = Some(self.fps_frame_counter as f32 / fps_delta.as_secs_f32());

            self.fps_last_measure = self.now;
            self.fps_frame_counter = 0;
        }
    }

    pub fn get_delta_time(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    pub fn get_time(&self) -> f32 {
        self.now
            .duration_since(self.start)
            .as_secs_f32()
    }

    pub fn get_fps(&self) -> f32 {
        self.fps.unwrap_or(std::f32::NAN)
    }
}

#[derive(Copy, Clone)]
pub struct KeyState {
    pub pressed: bool,
    pub changed: bool,
}

pub struct Input {
    states: HashMap<Scancode, KeyState>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    pub fn get_key_state(&self, key: Scancode) -> KeyState {
        self.states
            .get(&key)
            .copied()
            .unwrap_or(KeyState {
                pressed: false,
                changed: false,
            })
    }

    pub fn is_key_pressed(&self, key: Scancode) -> bool {
        self.get_key_state(key).pressed
    }

    pub fn is_key_clicked(&self, key: Scancode) -> bool {
        let key = self.get_key_state(key);

        key.pressed && key.changed
    }

    pub fn on_state_changed(&mut self, key: Scancode, pressed: bool) {
        if let Some(state) = self.states.get_mut(&key) {
            state.changed = state.pressed != pressed;
            state.pressed = pressed;
        } else {
            self.states.insert(key, KeyState {
                pressed,
                changed: true,
            });
        }
    }

    pub fn release_changed(&mut self) {
        for state in self.states.values_mut() {
            state.changed = false;
        }
    }

}

#[derive(Copy, Clone)]
pub struct Camera {
    pub location: Vec3f,
    pub direction: Vec3f,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            location: vec3f!(10.0, 10.0, 10.0),
            direction: vec3f!(-0.544, -0.544, -0.544).normalized(),
        }
    }

    pub fn response(&mut self, timer: &Timer, input: &Input) {
        let mut movement = vec3f!(
            (input.is_key_pressed(Scancode::W) as i32 - input.is_key_pressed(Scancode::S) as i32) as f32,
            (input.is_key_pressed(Scancode::D) as i32 - input.is_key_pressed(Scancode::A) as i32) as f32,
            (input.is_key_pressed(Scancode::R) as i32 - input.is_key_pressed(Scancode::F) as i32) as f32,
        );
        let mut rotation = vec2f!(
            (input.is_key_pressed(Scancode::Right) as i32 - input.is_key_pressed(Scancode::Left) as i32) as f32,
            (input.is_key_pressed(Scancode::Down ) as i32 - input.is_key_pressed(Scancode::Up  ) as i32) as f32,
        );

        movement *= timer.get_delta_time() * 256.0;
        rotation *= timer.get_delta_time() * 1.5;

        let dir = self.direction;
        let right = (dir % vec3f!(0.0, 0.0, 1.0)).normalized();
        let up = (right % dir).normalized();

        self.location += dir * movement.x + right * movement.y + up * movement.z;

        let mut azimuth = dir.z.acos();
        let mut elevator = dir.y.signum() * (
            dir.x / (
                dir.x * dir.x +
                dir.y * dir.y
            ).sqrt()
        ).acos();

        elevator -= rotation.x;
        azimuth  += rotation.y;

        azimuth = azimuth.clamp(0.01, std::f32::consts::PI - 0.01);

        self.direction = vec3f!(
            azimuth.sin() * elevator.cos(),
            azimuth.sin() * elevator.sin(),
            azimuth.cos(),
        );
    }
}

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    // yay, this code will not compile on non-local builds)))
    // bit of functional rust code)
    // --
    let location_map = map::builder::Map::parse(include_str!("../temp/e1m1.map")).unwrap();

    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("WEIRD-2", 1024, 768)
        .opengl()
        .build()
        .unwrap()
    ;

    let gl_context = window.gl_create_context().unwrap();
    window.gl_make_current(&gl_context).unwrap();

    let mut timer = Timer::new();
    let mut input = Input::new();
    let mut camera = Camera::new();

    // camera.location = Vec3f::new(-200.0, 2000.0, -50.0);
    // camera.location = Vec3f::new(-10.0, -30.0, 150.0);
    camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // let bsp = {
    //     let polygons = brush::get_map_polygons(&brushes, false);
    //     bsp::Bsp::build(&polygons)
    // };

    let mut builder = map::builder::Builder::new();
    builder.start_build_volumes(location_map.build_world_physical_polygons());

    // Perfrom volume face coplanarity check
    #[cfg(not)]
    {
        for volume in &builder.volumes {
            for face in &volume.faces {
                let main_polygon = &face.polygon;
                for physical_polygon in &face.physical_polygons {
                    if physical_polygon.polygon.plane != main_polygon.plane {
                        eprintln!("Coplanarity check failed");
                    }
                }
            }
        }
        eprintln!("Coplanarity check finished.");
    }

    builder.start_resolve_portals();
    // builder.start_remove_invisible();

    // Use origin set to remove all invisibles
    builder.start_remove_invisible(location_map.get_all_origins());

    #[cfg(not)]
    {
        let mut noportal_count = 0;
        for volume in &builder.volumes {
            let portal_count  = volume.faces
                .iter()
                .map(|f| f.portal_polygons.len())
                .sum::<usize>();
            if portal_count == 0 {
                noportal_count += 1;
            }
        }
        println!("Volumes without portals count: {}/{}", noportal_count, builder.volumes.len());
    }

    // Calculate average projected point count
    #[cfg(not)]
    {
        let mut point_count = 0;
        let mut face_count = 0;

        for volume in &builder.volumes {
            face_count += volume.faces.len();

            for face in &volume.faces {
                for polygon in &face.portal_polygons {
                    point_count += builder.portal_polygons[polygon.dst_volume_index].points.len();
                }

                for polygon in &face.physical_polygons {
                    point_count += polygon.polygon.points.len();
                }
            }
        }

        println!("Average points-per-face: {}", point_count as f32 / face_count as f32);
    }

    let mut do_sync_logical_camera = true;
    let mut do_enable_depth_test = true;
    let mut do_enable_slow_rendering = false;

    let mut logical_camera = camera;

    // let projection_matrix = math::Mat4f::projection_frustum(-0.5, 0.5, -0.5, 0.5, 0.25, 8192.0);
    // let mut view_matrix = math::Mat4f::view(
    //     camera.location,
    //     camera.location + camera.direction,
    //     Vec3f::new(0.0, 0.0, 1.0)
    // );

    struct PortalGraph {
        pub vertices: Vec<Vec3f>,
        pub edges: Vec<(u32, u32)>,
    }

    let portal_graph = 'build_portal_graph: {
        #[cfg(not)]
        {
            let get_volume_center = |volume: &map::builder::HullVolume| -> Vec3f {
                let mut collector = Vec3f::zero();
    
                for face in &volume.faces {
                    let mut face_collector = Vec3f::zero();
    
                    for point in &face.polygon.points {
                        face_collector += *point;
                    }
    
                    collector += face_collector / face.polygon.points.len() as f32;
                }
    
                collector / volume.faces.len() as f32
            };
    
            
            let volume_centers = Vec::from_iter(builder.volumes.iter().map(get_volume_center));
            
            let mut polygon_ids = BTreeSet::<usize>::from_iter(0..builder.volumes.len());
            let mut edges = Vec::<(u32, u32)>::new();
    
            while let Some(first_id) = polygon_ids.first().copied() {
                let mut visited_volume_ids = BTreeSet::<usize>::new();
                let mut edge_polygons = BTreeSet::new();
    
                edge_polygons.insert(first_id);
    
                while !edge_polygons.is_empty() {
                    let mut new_edge_polygons = BTreeSet::new();
    
                    for volume_index in edge_polygons {
                        let volume = &builder.volumes[volume_index];
    
                        for face in &volume.faces {
                            'portal_loop: for portal in &face.portal_polygons {
                                let dst_index = portal.dst_volume_index;
    
                                if visited_volume_ids.contains(&dst_index) {
                                    continue 'portal_loop;
                                }
    
                                edges.push((volume_index as u32, dst_index as u32));
    
                                new_edge_polygons.insert(dst_index);
                            }
                        }
    
                        visited_volume_ids.insert(volume_index);
                    }
    
                    edge_polygons = new_edge_polygons;
                }
    
                for v in visited_volume_ids {
                    polygon_ids.remove(&v);
                }
            }
    
            break 'build_portal_graph PortalGraph {
                edges,
                vertices: volume_centers
            };
        }

        #[allow(unreachable_code)]
        {
            break 'build_portal_graph PortalGraph {
                vertices: Vec::new(),
                edges: Vec::new()
            };
        }
    };

    'main_loop: loop {
        input.release_changed();

        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Quit { .. } => {
                    break 'main_loop;
                }
                Event::KeyUp { scancode, .. } => {
                    if let Some(code) = scancode {
                        input.on_state_changed(code, false);
                    }
                }
                Event::KeyDown { scancode, .. } => {
                    if let Some(code) = scancode {
                        input.on_state_changed(code, true);
                    }
                }
                _ => {}
            }
        }

        timer.response();
        camera.response(&timer, &input);

        // Synchronize logical camera with physical one
        if input.is_key_clicked(Scancode::Backslash) {
            do_sync_logical_camera = !do_sync_logical_camera;
        }

        if input.is_key_clicked(Scancode::Num0) {
            do_enable_depth_test = !do_enable_depth_test;
        }

        if input.is_key_clicked(Scancode::Minus) {
            do_enable_slow_rendering = !do_enable_slow_rendering;
        }

        if do_sync_logical_camera {
            logical_camera = camera;
        }

        println!("FPS: {}, DB={}, ", timer.get_fps(), do_enable_depth_test as u32);

        unsafe {
            if do_enable_depth_test {
                glu_sys::glEnable(glu_sys::GL_DEPTH_TEST);
            } else {
                glu_sys::glDisable(glu_sys::GL_DEPTH_TEST);
            }

            if do_enable_depth_test {
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT | glu_sys::GL_DEPTH_BUFFER_BIT);
            } else {
                glu_sys::glClear(glu_sys::GL_COLOR_BUFFER_BIT);
            }

            glu_sys::glClearColor(0.30, 0.47, 0.80, 0.0);

            glu_sys::glLoadIdentity();
            glu_sys::gluPerspective(60.0, 8.0 / 6.0, 0.01, 8192.0);

            glu_sys::gluLookAt(
                camera.location.x as f64,
                camera.location.y as f64,
                camera.location.z as f64,
                camera.location.x as f64 + camera.direction.x as f64,
                camera.location.y as f64 + camera.direction.y as f64,
                camera.location.z as f64 + camera.direction.z as f64,
                0.0,
                0.0,
                1.0
            );

            // println!("camera: <{}, {}, {}>", camera.location.x, camera.location.y, camera.location.z);

            // -- Render by BSP
            #[cfg(not)]
            'render: {
                // println!("Rendering started!");

                let start_volume_index_opt = builder.volume_bsp
                    .as_ref()
                    .unwrap()
                    .traverse(logical_camera.location);

                let Some(start_volume_index) = start_volume_index_opt else {
                    break 'render;
                };

                // !!!FOR TOMORROW!!!

                // Idea: sort 'builder.volume_bsp' in BSP traversal order. VolumeBSP is granting

                // let start_volume = builder.volumes.get(start_volume_index).unwrap();

                struct RenderContext<'t> {
                    camera_location: Vec3f,
                    camera_plane: geom::Plane,
                    builder: &'t map::builder::Builder,
                    depth: usize,

                    render_set: BTreeSet<usize>,
                    set_miss_count: usize,
                }

                impl<'t> RenderContext<'t> {
                    /// Render all volumes starting from
                    pub fn render(&mut self, start_volume_index: usize) {
                        #[cfg(not)]
                        {

                        // Set of all rendered volumes (volume_id -> (parent_count, weight))
                        let mut rendered_volume_set =
                            BTreeMap::<usize, (Cell<BTreeSet<usize>>, Cell<u32>)>::new();

                        // Volume rendering tree
                        // let mut render_tree = BTreeMap::<usize, (BTreeSet<usize>, BTreeSet<usize>)>::new();

                        // usize -> ({usize}, {usize})

                        // Render list building edge
                        let mut edge = BTreeSet::<usize>::new();

                        edge.insert(start_volume_index);

                        while !edge.is_empty() {
                            let mut new_edge: BTreeSet<usize> = BTreeSet::<usize>::new();

                            for index in edge.iter().copied() {
                                if !rendered_volume_set.contains_key(&index) {
                                    rendered_volume_set.insert(
                                        index,
                                        (Cell::new(BTreeSet::new()), Cell::new(0))
                                    );
                                }

                                for face in &self.builder.volumes[index].faces {
                                    'portal_rendering: for portal in &face.portal_polygons {
                                        let portal_polygon = self.builder.portal_polygons
                                            .get(portal.polygon_set_index)
                                            .unwrap();

                                        // Perform standard backface culling
                                        let backface_cull_result =
                                            (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                            >= 0.0
                                        ;
                                        if backface_cull_result != portal.is_front {
                                            continue 'portal_rendering;
                                        }

                                        // Perform visibility check
                                        if self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back {
                                            continue 'portal_rendering;
                                        }

                                        if rendered_volume_set.contains_key(&portal.dst_volume_index) {
                                            continue 'portal_rendering;
                                        }

                                        new_edge.insert(portal.dst_volume_index);
                                    }
                                }
                            }

                            edge = new_edge;
                        }

                        // Calculate entry parent sets
                        for (volume_index, _) in &rendered_volume_set {
                            let volume = &self.builder.volumes[*volume_index];

                            let portal_iter = volume
                                .faces
                                .iter()
                                .flat_map(|face| face.portal_polygons.iter());

                            'portal_rendering: for portal in portal_iter {
                                let portal_polygon = self.builder.portal_polygons
                                    .get(portal.polygon_set_index)
                                    .unwrap();

                                // Perform standard backface culling
                                let backface_cull_result =
                                    (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                    >= 0.0
                                ;
                                if backface_cull_result != portal.is_front {
                                    continue 'portal_rendering;
                                }

                                // Perform visibility check
                                if self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back {
                                    continue 'portal_rendering;
                                }

                                let dst_data = rendered_volume_set.get(&portal.dst_volume_index).unwrap();

                                let mut parent_set = dst_data.0.replace(BTreeSet::new());
                                parent_set.insert(*volume_index);
                                _ = dst_data.0.replace(parent_set);
                            }
                        }

                        // Display parent sets
                        for (volume, (parent_cell, _)) in &rendered_volume_set {
                            let parents = parent_cell.replace(BTreeSet::new());

                            println!("{} {:?}", *volume, parents);

                            _ = parent_cell.replace(parents);
                        }

                        let mut finalized_volumes = Vec::<(usize, u32)>::new();

                        // Current resolution process edge
                        let mut edge = BTreeSet::<usize>::new();

                        edge.insert(start_volume_index);

                        while !edge.is_empty() {
                            // resolve children of all volumes
                            let mut new_edge = BTreeSet::<usize>::new();

                            // Resolve volume
                            for edge_volume_index in edge {
                                // Edge content reference
                                let edge_data = rendered_volume_set.get(&edge_volume_index).unwrap();

                                let volume = &self.builder.volumes[edge_volume_index];

                                let portal_iter = volume
                                    .faces
                                    .iter()
                                    .flat_map(|face| face.portal_polygons.iter());

                                'portal_rendering: for portal in portal_iter {
                                    let portal_polygon = self.builder.portal_polygons
                                        .get(portal.polygon_set_index)
                                        .unwrap();

                                    // Perform standard backface culling
                                    let backface_cull_result =
                                        (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                        >= 0.0
                                    ;
                                    if backface_cull_result != portal.is_front {
                                        continue 'portal_rendering;
                                    }

                                    // Perform visibility check
                                    if self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back {
                                        continue 'portal_rendering;
                                    }

                                    // try to resolve child
                                    let dst_data = rendered_volume_set.get(&portal.dst_volume_index).unwrap();

                                    let mut parent_set = dst_data.0.replace(BTreeSet::new());

                                    parent_set.remove(&edge_volume_index);

                                    dst_data.1.set(dst_data.1.get() + edge_data.1.get() + 1);

                                    if parent_set.is_empty() {
                                        new_edge.insert(portal.dst_volume_index);
                                        finalized_volumes.push((edge_volume_index, 0));
                                    }

                                    _ = dst_data.0.replace(parent_set);
                                }

                                // push (edge, weight pair)
                                // finalized_volumes.push((edge_volume_index, edge_data.1.get()));
                            }

                            edge = new_edge;
                        }

                        {
                            let finalized_set = finalized_volumes
                                .iter()
                                .map(|(i, _)| *i)
                                .collect::<BTreeSet<_>>();
                            if finalized_set.len() != finalized_volumes.len() {
                                panic!("Length miss!");
                            }
                        }
                            
                        // finalized_volumes
                        //     .sort_by(|a, b| std::cmp::Ord::cmp(&b.1, &a.1));

                        for (index, weight) in &finalized_volumes {
                            println!("{index} {weight}");
                        }
                    
                        }

                        let mut rendered_volume_vec = Vec::<usize>::new();

                        let mut edge = BTreeSet::new();

                        edge.insert(start_volume_index);

                        let mut insertion_counter = 0u32;

                        while !edge.is_empty() {
                            println!("edge: {:?}", edge);

                            let mut new_edge = BTreeSet::new();

                            for volume_index in edge {
                                rendered_volume_vec.push(volume_index);

                                let volume = self.builder.volumes.get(volume_index).unwrap();

                                let portal_iter = volume
                                    .faces
                                    .iter()
                                    .flat_map(|face| face.portal_polygons.iter())
                                ;

                                'portal_rendering: for portal in portal_iter {
                                    let portal_polygon = self.builder.portal_polygons
                                        .get(portal.polygon_set_index)
                                        .unwrap();

                                    // Perform standard backface culling
                                    let backface_cull_result =
                                        (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                        >= 0.0
                                    ;
                                    if backface_cull_result != portal.is_front {
                                        continue 'portal_rendering;
                                    }

                                    // Perform visibility check
                                    if self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back {
                                        continue 'portal_rendering;
                                    }

                                    new_edge.insert(portal.dst_volume_index);
                                    
                                }
                            }

                            edge = new_edge;
                        }

                        print!("{:?} -> ", rendered_volume_vec);

                        let mut added_set = BTreeSet::<usize>::new();
                        rendered_volume_vec = rendered_volume_vec
                            .into_iter()
                            .rev()
                            .filter(|v| added_set.insert(*v))
                            .collect();
                        println!("{:?}", rendered_volume_vec);


                        // let mut finalized_volumes = rendered_volume_set.iter()
                        //     .map(|(i, w)| (*i, *w))
                        //     .collect::<Vec<_>>();
                        // finalized_volumes
                        //     .sort_by(|(_, lhs), (_, rhs)|
                        //         std::cmp::Ord::cmp(rhs, lhs)
                        //     );
                        // println!("rlist: {:?}", finalized_volumes);

                        // let render_list = rendered_volume_set
                        //     .iter()
                        //     .map(|(i, _)| *i)
                        //     .collect::<Vec<_>>();

                        for index in rendered_volume_vec {
                            let volume = &self.builder.volumes[index];

                            for face in &volume.faces {
                                'polygon_rendering: for physical_polygon in &face.physical_polygons {
                                    let polygon = &physical_polygon.polygon;

                                    if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                                        continue 'polygon_rendering;
                                    }
    
                                    let polygon = &physical_polygon.polygon;
    
                                    let color = [
                                        ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8,
                                    ];
                                    // let color = physical_polygon.color.to_le_bytes();
    
                                    unsafe {
                                        glu_sys::glColor3ub(color[0], color[1], color[2]);
                
                                        glu_sys::glBegin(glu_sys::GL_POLYGON);
                                        for point in &polygon.points {
                                            glu_sys::glVertex3f(point.x, point.y, point.z);
                                        }
                                        glu_sys::glEnd();
                                    }
                                }
                            }
                        }
                    }

                    pub fn render_volume(&mut self, volume: &map::builder::HullVolume) {
                        // if self.depth >= 32 {
                        //     eprintln!("Rendering depth exceeded 32!");
                        //     return;
                        // }

                        // if self.depth >= 4 {
                        //     return;
                        // }

                        self.depth += 1;

                        for face in &volume.faces {
                            'portal_rendering: for portal in &face.portal_polygons {
                                let portal_polygon = self.builder.portal_polygons
                                    .get(portal.polygon_set_index)
                                    .unwrap();

                                // Perform standard backface culling
                                let backface_cull_result =
                                    (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                    >= 0.0
                                ;
                                if backface_cull_result != portal.is_front {
                                    continue 'portal_rendering;
                                }

                                // Perform visibility check
                                if self.camera_plane.get_polygon_relation(&portal_polygon) == geom::PolygonRelation::Back {
                                    continue 'portal_rendering;
                                }

                                // Check repeats
                                if self.render_set.contains(&portal.dst_volume_index) {
                                    self.set_miss_count += 1;
                                    continue 'portal_rendering;
                                }
                                _ = self.render_set.insert(portal.dst_volume_index);

                                let dst_volume = self.builder.volumes.get(portal.dst_volume_index).unwrap();

                                self.render_volume(dst_volume);
                            }
                        }

                        for face in &volume.faces {
                            'polygon_rendering: for physical_polygon in &face.physical_polygons {
                                let polygon = &physical_polygon.polygon;

                                if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                                    continue 'polygon_rendering;
                                }

                                let polygon = &physical_polygon.polygon;

                                let color = [
                                    ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8,
                                ];
                                // let color = physical_polygon.color.to_le_bytes();

                                unsafe {
                                    glu_sys::glColor3ub(color[0], color[1], color[2]);
            
                                    glu_sys::glBegin(glu_sys::GL_POLYGON);
                                    for point in &polygon.points {
                                        glu_sys::glVertex3f(point.x, point.y, point.z);
                                    }
                                    glu_sys::glEnd();
                                }
                            }
                        }

                        self.depth -= 1;
                    }
                }

                let mut render_context = RenderContext {
                    camera_location: logical_camera.location,
                    camera_plane: geom::Plane::from_point_normal(
                        logical_camera.location,
                        logical_camera.direction
                    ),
                    builder: &builder,
                    depth: 0,

                    render_set: {
                        let mut set = BTreeSet::new();
                        set.insert(start_volume_index);
                        set
                    },
                    set_miss_count: 0,
                };

                // render_context.render_volume(start_volume);
                render_context.render(start_volume_index);

                println!("Render set miss count: {}", render_context.set_miss_count);
            }
            
            // Render volumes by VBSP
            // #[cfg(not)]
            {
                fn traverse_vbsp(camera_location: Vec3f, bsp: &map::builder::VolumeBsp, volume_order: &mut Vec<usize>) {
                    match bsp {
                        map::builder::VolumeBsp::Node {
                            plane,
                            front,
                            back
                        } => {
                            let (first, second) = match plane.get_point_relation(camera_location) {
                                geom::PointRelation::Front | geom::PointRelation::OnPlane => (front, back),
                                geom::PointRelation::Back => (back, front)
                            };

                            if let Some(first) = first.as_ref() {
                                traverse_vbsp(camera_location, first, volume_order);
                            }

                            if let Some(second) = second.as_ref() {
                                traverse_vbsp(camera_location, second, volume_order);
                            }
                        }
                        map::builder::VolumeBsp::Leaf(index) => {
                            volume_order.push(*index);
                        }
                    }
                }

                let mut volume_order = Vec::<usize>::new();

                traverse_vbsp(
                    logical_camera.location,
                    &builder.volume_bsp.as_ref().unwrap(),
                    &mut volume_order
                );

                volume_order.reverse();

                for volume_index in volume_order {
                    let volume = &builder.volumes[volume_index];

                    let physical_polygon_iter = volume
                        .faces
                        .iter()
                        .flat_map(|face| face.physical_polygons.iter());

                    'physical_polygon_loop: for physical_polygon in physical_polygon_iter {
                        let polygon = &physical_polygon.polygon;

                        if (polygon.plane.normal ^ logical_camera.location) - polygon.plane.distance <= 0.0 {
                            continue 'physical_polygon_loop;
                        }

                        let polygon = &physical_polygon.polygon;

                        let color = [
                            ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                            ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                            ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                        ];

                        glu_sys::glColor3ub(color[0], color[1], color[2]);

                        glu_sys::glBegin(glu_sys::GL_POLYGON);
                        for point in &polygon.points {
                            glu_sys::glVertex3f(point.x, point.y, point.z);
                        }
                        glu_sys::glEnd();
                    }
                }
            }

            // Render pre-calculated portal graph
            #[cfg(not)]
            {
                glu_sys::glColor3f(1.0, 0.0, 0.0);
                glu_sys::glLineWidth(3.0);


                for (first_index, second_index) in &portal_graph.edges {
                    let first = portal_graph.vertices[*first_index as usize];
                    let second = portal_graph.vertices[*second_index as usize];

                    glu_sys::glBegin(glu_sys::GL_LINE_STRIP);
                    glu_sys::glVertex3f(first.x, first.y, first.z);
                    glu_sys::glVertex3f(second.x, second.y, second.z);
                    glu_sys::glEnd();
                }
    

                glu_sys::glLineWidth(1.0);
            }

            // Render portal graph
            #[cfg(not)]
            {
                let get_volume_center = |volume_index: usize| -> Vec3f {
                    let volume = &builder.volumes[volume_index];

                    let mut collector = Vec3f::zero();

                    for face in &volume.faces {
                        let mut face_collector = Vec3f::zero();

                        for point in &face.polygon.points {
                            face_collector += *point;
                        }

                        collector += face_collector / face.polygon.points.len() as f32;
                    }

                    collector / volume.faces.len() as f32
                };

                let mut polygon_ids = BTreeSet::<usize>::from_iter(0..builder.volumes.len());

                glu_sys::glColor3f(1.0, 0.0, 0.0);
                glu_sys::glLineWidth(3.0);

                while let Some(first_id) = polygon_ids.first().copied() {
                    let mut visited_volume_ids = BTreeSet::<usize>::new();
                    let mut edge_polygons = vec![
                        (first_id, get_volume_center(first_id))
                    ];

                    while !edge_polygons.is_empty() {
                        let mut new_edge_polygons = Vec::new();
    
                        for (volume_index, volume_center) in edge_polygons {
                            let volume = &builder.volumes[volume_index];
    
                            for face in &volume.faces {
                                'portal_loop: for portal in &face.portal_polygons {
                                    let dst_index = portal.dst_volume_index;
    
                                    if visited_volume_ids.contains(&dst_index) {
                                        continue 'portal_loop;
                                    }
    
                                    let center = get_volume_center(dst_index);
    
                                    glu_sys::glBegin(glu_sys::GL_LINE_STRIP);
                                    glu_sys::glVertex3f(volume_center.x, volume_center.y, volume_center.z);
                                    glu_sys::glVertex3f(center.x, center.y, center.z);
                                    glu_sys::glEnd();
    
                                    new_edge_polygons.push((dst_index, center));
                                }
                            }
    
                            visited_volume_ids.insert(volume_index);
                        }
    
                        edge_polygons = new_edge_polygons;
                    }

                    for v in visited_volume_ids {
                        polygon_ids.remove(&v);
                    }
                }

                glu_sys::glLineWidth(1.0);
            }

            // Render hull volumes
            #[cfg(not)]
            {
                let render_hull_volume = |hull_volume: &map::builder::HullVolume, render_hull: bool, render_physical: bool| {
                    for hull_polygon in &hull_volume.faces {
                        let polygon = &hull_polygon.polygon;

                        // backface culling
                        // if (polygon.plane.normal ^ camera_location) - polygon.plane.distance <= 0.0 {
                        //     continue 'face_polygon_loop;
                        // }

                        
                        if render_hull {
                            let color = [
                                ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                            ];

                            glu_sys::glColor3ub(color[0], color[1], color[2]);
                            
                            glu_sys::glBegin(glu_sys::GL_LINE_LOOP);
                            for point in &polygon.points {
                                glu_sys::glVertex3f(point.x, point.y, point.z);
                            }
                            glu_sys::glEnd();
                        }

                        if render_physical {
                            for physical_polygon in &hull_polygon.physical_polygons {
                                let polygon = &physical_polygon.polygon;
    
                                let color = [
                                    ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                                ];
        
                                glu_sys::glColor3ub(color[0], color[1], color[2]);
        
                                glu_sys::glBegin(glu_sys::GL_POLYGON);
                                for point in &polygon.points {
                                    glu_sys::glVertex3f(point.x, point.y, point.z);
                                }
                                glu_sys::glEnd();
                            }
                        }
                    }
                };

                static mut volume_index: u32 = 0;
                if input.is_key_clicked(Scancode::Space) {
                    volume_index += 1;
                }

                println!("Current volume index: {}", volume_index as usize % builder.volumes.len());

                render_hull_volume(
                    &builder.volumes[volume_index as usize % builder.volumes.len()],
                    false,
                    true
                );

                for hull_volume in &builder.volumes {
                    render_hull_volume(hull_volume, true, false);
                }
            }

            // Render BSP
            #[cfg(not)]
            {
                struct VisitContext {
                    location: Vec3f,
                    // debug_plane: Plane,
                }
                impl VisitContext {
                    pub unsafe fn visit(&mut self, node: &bsp::Bsp) {
                        match node {
                            bsp::Bsp::Element { splitter, front, back } => {
                                match splitter.get_point_relation(self.location) {
                                    PointRelation::Front => {
                                        self.visit(back);
                                        self.visit(front);
                                    }
                                    _ => {
                                        self.visit(front);
                                        self.visit(back);
                                    }
                                }
                            }
                            bsp::Bsp::Leaf { polygons } => {
                                
                                'polygon_loop: for polygon in polygons {
                                    // backface culling
                                    if (polygon.plane.normal ^ self.location) - polygon.plane.distance <= 0.0 {
                                        continue 'polygon_loop;
                                    }
                                    
                                    let color = [
                                        ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                        ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                                    ];
            
                                    glu_sys::glColor3ub(color[0], color[1], color[2]);
            
                                    glu_sys::glBegin(glu_sys::GL_POLYGON);
                                    for point in &polygon.points {
                                        glu_sys::glVertex3f(point.x, point.y, point.z);
                                    }
                                    glu_sys::glEnd();
    
                                    // draw intersection line
                                    // let intr = self.debug_plane.intersect_polygon(polygon);
                                    // if let Some((begin, end)) = intr {
                                    //     glu_sys::glColor3ub(0xFF, 0x00, 0x00);
                                    //     glu_sys::glBegin(glu_sys::GL_LINE_STRIP);
                                    //     glu_sys::glVertex3f(begin.x, begin.y, begin.z);
                                    //     glu_sys::glVertex3f(end.x, end.y, end.z);
                                    //     glu_sys::glEnd();
                                    // }
    
                                }
                            }
                        }
                    }
                }
            
                let mut vcontext = VisitContext {
                    location: logical_camera.location,
                    // debug_plane: Plane::from_point_normal(
                    //     Vec3f::new(-200.0, 2000.0, -50.0),
                    //     Vec3f::new(1.0, 2.0, 0.0)
                    // )
                };
    
                vcontext.visit(&bsp);
            }
                    // render_brushes(&brushes, camera.location);
        }

        window.gl_swap_window();
    }
}
