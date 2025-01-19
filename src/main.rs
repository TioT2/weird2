
use std::collections::{BTreeSet, HashMap};

use brush::Brush;
use math::Vec3f;
use sdl2::{event::Event, keyboard::Scancode};

/// Basic math utility
#[macro_use]
pub mod math;

/// Random number generator
pub mod rand;

/// Binary space partition implementation
pub mod bsp;

/// Brush builder
pub mod brush;

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
}

impl Timer {
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            now,
            dt: std::time::Duration::from_millis(60),
        }
    }

    pub fn response(&mut self) {
        let new_now = std::time::Instant::now();
        self.dt = new_now.duration_since(self.now);
        self.now = new_now;
    }

    pub fn get_delta_time(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    pub fn get_time(&self) -> f32 {
        self.now
            .duration_since(self.start)
            .as_secs_f32()
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

pub unsafe fn render_brushes(brushes: &[Brush], camera_location: Vec3f) {
    for brush in brushes {
        'polygon_loop: for polygon in &brush.polygons {
            let point_normal = Vec3f::cross(
                (polygon.points[2] - polygon.points[1]).normalized(),
                (polygon.points[0] - polygon.points[1]).normalized(),
            ).normalized();
            // point_normal = point_normal * (point_normal ^ polygon.plane.normal).signum();

            if (point_normal ^ camera_location) - polygon.plane.distance <= 0.0 {
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
        }
    }
}

fn main() {
    print!("\n\n\n\n\n\n\n\n");

    // yay, this code will not compile on non-local builds)))
    // bit of functional rust code)
    // --
    let brushes = map::builder::Map::parse(include_str!("../temp/e1m1.map"))
        .expect("Map parsing error occured!")
        .find_entity("classname", Some("worldspawn"))
        .expect("No worldspawn entity in map!")
        .brushes
        .iter()
        .map(|brush| brush
            .faces
            .iter()
            .map(|face| geom::Plane::from_points(
                face.p1,
                face.p0,
                face.p2
            ))
            .collect::<Vec<_>>()
        )
        .filter_map(|brush_set| Brush::from_planes(brush_set.as_slice()))
        .collect::<Vec<_>>();

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

    camera.location = Vec3f::new(-200.0, 2000.0, -50.0);
    // camera.location = Vec3f::new(-10.0, -30.0, 150.0);
    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // let bsp = {
    //     let polygons = brush::get_map_polygons(&brushes, false);
    //     bsp::Bsp::build(&polygons)
    // };

    let mut builder = map::builder::Builder::new();
    builder.start_build_volumes(&brushes);

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
    builder.start_remove_invisible();

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

    let mut logical_camera = camera;

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

        if do_sync_logical_camera {
            logical_camera = camera;
        }

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

            // Render by BSP
            // #[cfg(not)]
            'render: {
                // println!("Rendering started!");

                let volume_id_opt = builder.volume_bsp.as_ref().unwrap().traverse(logical_camera.location);

                let Some(volume_index) = volume_id_opt else {
                    break 'render;
                };

                let start_volume = builder.volumes.get(volume_index).unwrap();

                struct RenderContext<'t> {
                    camera_location: Vec3f,
                    camera_direction: Vec3f,
                    camera_plane: geom::Plane,
                    builder: &'t map::builder::Builder,
                    depth: usize,

                    render_set: BTreeSet<usize>,
                    set_miss_count: usize,
                }

                impl<'t> RenderContext<'t> {
                    pub fn render_volume(&mut self, volume: &map::builder::HullVolume) {
                        // if self.depth >= 32 {
                        //     eprintln!("Rendering depth exceeded 32!");
                        //     return;
                        // }

                        self.depth += 1;

                        for face in &volume.faces {
                            'portal_rendering: for portal in &face.portal_polygons {
                                let portal_polygon = self.builder.portal_polygons
                                    .get(portal.polygon_set_index)
                                    .unwrap();

                                // let normal_check = self.camera_direction ^ portal_polygon.plane.normal >= 0.0;
                                // if normal_check == portal.is_front {
                                //     continue 'portal_rendering;
                                // }

                                // perform check
                                let backface_cull_result =
                                    (portal_polygon.plane.normal ^ self.camera_location) - portal_polygon.plane.distance
                                    >= 0.0
                                ;
                                if backface_cull_result != portal.is_front {
                                    continue 'portal_rendering;
                                }

                                if self.render_set.contains(&portal.dst_volume_index) {
                                    // println!("Volume is already rendered!");
                                    self.set_miss_count += 1;
                                    continue 'portal_rendering;
                                }
                                _ = self.render_set.insert(portal.dst_volume_index);

                                let dst_volume = self.builder.volumes.get(portal.dst_volume_index).unwrap();

                                self.render_volume(dst_volume);
                            }

                            'polygon_rendering: for physical_polygon in &face.physical_polygons {
                                let polygon = &physical_polygon.polygon;

                                // if polygon.plane.normal ^ self.camera_direction >= 0.0 {
                                //     continue 'polygon_rendering;
                                // }

                                if (polygon.plane.normal ^ self.camera_location) - polygon.plane.distance <= 0.0 {
                                    continue 'polygon_rendering;
                                }

                                let polygon = &physical_polygon.polygon;

                                let color = [
                                    ((polygon.plane.normal.x + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.y + 1.0) / 2.0 * 255.0) as u8,
                                    ((polygon.plane.normal.z + 1.0) / 2.0 * 255.0) as u8
                                ];

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
                    camera_direction: logical_camera.direction,
                    camera_location: logical_camera.location,
                    camera_plane: geom::Plane::from_point_normal(
                        logical_camera.location,
                        logical_camera.direction
                    ),
                    builder: &builder,
                    depth: 0,

                    render_set: {
                        let mut set = BTreeSet::new();
                        set.insert(volume_index);
                        set
                    },
                    set_miss_count: 0,
                };

                render_context.render_volume(start_volume);

                println!("Render set miss count: {}", render_context.set_miss_count);
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
                render_hull_volume(
                    &builder.volumes[volume_index as usize % builder.volumes.len()],
                    false,
                    true
                );

                for hull_volume in &builder.volumes {
                    render_hull_volume(hull_volume, true, true);
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
