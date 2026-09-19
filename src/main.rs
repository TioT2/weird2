//! Hardware-acceleration-free 3D engine

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Render and Physics BSP's building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with BSP's.

use std::{io::{Read, Write}, sync::{Arc, mpsc}};
use zerocopy::IntoBytes;

use crate::{
    frame_slice::FrameSliceMut,
    math::{Mat4f, Vec2f, Vec3f}
};

pub mod math;
pub mod system_font;
pub mod frame_slice;
pub mod rand;
pub mod geom;
pub mod bsp;
pub mod map;
pub mod res;
pub mod camera;
pub mod timer;
pub mod input;
pub mod flags;
pub mod render;

/// Make u64 from [u16; 4]
pub const fn u64_from_u16(value: [u16; 4]) -> u64 {
    ( value[0] as u64       ) |
    ((value[1] as u64) << 16) |
    ((value[2] as u64) << 32) |
    ((value[3] as u64) << 48)
}

/// Convert u64 into [u16; 4]
pub const fn u64_into_u16(value: u64) -> [u16; 4] {
    [
        ((value      ) & 0xFFFF) as u16,
        ((value >> 16) & 0xFFFF) as u16,
        ((value >> 32) & 0xFFFF) as u16,
        ((value >> 48) & 0xFFFF) as u16,
    ]
}

/// Input to render about world change
pub enum RenderInputMessage {
    /// Request for frame rendering
    NewFrame {
        /// Frame target buffer
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
        shadow_camera: Option<camera::Camera>,
        camera: camera::Camera,
        projection_matrix: Mat4f,
        rasterization_mode: render::RasterizationMode,
    }
}

/// Render output message
pub enum RenderOutputMessage {
    /// Response containing rendered frame
    RenderedFrame {
        frame_buffer: Vec<u64>,
        width: u32,
        height: u32,
        stride: u32,
    }
}

/// Present frame from slice to window surface
fn present_frame(
    mut frame: FrameSliceMut<'_, u32>,
    window_surface: &mut sdl2::video::WindowSurfaceRef,
) {
    let width = frame.width() as u32;
    let height = frame.height() as u32;
    let stride = frame.stride() as u32 * 4;
    let surface_bytes = frame.as_flat().unwrap().as_mut_bytes();

    let mut render_surface = match sdl2::surface::Surface::from_data(
        surface_bytes,
        width,
        height,
        stride,
        sdl2::pixels::PixelFormatEnum::ABGR8888
    ) {
        Ok(surface) => surface,
        Err(err) => {
            eprintln!("Source surface create error: {}", err);
            return;
        }
    };

    // Disable alpha blending
    if let Err(err) = render_surface.set_blend_mode(sdl2::render::BlendMode::None) {
        eprintln!("Cannot disable render surface blending: {}", err);
    };

    // Perform render surface blit
    let (window_w, window_h) = window_surface.size();
    let src_rect = sdl2::rect::Rect::new(0, 0, width, height);
    let dst_rect = sdl2::rect::Rect::new(0, 0, window_w, window_h);

    if let Err(err) = render_surface.blit_scaled(src_rect, window_surface, dst_rect) {
        eprintln!("Surface blit failed: {}", err);
    }

    if let Err(err) = window_surface.update_window() {
        eprintln!("Window update failed: {}", err);
    }
}

/// Wrap function call with time calculation
fn with_time_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = std::time::Instant::now();
    let value = f();
    let end = std::time::Instant::now();

    (value, end.duration_since(start).as_nanos() as f64 / 1_000_000f64)
}

/// Initialize rendering thread
fn init_render_thread(
    map: Arc<bsp::Map>,
    material_table: Arc<res::MaterialTable>
) -> (mpsc::Sender<RenderInputMessage>, mpsc::Receiver<RenderOutputMessage>) {
    let (render_in_sender, render_in_reciever) = mpsc::channel::<RenderInputMessage>();
    let (render_out_sender, render_out_reciever) = mpsc::channel::<RenderOutputMessage>();

    // Spawn render thread
    _ = std::thread::spawn(move || {
        // Create new local timer
        let mut timer = timer::Timer::default();

        let msg_reciever = render_in_reciever;
        let msg_sender = render_out_sender;

        // Map reference
        let map_arc = map.clone();
        let material_table_arc = material_table;

        let map = map_arc.as_ref();
        let material_table = material_table_arc.as_ref();

        let material_reference_table = material_table
            .build_reference_table(map);

        // Send empty frame to match sending order
        _ = msg_sender.send(RenderOutputMessage::RenderedFrame {
            frame_buffer: Vec::new(),
            width: 0,
            height: 0,
            stride: 0
        });

        'frame_render_loop: loop {
            let Ok(message) = msg_reciever.recv() else {
                break 'frame_render_loop;
            };

            match message {
                RenderInputMessage::NewFrame {
                    mut frame_buffer,
                    width,
                    height,
                    shadow_camera,
                    camera,
                    projection_matrix,
                    rasterization_mode
                } => {
                    timer.response();

                    let time = timer.get_time();

                    // Clear framebuffer
                    frame_buffer.fill(0);

                    // Very long function call, actually
                    let mut render_context = render::RenderContext {
                        camera: render::RenderCamera {
                            view_projection: camera.view() * projection_matrix,
                            location: camera.location(),
                            half_fw: width as f32 * 0.5,
                            half_fh: height as f32 * 0.5,
                        },

                        shadow_camera: shadow_camera.map(|shadow_camera| render::RenderCamera {
                            view_projection: shadow_camera.view() * projection_matrix,
                            location: shadow_camera.location(),
                            half_fw: width as f32 * 0.5,
                            half_fh: height as f32 * 0.5,
                        }),

                        // Construct target frame slice
                        frame: FrameSliceMut::<u64>::new(
                            width as usize,
                            height as usize,
                            width as usize,
                            frame_buffer.as_mut_slice()
                        ),

                        map,
                        material_table: &material_reference_table,
                        rasterization_mode,

                        sky_background_uv_offset: Vec2f::broadcast(time * -12.0),
                        sky_uv_offset: Vec2f::broadcast(time * 16.0),
                    };

                    render_context.render();

                    // Send output message
                    _ = msg_sender.send(RenderOutputMessage::RenderedFrame {
                        frame_buffer,
                        width,
                        height,
                        stride: width,
                    });
                }
            }
        }
    });

    (render_in_sender, render_out_reciever)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable/disable map caching
    let do_enable_map_caching = true;

    // Synchronize visible-set-building and projection cameras
    let mut shadow_camera: Option<camera::Camera> = None;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut rasterization_mode = render::RasterizationMode::Full;

    let data_path = ".local/";
    let (map_name, map_src_format, wad_name) =
        // ("d1_trainstation_01", "vmf", "base");
        ("quake/e1m1", "map", "quake/gfx/base.wad");
        // ("quake/e1m5", "map", "quake/gfx/medieval.wad");

    // Load map
    let map = {
        // yay, this code will not work on non-local builds)))
        // --

        let wbsp_path = format!("{}wbsp/{}.wbsp", data_path, map_name);
        let map_src_path = format!("{}{}.{}", data_path, map_name, map_src_format);

        /// Load map source from local storage and compile it
        fn load_and_compile(path: &str, src_format: &str) -> bsp::Map {
            let source = std::fs::read_to_string(path).unwrap();

            let map = match src_format {
                "map" => map::q1_map::Map::parse(&source).unwrap().build_wmap(),
                "vmf" => map::source_vmf::Entry::parse_vmf(&source).unwrap().build_wmap().unwrap(),
                _ => panic!("'{}' map format is not implemented", src_format)
            };

            // Build BSP
            let build_start = std::time::Instant::now();
            let mut bsp = bsp::compiler::compile(&map).unwrap();
            println!("BSP compilation time: {}", build_start.elapsed().as_secs_f32());

            // Bake lightmaps
            let bake_start = std::time::Instant::now();
            bsp::lightmap_baker::bake(&mut bsp, &map);
            println!("Lightmap baking time: {}", bake_start.elapsed().as_secs_f32());

            bsp
        }

        if do_enable_map_caching {
            match std::fs::File::open(&wbsp_path) {
                Ok(mut bsp_file) => {
                    // Load map from map cache
                    let mut bsp_file_data = Vec::new();
                    bsp_file.read_to_end(&mut bsp_file_data).unwrap();
                    bsp::wbsp::load(&bsp_file_data).unwrap()
                }
                Err(_) => {
                    // Compile map
                    let map = load_and_compile(&map_src_path, map_src_format);

                    if let Some(directory) = std::path::Path::new(&wbsp_path).parent() {
                        _ = std::fs::create_dir(directory);
                    }

                    // Save map to map cache
                    if let Ok(mut file) = std::fs::File::create(&wbsp_path) {
                        bsp::wbsp::save(&map, &mut file).unwrap()
                    }

                    map
                }
            }
        } else {
            load_and_compile(&map_src_path, map_src_format)
        }
    };

    let map = Arc::new(map);

    // Display BSP statistics
    {
        pub struct BspStat {
            pub nodes: u64,
            pub leafs: u64,
            pub leaf_depth_sum: u64,
            pub depth_max: u64,
            pub total_disbalance: u64,
        }

        let stat = map.get_world_model().get_bsp().fold_ref(
            |_| BspStat {
                nodes: 1,
                leafs: 1,
                leaf_depth_sum: 1,
                depth_max: 1,
                total_disbalance: 0,
            },
            |fstat, bstat| BspStat {
                nodes: fstat.nodes + bstat.nodes + 1,
                leafs: fstat.leafs + bstat.leafs,
                leaf_depth_sum: fstat.leaf_depth_sum + bstat.leaf_depth_sum + fstat.leafs + bstat.leafs,
                depth_max: u64::max(fstat.depth_max, bstat.depth_max) + 1,
                total_disbalance: fstat.total_disbalance + bstat.total_disbalance + u64::abs_diff(fstat.nodes, bstat.nodes)
            }
        );

        println!("nodes           : {}", stat.nodes);
        println!("leafs           : {}", stat.leafs);
        println!("avg. leaf depth : {}", stat.leaf_depth_sum as f64 / stat.leafs as f64);
        println!("depth           : {}", stat.depth_max);
        println!("disbalance      : {}", stat.total_disbalance);
        println!("avg. disbalance : {}", stat.total_disbalance as f64 / (stat.nodes - stat.leafs) as f64);
    }

    let material_table = {
        let mut wp = std::path::PathBuf::new();
        wp.push(data_path);
        wp.push(wad_name);

        res::MaterialTable::load_wad2(&std::fs::read(&wp).unwrap()).unwrap()
    };
    let material_table = Arc::new(material_table);

    // Setup window
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    let window = video
        .window("WEIRD-2", 1280, 720)
        .position_centered()
        .resizable()
        .build()
        .unwrap();

    // Setup systems
    let mut timer = timer::Timer::default();
    let mut input = input::Input::default();

    // camera.location = Vec3f::new(-174.0, 2114.6, -64.5); // -200, 2000, -50
    // camera.direction = Vec3f::new(-0.4, 0.9, 0.1);

    // heavy scene
    let mut camera = camera::Camera::new(Vec3f::new(1402.4, 1913.7, -86.3), Vec3f::new(-0.74, 0.63, -0.24));

    // camera.location = Vec3f::new(-543.3503, 1378.1802, 434.5833);
    // camera.direction = Vec3f::new(-0.004, 0.935, -0.354).normalized();

    // camera.direction = Vec3f::new(0.055, -0.946, 0.320); // (-0.048328593, -0.946524262, 0.318992347)

    // // sky
    // camera.location = Vec3f::new(-72.9, 698.3, -118.8);
    // camera.direction = Vec3f::new(0.37, 0.68, 0.63);

    // camera.location = Vec3f::new(30.0, 40.0, 50.0);

    // Camera used for visible set building

    // Buffer that contains rendered pixels
    let mut hdr_frame_buffer = Vec::<u64>::new();

    // LDR framebuffer
    let mut ldr_frame_buffer = Vec::<u32>::new();

    // Render thread IO channels
    let (render_in, render_out) = init_render_thread(map.clone(), material_table.clone());

    'main_loop: loop {
        input.release_changed();

        while let Some(event) = event_pump.poll_event() {
            match event {
                sdl2::event::Event::Quit { .. } => {
                    break 'main_loop;
                }
                sdl2::event::Event::KeyUp { scancode: Some(code), .. } => {
                    input.on_state_changed(code, false);
                }
                sdl2::event::Event::KeyDown { scancode: Some(code), .. } => {
                    input.on_state_changed(code, true);
                }
                _ => {}
            }
        }

        timer.response();
        camera.response(&timer, &input);

        // Toggle shadow camera
        if input.is_key_clicked(input::Key::Num9) {
            shadow_camera = if shadow_camera.is_none() {
                Some(camera)
            } else {
                None
            };
        }

        if input.is_key_clicked(input::Key::Num0) {
            rasterization_mode = rasterization_mode.next();
        }

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        /// Rendering resolution scale
        const FRAME_SCALE: usize = 2;

        let (frame_width, frame_height) = (
            window_width / FRAME_SCALE,
            window_height / FRAME_SCALE,
        );

        // Calculate aspect ratio
        let (aspect_x, aspect_y) = if window_width > window_height {
            (window_width as f32 / window_height as f32, 1.0)
        } else {
            (1.0, window_height as f32 / window_width as f32)
        };

        // Build projection matrix
        let projection_matrix = Mat4f::projection_frustum_inf_far(
            -0.5 * aspect_x, 0.5 * aspect_x,
            -0.5 * aspect_y, 0.5 * aspect_y,
            2.0 / 3.0
        );

        // Resize frame buffer to fit window's size
        hdr_frame_buffer.resize(frame_width * frame_height, 0);

        let send_res = render_in.send(RenderInputMessage::NewFrame {
            frame_buffer: hdr_frame_buffer,
            width: frame_width as u32,
            height: frame_height as u32,
            shadow_camera,
            camera,
            projection_matrix,
            rasterization_mode,
        });

        if let Err(e) = send_res {
            println!("Sending error: {}", e);
            break 'main_loop;
        }

        let Ok(render_result) = render_out.recv() else {
            eprintln!("Render thread dropped");
            break 'main_loop;
        };

        // Previous frame contents
        let prev_hdr_frame_buffer = match render_result {
            RenderOutputMessage::RenderedFrame {
                frame_buffer: rendered_hdr_buffer,
                width,
                height,
                stride
            } => {
                // Resize ldr buffer to match hdr buffer's size
                ldr_frame_buffer.resize(stride as usize * height as usize, 0);

                // Map from hdr to ldr
                let tm_time = with_time_ms(
                    || render::hdr_to_ldr(&rendered_hdr_buffer, &mut ldr_frame_buffer, true)
                ).1;

                let mut ldr_frame = FrameSliceMut::new(width as usize, height as usize, stride as usize, &mut ldr_frame_buffer);

                // Display frame statistics
                let mut fw = system_font::writer(ldr_frame.reborrow_mut());
                writeln!(fw)?;
                writeln!(fw, " FPS: {} ({}ms)", timer.get_fps(), 1000.0 / timer.get_fps())?;
                writeln!(fw, " SC={}, RM={}", shadow_camera.is_some() as u32, rasterization_mode as u32)?;
                writeln!(fw, " TM: {}ms", tm_time)?;
                writeln!(fw, " RES: {}x{}", width, height)?;

                // Present rendered frame
                match window.surface(&event_pump) {
                    Ok(mut window_surface) => present_frame(ldr_frame.reborrow_mut(), &mut window_surface),
                    Err(err) => eprintln!("Cannot get window surface: {}", err),
                };

                /* Set previous buffer memory */
                Some(rendered_hdr_buffer)
            }
        };

        hdr_frame_buffer = prev_hdr_frame_buffer.unwrap_or(Vec::new());
    }

    Ok(())
}
