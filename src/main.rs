//! Hardware-acceleration-free 3D engine

// Resources:
// [WMAP -> WBSP], WRES -> WDAT/WRES
//
// WMAP - In-development map format, using during map editing
// WBSP - Intermediate format, used to exchange during different map compilation stages (e.g. Render and Physics BSP's building/Optimization/Lightmapping/etc.)
// WRES - Resource format, contains textures/sounds/models/etc.
// WDAT - Data format, contains 'final' project with BSP's.

use std::io::{Read, Write};
use zerocopy::IntoBytes;

use crate::{
    frame_slice::FrameSliceMut,
    math::{Mat4f, Vec2f, Vec3f}
};

// Engine modules
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Enable/disable map caching
    let do_enable_map_caching = true;

    // Synchronize visible-set-building and projection cameras
    let mut shadow_camera: Option<camera::Camera> = None;

    // Enable rendering with synchronization after some portion of frame pixels renderend
    let mut rasterization_mode = render::RasterizationMode::Full;

    let mut frame_scale = 1usize;

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

    let material_reference_table = material_table.build_reference_table(&map);

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
    let mut timer = timer::Timer::new();
    let mut input = input::Input::new();

    // Per-frame value averager
    let mut averager = timer::Averager::new();

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

    // Buffer containing rendered pixels
    let mut hdr_frame_buffer = Vec::<u64>::new();

    // Ldr frame buffer (used then rendering to window requires blitting)
    let mut ldr_frame_buffer = Vec::<u32>::new();

    'main_loop: loop {
        let _tpfa = averager.start_measure("spf");

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

        // Control frame scale
        if input.is_key_clicked(input::Key::Equals) {
            frame_scale += 1;
        }
        if input.is_key_clicked(input::Key::Minus) {
            frame_scale -= 1;
        }
        frame_scale = frame_scale.clamp(1, 8);

        // Acquire window extent
        let (window_width, window_height) = {
            let (w, h) = window.size();

            (w as usize, h as usize)
        };

        let (frame_width, frame_height) = (
            window_width / frame_scale,
            window_height / frame_scale,
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

        hdr_frame_buffer.resize(frame_width * frame_height, 0);
        // hdr_frame_buffer.fill(0x00FF_0000_00FF);
        hdr_frame_buffer.fill(0); // it's faster!

        // Resize frame buffer to fit window's size
        let mut hdr_frame = FrameSliceMut::new(
            frame_width,
            frame_height,
            frame_width,
            &mut hdr_frame_buffer
        );

        let render_measure = averager.start_measure("rendering");

        // Rendering context
        let mut render_context = render::Context {
            camera: render::Camera {
                view_projection: camera.view() * projection_matrix,
                location: camera.location(),
                half_fw: frame_width as f32 * 0.5,
                half_fh: frame_height as f32 * 0.5,
            },

            shadow_camera: shadow_camera.map(|shadow_camera| render::Camera {
                view_projection: shadow_camera.view() * projection_matrix,
                location: shadow_camera.location(),
                half_fw: frame_width as f32 * 0.5,
                half_fh: frame_height as f32 * 0.5,
            }),

            // Construct target frame slice
            frame: hdr_frame.reborrow_mut(),

            map: &map,
            material_table: &material_reference_table,
            rasterization_mode,

            sky_background_uv_offset: Vec2f::broadcast(timer.get_time() * -12.0),
            sky_uv_offset: Vec2f::broadcast(timer.get_time() * 16.0),
        };

        render_context.render();

        render_measure.finish();

        // Access LDR frame buffer
        let mut window_surface = match window.surface(&event_pump) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Failed to acquire window surface: {}", err);
                continue 'main_loop;
            }
        };

        // Window surface parameters
        let (ws_width, ws_height) = window_surface.size();
        let ws_pitch = window_surface.pitch();
        let ws_bpp = window_surface.pixel_format_enum().byte_size_per_pixel();
        let ws_pfe = window_surface.pixel_format_enum();

        // Pixel swap flag
        let do_swap = matches!(ws_pfe, sdl2::pixels::PixelFormatEnum::ARGB8888);

        // Present to certain ldr frame buffer
        let present_to = |direct: bool, mut ldr_fb: FrameSliceMut<u32>| -> Result<(), Box<dyn std::error::Error>> {
            averager.measure("tonemapping", || {
                for (src, dst) in hdr_frame.iter().zip(ldr_fb.iter_mut()) {
                    render::hdr_to_ldr(src, dst, true, direct && do_swap);
                }
            });

            let tm_time = averager.get("tonemapping").unwrap() as f32;
            let rnd_time = averager.get("rendering").unwrap() as f32;
            let spf = averager.get("spf").unwrap() as f32;

            let mut fw = system_font::writer(ldr_fb.reborrow_mut());
            writeln!(fw)?;
            writeln!(fw, " FPS: {} ({} ms)", 1.0 / spf, spf * 1000.0)?;
            writeln!(fw, " SC={}, RM={}, FS={}, DFB={}", shadow_camera.is_some() as u32, rasterization_mode as u32, frame_scale, direct as u32)?;
            writeln!(fw, " RND: {}ms, TM: {}ms", rnd_time * 1000.0, tm_time * 1000.0)?;
            writeln!(fw, " RES: {}x{} -> {}x{}", hdr_frame.width(), hdr_frame.height(), ws_width, ws_height)?;
            writeln!(fw, " PF: {:?}", ws_pfe)?;
            Ok(())
        };

        if ws_width as usize == frame_width && ws_height as usize == frame_height && ws_pitch % 4 == 0 && ws_bpp == 4 {
            window_surface.with_lock_mut(|bytes| {
                let data: &mut [u32] = match zerocopy::FromBytes::mut_from_prefix_with_elems(
                    bytes, ws_pitch as usize / 4 * ws_height as usize
                ) {
                    Ok((pixels, _)) => pixels,
                    Err(err) => {
                        eprintln!("Cannot acquire raw FB: {}", err);
                        return Ok(());
                    }
                };

                present_to(true, FrameSliceMut::new(
                    ws_width as usize,
                    ws_height as usize,
                    ws_pitch as usize / 4,
                    data
                ))
            })?;
        } else {
            // Resize ldr buffer to match hdr buffer's size
            ldr_frame_buffer.resize(hdr_frame.stride() * hdr_frame.height(), 0);

            present_to(false, FrameSliceMut::new(
                hdr_frame.width(),
                hdr_frame.height(),
                hdr_frame.stride(),
                &mut ldr_frame_buffer
            ))?;

            // Make sdl2 frame and perform blit
            let ldr_surface = sdl2::surface::Surface::from_data(
                ldr_frame_buffer.as_mut_bytes(),
                hdr_frame.width() as u32,
                hdr_frame.height() as u32,
                hdr_frame.stride() as u32 * 4,
                sdl2::pixels::PixelFormatEnum::ABGR8888
            );
            let mut ldr_surface = match ldr_surface {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("Cannot construct LDR surface from pixels: {}", err);
                    continue 'main_loop;
                }
            };
            if let Err(err) = ldr_surface.set_blend_mode(sdl2::render::BlendMode::None) {
                eprintln!("Failed to set surface blend mode: {}", err);
            }

            let src_rect = sdl2::rect::Rect::new(0, 0, hdr_frame.width() as u32, hdr_frame.height() as u32);
            let dst_rect = sdl2::rect::Rect::new(0, 0, window_surface.width(), window_surface.height());

            if let Err(err) = ldr_surface.blit_scaled(src_rect, &mut window_surface, dst_rect) {
                eprintln!("Surface blit failed: {}", err);
            }
        }

        // Update window
        if let Err(err) = window_surface.update_window() {
            eprintln!("Window update error: {}", err);
        }

        averager.update();
    }

    Ok(())
}
