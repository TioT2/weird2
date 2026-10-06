//! Time counter structure

use std::{cell::{Cell, RefCell}, collections::HashMap, rc::Rc};

/// Time measure utility
#[derive(Clone)]
pub struct Timer {
    /// Timer initialization time
    start: std::time::Instant,

    /// Last update time
    now: std::time::Instant,

    /// Duration between last two timer updates
    dt: std::time::Duration,

    /// Amount of timer updates from initialization moment
    total_frame_count: u64,

    /// Current FPS updating duration
    fps_duration: std::time::Duration,

    /// Last time FPS was measured
    fps_last_measure: std::time::Instant,

    /// Count of timer updates after last FPS recalculation
    fps_frame_counter: usize,

    /// Frames Per Second
    fps: Option<f32>,
}

impl Timer {
    /// Create new timer starting from moment of creation
    pub fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            start: now,
            now,
            dt: std::time::Duration::from_millis(60),
            total_frame_count: 0,
            
            fps_duration: std::time::Duration::from_millis(1000),
            fps_last_measure: now,
            fps_frame_counter: 0,
            fps: None,
        }
    }

    /// Update timer
    pub fn response(&mut self) {
        let new_now = std::time::Instant::now();
        self.dt = new_now.duration_since(self.now);
        self.now = new_now;
        
        self.total_frame_count += 1;
        self.fps_frame_counter += 1;

        let fps_delta = self.now - self.fps_last_measure;
        if fps_delta > self.fps_duration {
            self.fps = Some(self.fps_frame_counter as f32 / fps_delta.as_secs_f32());

            self.fps_last_measure = self.now;
            self.fps_frame_counter = 0;
        }
    }

    /// Get current duration between frames
    pub fn get_delta_time(&self) -> f32 {
        self.dt.as_secs_f32()
    }

    /// Get current time
    pub fn get_time(&self) -> f32 {
        self.now
            .duration_since(self.start)
            .as_secs_f32()
    }

    /// Get current framaes-per-second
    pub fn get_fps(&self) -> f32 {
        self.fps.unwrap_or(f32::NAN)
    }

    /// Set new frame-per-second calculation duration
    pub fn set_fps_duration(&mut self, new_fps_duration: std::time::Duration) {
        self.fps_duration = new_fps_duration;
        self.fps_frame_counter = 0;
        self.fps = None;
        self.fps_last_measure = std::time::Instant::now();
    }

    /// Get count of frames elapsed from start
    pub fn get_frame_count(&self) -> u64 {
        self.total_frame_count
    }
}

/// Average value state
#[derive(Copy, Clone, Default)]
struct AvgState {
    /// Duration sum
    sum: std::time::Duration,

    /// Average value
    avg: f64,

    /// Amount of cells
    n: u32,
}

/// Some value averager
pub struct Averager {
    /// Duration between two measures
    measure_duration: std::time::Duration,

    /// Last measure time
    last_measure: std::time::Instant,

    /// Averaged value set
    keys: RefCell<HashMap<String, Rc<Cell<AvgState>>>>,
}

impl Averager {
    /// Create new value averager
    pub fn new() -> Self {
        Self {
            measure_duration: std::time::Duration::from_secs(1),
            last_measure: std::time::Instant::now(),
            keys: RefCell::new(HashMap::new()),
        }
    }

    /// Try to update frame average values
    pub fn update(&mut self) {
        let keys = self.keys.borrow_mut();

        let now = std::time::Instant::now();
        if now.duration_since(self.last_measure) >= self.measure_duration {
            self.last_measure = now;
            for value in keys.values() {
                value.update(|state| AvgState {
                    sum: std::time::Duration::default(),
                    avg: if state.n != 0 {
                        state.sum.as_secs_f64() / state.n as f64
                    } else {
                        state.avg
                    },
                    n: 0
                });
            }
        }
    }

    /// Get an average value
    pub fn get(&self, key: &str) -> Option<f64> {
        Some(self.keys.borrow().get(key)?.get().avg)
    }

    /// Start a measure
    pub fn start_measure(&self, key: &str) -> Measure {
        let mut keys = self.keys.borrow_mut();

        Measure {
            start: std::time::Instant::now(),
            state: match keys.get(key) {
                Some(state) => state.clone(),
                None => keys.entry(key.to_owned()).or_default().clone()
            }
        }
    }

    /// Run measuring function
    pub fn measure<T>(&self, key: &str, f: impl FnOnce() -> T) -> T {
        let _m = self.start_measure(key);
        f()
    }
}

/// Measure handle. Should be obtained right before operation start and destroyed right after operation end.
pub struct Measure {
    /// Measure start
    start: std::time::Instant,

    /// Measured value state reference
    state: Rc<Cell<AvgState>>,
}

impl Measure {
    /// Finish a measure
    pub fn finish(self) {}
}

impl Drop for Measure {
    fn drop(&mut self) {
        let now = std::time::Instant::now();
        self.state.update(|mut state| {
            state.sum += now.duration_since(self.start);
            state.n += 1;
            state
        })
    }
}
