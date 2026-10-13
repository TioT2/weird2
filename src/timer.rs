//! Time counter structure

use std::{cell::Cell, collections::BTreeMap, rc::Rc};

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

/// Simple profiler
pub struct Profiler {
    /// Duration between two measures
    measure_duration: std::time::Duration,

    /// Last measure time
    last_measure: std::time::Instant,

    /// Averaged value set
    keys: BTreeMap<String, Rc<Cell<AvgState>>>,
}

/// Averaged value reference
pub struct ValueRef {
    state: Rc<Cell<AvgState>>,
}

impl ValueRef {
    /// Dereference value
    pub fn get(&self) -> f64 {
        self.state.get().avg
    }

    /// Start next measure of referenced value
    pub fn start(&self) -> Measure {
        Measure {
            start: Some(std::time::Instant::now()),
            state: self.state.clone(),
        }
    }

    /// Create a new measure but not start it
    pub fn start_paused(&self) -> Measure {
        Measure {
            start: None,
            state: self.state.clone(),
        }
    }
}

impl Profiler {
    /// Create new value averager
    pub fn new() -> Self {
        Self {
            measure_duration: std::time::Duration::from_secs(1),
            last_measure: std::time::Instant::now(),
            keys: BTreeMap::new(),
        }
    }

    /// Iterate through all values
    pub fn iter<'t>(&'t self) -> impl Iterator<Item = (&'t str, f64)> {
        self.keys.iter().map(|(k, v)| (k.as_str(), v.get().avg))
    }

    /// Try to update frame average values
    pub fn update(&mut self) {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_measure) >= self.measure_duration {
            self.last_measure = now;

            for value in self.keys.values() {
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

    /// Get value by key
    pub fn value(&mut self, key: &str) -> ValueRef {
        let state = match self.keys.get(key) {
            Some(state) => state.clone(),
            None => self.keys.entry(key.to_owned()).or_default().clone(),
        };

        ValueRef { state }
    }

    /// Get an average value
    pub fn get(&self, key: &str) -> Option<f64> {
        Some(self.keys.get(key)?.get().avg)
    }
}

/// Measure handle. Should be obtained right before operation start and destroyed right after operation end.
pub struct Measure {
    /// Measure start
    start: Option<std::time::Instant>,

    /// Measured value state reference
    state: Rc<Cell<AvgState>>,
}

impl Measure {
    /// Pause a measure
    pub fn pause(&mut self) {
        let now = std::time::Instant::now();

        let Some(start) = self.start.take() else {
            return;
        };

        self.state.update(|mut state| {
            state.sum += now.duration_since(start);
            state
        });
    }

    /// Resume a measure
    pub fn resume(&mut self) {
        if self.start.is_some() {
            return;
        }
        self.start = Some(std::time::Instant::now());
    }

    /// Check if measure is paused
    pub const fn is_paused(&self) -> bool {
        self.start.is_none()
    }

    /// Finish a measure
    pub fn finish(self) {}
}

impl Drop for Measure {
    fn drop(&mut self) {
        self.pause();

        self.state.update(|mut state| {
            state.n += 1;
            state
        })

    }
}
