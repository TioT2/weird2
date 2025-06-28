use std::collections::HashMap;

/// Key code
pub type Key = sdl2::keyboard::Scancode;

#[derive(Copy, Clone)]
pub struct KeyState {
    /// Is key pressed on the current frame
    pub pressed: bool,
    /// Is key 'pressed' state changed in comparison with the previous frame
    pub changed: bool,
}

/// Input controller
pub struct Input {
    states: HashMap<Key, KeyState>,
}

impl Input {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    pub fn get_key_state(&self, key: Key) -> KeyState {
        self.states
            .get(&key)
            .copied()
            .unwrap_or(KeyState {
                pressed: false,
                changed: false,
            })
    }

    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.get_key_state(key).pressed
    }

    pub fn is_key_clicked(&self, key: Key) -> bool {
        let key = self.get_key_state(key);

        key.pressed && key.changed
    }

    pub fn on_state_changed(&mut self, key: Key, pressed: bool) {
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
