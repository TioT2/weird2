//! Keyboard input state holder implementation

/// Key code
pub type Key = sdl2::keyboard::Scancode;

/// State of the keyboard key
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct KeyState(u8);

impl KeyState {
    const PRESSED: u8 = 0b0000_0001;
    const CHANGED: u8 = 0b0000_0010;

    /// Construct new key state
    pub const fn new(pressed: bool, changed: bool) -> Self {
        let mut res = 0;
        if pressed { res |= Self::PRESSED; }
        if changed { res |= Self::CHANGED; }

        Self(res)
    }

    /// Check if key is changed
    pub const fn pressed(self) -> bool {
        self.0 & Self::PRESSED != 0
    }

    /// Check if key is pressed
    pub const fn changed(self) -> bool {
        self.0 & Self::CHANGED != 0
    }
}

/// Number of keys in input
const KEY_NUMBER: usize = sdl2::keyboard::Scancode::Num as usize;

/// Keyboard state holder
pub struct Input {
    states: Box<[KeyState; KEY_NUMBER]>,
}

impl Default for Input {
    fn default() -> Self {
        Self { states: Box::new([KeyState::new(false, false); _]) }
    }
}

impl Input {
    /// Get state of some key
    pub fn get_key_state(&self, key: Key) -> KeyState {
        self.states[key as usize]
    }

    /// Check key for being pressed
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.get_key_state(key).pressed()
    }

    /// Check key for being clicked
    pub fn is_key_clicked(&self, key: Key) -> bool {
        let key = self.get_key_state(key);
        key.pressed() && key.changed()
    }

    /// Update input on keyboard state change
    pub fn on_state_changed(&mut self, key: Key, pressed: bool) {
        let state = &mut self.states[key as usize];
        *state = KeyState::new(pressed, state.pressed() != pressed);
    }

    /// Release all changed flags on frame end
    pub fn release_changed(&mut self) {
        for state in self.states.iter_mut() {
            *state = KeyState::new(state.pressed(), false);
        }
    }

}
