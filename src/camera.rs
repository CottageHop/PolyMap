use glam::{Mat4, Vec2, Vec3};

/// Camera for a 3D map view with smooth zooming and adjustable tilt.
pub struct Camera {
    /// World-space position the camera is looking at
    pub position: Vec2,
    /// Target position (for smooth interpolation)
    target_position: Vec2,
    /// Current zoom level (logarithmic scale — each +1.0 = 2x closer)
    pub zoom: f32,
    /// Target zoom (for smooth interpolation)
    target_zoom: f32,
    /// Camera tilt angle in radians (0 = top-down, PI/2 = horizon)
    pub tilt: f32,
    /// Target tilt
    target_tilt: f32,
    /// Viewport dimensions in pixels
    pub viewport: Vec2,
    /// Interpolation speed
    pub smoothing: f32,
    /// Cached VP matrix (rebuilt only when camera moves)
    cached_vp: Mat4,
    cached_inv_vp: Mat4,
    /// True if camera state changed since last matrix build
    pub dirty: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// View-projection matrix (4x4)
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space (x, y)
    pub position: [f32; 2],
    /// Viewport size in pixels (width, height)
    pub viewport: [f32; 2],
    /// Zoom factor
    pub zoom: f32,
    /// Time in seconds (for animations)
    pub time: f32,
    /// Tilt angle in radians
    pub tilt: f32,
    pub _padding: f32,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            position: [0.0, 0.0],
            viewport: [800.0, 600.0],
            zoom: 0.8,
            time: 0.0,
            tilt: 1.5,
            _padding: 0.0,
        }
    }
}

impl Camera {
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        let mut cam = Self {
            position: Vec2::ZERO,
            target_position: Vec2::ZERO,
            zoom: -0.5,
            target_zoom: -0.5,
            tilt: 1.48,
            target_tilt: 1.48,
            viewport: Vec2::new(viewport_width, viewport_height),
            smoothing: 15.0,
            cached_vp: Mat4::IDENTITY,
            cached_inv_vp: Mat4::IDENTITY,
            dirty: true,
        };
        cam.rebuild_cache();
        cam
    }

    /// Zoom in/out by delta, centered on a screen-space pixel position.
    /// The world point under `screen_pos` stays fixed after zooming.
    pub fn zoom_at(&mut self, delta: f32, screen_pos: Vec2) {
        let old_zoom = self.target_zoom;
        let new_zoom = (old_zoom + delta).clamp(-2.0, 2.5);
        if (new_zoom - old_zoom).abs() < 1e-6 {
            return;
        }

        // Offset from screen center in pixels
        let center = self.viewport * 0.5;
        let pixel_offset = Vec2::new(screen_pos.x - center.x, -(screen_pos.y - center.y));

        // World-space offset at old vs new zoom
        let old_scale = self.world_units_per_pixel_at(old_zoom);
        let new_scale = self.world_units_per_pixel_at(new_zoom);

        // Shift so the world point under the cursor stays put
        self.target_position += pixel_offset * (old_scale - new_scale);
        self.target_zoom = new_zoom;
    }

    /// How many world units one pixel represents at a given zoom level
    fn world_units_per_pixel_at(&self, zoom: f32) -> f32 {
        100.0 / (2.0_f32.powf(zoom) * self.viewport.x.min(self.viewport.y))
    }

    /// Tilt the camera. Positive = more tilted (toward horizon).
    pub fn tilt_by(&mut self, delta: f32) {
        self.target_tilt = (self.target_tilt + delta).clamp(0.4, 1.5);
    }

    /// Pan the camera by a pixel offset (converted to world space)
    pub fn pan_by(&mut self, pixel_delta: Vec2) {
        let world_scale = self.world_units_per_pixel();
        self.target_position -= Vec2::new(pixel_delta.x, -pixel_delta.y) * world_scale;
    }

    /// How many world units one pixel represents at current zoom
    fn world_units_per_pixel(&self) -> f32 {
        self.world_units_per_pixel_at(self.zoom)
    }

    /// Smooth update — call once per frame with delta time.
    /// Returns true if camera moved (dirty).
    pub fn update(&mut self, dt: f32) -> bool {
        let old_pos = self.position;
        let old_zoom = self.zoom;
        let old_tilt = self.tilt;

        let t = 1.0 - (-self.smoothing * dt).exp();
        self.position = self.position.lerp(self.target_position, t);
        self.zoom = self.zoom + (self.target_zoom - self.zoom) * t;
        self.tilt = self.tilt + (self.target_tilt - self.tilt) * t;

        // Snap to target when very close to prevent subpixel jitter
        if (self.position - self.target_position).length() < 1e-4 {
            self.position = self.target_position;
        }
        if (self.zoom - self.target_zoom).abs() < 1e-4 {
            self.zoom = self.target_zoom;
        }
        if (self.tilt - self.target_tilt).abs() < 1e-4 {
            self.tilt = self.target_tilt;
        }

        // Check if camera materially changed
        let moved = (self.position - old_pos).length() > 1e-5
            || (self.zoom - old_zoom).abs() > 1e-5
            || (self.tilt - old_tilt).abs() > 1e-5;

        if moved {
            self.dirty = true;
            self.rebuild_cache();
        }

        moved
    }

    /// Rebuild the cached VP and inverse VP matrices.
    fn rebuild_cache(&mut self) {
        self.cached_vp = self.build_view_proj();
        self.cached_inv_vp = self.cached_vp.inverse();
        self.dirty = false;
    }

    /// Build the view-projection matrix for 3D rendering.
    fn build_view_proj(&self) -> Mat4 {
        let aspect = self.viewport.x / self.viewport.y;
        let zoom_scale = 100.0 / 2.0_f32.powf(self.zoom);
        let half_w = zoom_scale * aspect * 0.5;
        let half_h = zoom_scale * 0.5;

        // Scale near/far with zoom so clipping planes never cut through visible geometry.
        // At max zoom-out the view spans ~400 units; near/far must exceed this.
        let clip_depth = (zoom_scale * 2.0).max(100.0);
        let proj = Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, -clip_depth, clip_depth);

        // View: look at the camera position from above, tilted
        let eye = Vec3::new(
            self.position.x,
            self.position.y - self.tilt.cos() * 20.0,
            self.tilt.sin() * 20.0,
        );
        let target = Vec3::new(self.position.x, self.position.y, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let view = Mat4::look_at_rh(eye, target, up);

        proj * view
    }

    /// Build the uniform to send to the GPU
    pub fn uniform(&self, time: f32) -> CameraUniform {
        CameraUniform {
            view_proj: self.cached_vp.to_cols_array_2d(),
            position: self.position.into(),
            viewport: self.viewport.into(),
            zoom: self.zoom,
            time,
            tilt: self.tilt,
            _padding: 0.0,
        }
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = zoom;
        self.target_zoom = zoom;
        self.rebuild_cache();
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.viewport = Vec2::new(width, height);
        self.rebuild_cache();
    }

    /// Set the target position for smooth panning.
    pub fn set_target_position(&mut self, x: f32, y: f32) {
        self.target_position = Vec2::new(x, y);
    }

    /// Set the target zoom for smooth zooming.
    pub fn set_target_zoom(&mut self, zoom: f32) {
        self.target_zoom = zoom.clamp(-2.0, 2.5);
    }

    /// Set the target tilt for smooth tilting.
    pub fn set_target_tilt(&mut self, tilt: f32) {
        self.target_tilt = tilt.clamp(0.4, 1.5);
    }

    /// Unproject screen pixel coordinates to world XY (at z=0 plane).
    pub fn unproject_to_world(&self, screen_x: f32, screen_y: f32) -> Vec2 {
        let ndc_x = (screen_x / self.viewport.x) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / self.viewport.y) * 2.0;
        let near = self.cached_inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = self.cached_inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near3 = glam::Vec3::new(near.x / near.w, near.y / near.w, near.z / near.w);
        let far3 = glam::Vec3::new(far.x / far.w, far.y / far.w, far.z / far.w);
        let dir = far3 - near3;
        if dir.z.abs() < 1e-8 {
            return Vec2::new(near3.x, near3.y);
        }
        let t = -near3.z / dir.z;
        Vec2::new(near3.x + dir.x * t, near3.y + dir.y * t)
    }

    /// Project a world position (x, y, z) to screen pixel coordinates.
    /// Returns None if the point is behind the camera.
    pub fn project_to_screen(&self, world_x: f32, world_y: f32, world_z: f32) -> Option<Vec2> {
        let clip = self.cached_vp * glam::Vec4::new(world_x, world_y, world_z, 1.0);
        if clip.w.abs() < 1e-6 {
            return None;
        }
        let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        // NDC to pixel coordinates
        let screen_x = (ndc.x * 0.5 + 0.5) * self.viewport.x;
        let screen_y = (1.0 - (ndc.y * 0.5 + 0.5)) * self.viewport.y;
        Some(Vec2::new(screen_x, screen_y))
    }
}
