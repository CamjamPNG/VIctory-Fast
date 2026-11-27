//! Host work that a host library records into a title's command list in
//! place of a console one (the host's FidelityFX upscaler, say): the host
//! device, a title texture's host copy in the state the library is told,
//! and the host list itself.

use std::cell::RefCell;
use std::mem::MaybeUninit;

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES, ID3D12Device,
    ID3D12GraphicsCommandList, ID3D12Resource,
};

use crate::object::This;

/// The host D3D12 device, created on first use; `None` without a host GPU.
pub fn host_device() -> Option<ID3D12Device> {
    crate::host::host().map(|h| h.device.clone())
}

/// Returns the host D3D12 resource backing a title texture without requiring
/// an open command list (used for `ffxConfigure` descriptors such as
/// Frame Generation's `HUDLessColor`).
pub fn host_texture(resource: This) -> Option<ID3D12Resource> {
    crate::resource::host_copy(resource).map(|copy| copy.texture.clone())
}

const MAX_BATCHED_BARRIERS: usize = 16;

struct BarrierBatch {
    barriers: [MaybeUninit<D3D12_RESOURCE_BARRIER>; MAX_BATCHED_BARRIERS],
    len: usize,
}

impl BarrierBatch {
    const fn new() -> Self {
        Self {
            barriers: [const { MaybeUninit::uninit() }; MAX_BATCHED_BARRIERS],
            len: 0,
        }
    }

    fn push(&mut self, list: &ID3D12GraphicsCommandList, barrier: D3D12_RESOURCE_BARRIER) {
        if self.len == MAX_BATCHED_BARRIERS {
            self.flush(list);
        }
        self.barriers[self.len].write(barrier);
        self.len += 1;
    }

    fn flush(&mut self, list: &ID3D12GraphicsCommandList) {
        if self.len == 0 {
            return;
        }
        let n = self.len;
        self.len = 0;
        // SAFETY: `barriers[..n]` were initialized by `push`; `list` is open.
        unsafe {
            let slice = std::slice::from_raw_parts(self.barriers.as_ptr().cast(), n);
            list.ResourceBarrier(slice);
        }
    }
}

/// Prepared host texture returned by [`Recording::prepare_texture`].
pub struct PreparedTexture {
    pub resource: ID3D12Resource,
    pub state: u32,
    pub desc: D3D12_RESOURCE_DESC,
}

/// The active host list for one synchronous external recording call.
pub struct Recording<'a> {
    host: &'a crate::host::HostList,
    barriers: RefCell<BarrierBatch>,
}

impl Recording<'_> {
    /// Move a title texture to the state the provider will see. The list
    /// holds the backing resource through submission.
    pub fn texture(&self, resource: This, state: u32) -> Option<(ID3D12Resource, u32)> {
        let copy = crate::resource::host_copy(resource)?;
        let legal = crate::host::legal_for(copy.flags(), state);
        if let Some((texture, before)) = self.host.transition(resource, state) {
            let barrier = crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(legal as i32),
            );
            self.barriers.borrow_mut().push(&self.host.list, barrier);
        }
        Some((copy.texture.clone(), legal))
    }

    /// Prepare aliases, mip uploads and depth twins like a compute SRV.
    pub fn prepare_input(&self, resource: This) -> Option<()> {
        let copy = crate::resource::host_copy(resource)?;
        self.prepare_input_copy(resource, &copy.texture);
        Some(())
    }

    fn prepare_input_copy(&self, resource: This, texture: &ID3D12Resource) {
        // Flush any queued barriers before alias/upload/twin helpers record commands.
        self.flush_barriers();
        let aliased = crate::resource::alias_of(resource)
            .is_some_and(|source| crate::texture::alias(self.host, resource, source));
        if !aliased && crate::texture::title_written(texture) {
            crate::texture::upload_from(self.host, resource, 0);
        }
        self.host.twin_use(resource, false, false);
    }

    /// Single-lookup preparation for an external pass (FidelityFX upscale /
    /// frame generation): prepares inputs if requested, checks cached resource
    /// flags without redundant `GetDesc()` calls, queues the state transition
    /// into the batch, and returns the host resource, legal state, and desc.
    pub fn prepare_texture(
        &self,
        resource: This,
        state: u32,
        is_input: bool,
    ) -> Option<PreparedTexture> {
        let copy = crate::resource::host_copy(resource)?;
        if is_input {
            self.prepare_input_copy(resource, &copy.texture);
        }
        let legal = crate::host::legal_for(copy.flags(), state);
        if legal != state {
            return None;
        }
        if let Some((texture, before)) = self.host.transition(resource, state) {
            let barrier = crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(legal as i32),
            );
            self.barriers.borrow_mut().push(&self.host.list, barrier);
        }
        // SAFETY: `copy.texture` is a live D3D12 resource.
        let desc = unsafe { copy.texture.GetDesc() };
        Some(PreparedTexture {
            resource: copy.texture.clone(),
            state: legal,
            desc,
        })
    }

    /// Flush all queued resource transitions in a single `ResourceBarrier` call.
    pub fn flush_barriers(&self) {
        self.barriers.borrow_mut().flush(&self.host.list);
    }

    /// Access flushes any pending host barriers before the provider runs.
    pub fn commands(&self) -> &ID3D12GraphicsCommandList {
        self.flush_barriers();
        &self.host.list
    }
}

/// The producer recorded its output on the host's command list.
pub fn output_written(resource: This) {
    crate::texture::host_written(resource);
}

/// Settle submitted title work before releasing a provider's private GPU resources.
pub fn finish_submitted() {
    crate::queue::before_sync();
    crate::host::settle_for_fence();
}

/// Runs `f` with the host list behind the title list `list` (open for
/// recording); `None` without one. What `f` binds (descriptor heaps, root
/// signatures, pipeline) is replaced before the list's next draw or
/// dispatch.
pub fn record<R>(list: This, f: impl FnOnce(&Recording<'_>) -> R) -> Option<R> {
    crate::list::thin::with_external(list, |host, state| {
        let recording = Recording {
            host,
            barriers: RefCell::new(BarrierBatch::new()),
        };
        let out = f(&recording);
        recording.flush_barriers();
        host.mark_uav_dirty();
        state.external_compute();
        out
    })
}
