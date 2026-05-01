use foreign_types::ForeignTypeRef;
use metal::{BufferRef, CommandQueueRef, DeviceRef};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLDevice};

pub(super) unsafe fn buffer_ref(buffer: &BufferRef) -> &ProtocolObject<dyn MTLBuffer> {
    // `metal` and `objc2-metal` both wrap the same Objective-C object pointer.
    &*(buffer.as_ptr().cast())
}

pub(super) unsafe fn command_queue_ref(
    queue: &CommandQueueRef,
) -> &ProtocolObject<dyn MTLCommandQueue> {
    &*(queue.as_ptr().cast())
}

pub(super) unsafe fn device_ref(device: &DeviceRef) -> &ProtocolObject<dyn MTLDevice> {
    &*(device.as_ptr().cast())
}
