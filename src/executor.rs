use crate::Error;
//use nix::unistd::Pid;

pub mod forkserver;
pub mod pipes;
pub mod rpc;
pub mod shmem;

pub use forkserver::BitmapTracer;
pub use forkserver::ForkServerExecutor;
pub mod frontend;

pub use crate::datatype::ExecutionStatus;

// Constructor the Feedback after execution.
pub trait Tracer<'a, T> {
    fn get_feedback(&'a self) -> T;
}

// Ideally we only need to specify `I` as input for Executor.
pub trait Executor<'a, T, F, I> {
    // This shouldn't be in trait.
    fn new(
        tracer: T,
        args: Vec<String>,
        working_dir: Option<String>,
        timeout: u64,
        bind_to_cpu: bool,
    ) -> Result<Self, Error>
    where
        Self: Sized;
    // TODO(yongheng): This should be a inner api.
    fn run_target(&'a mut self, input: &I) -> Result<ExecutionStatus, Error>;

    // Execute the input and produce the feedback for feedback collector.
    fn execute(&'a mut self, input: I) -> crate::datatype::Feedback;
}
