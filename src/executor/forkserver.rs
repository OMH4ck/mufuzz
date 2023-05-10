use crate::datatype;
use crate::datatype::{Feedback, FeedbackData, NewBit, TestCase};
use crate::util;
use crate::Error;
use std::ops::Drop;

use nix::{
    sys::{
        signal::{kill, Signal},
        time::{TimeSpec, TimeValLike},
    },
    unistd::Pid,
};
use std::fs::{remove_file, File, OpenOptions};
use std::io::{self, prelude::*, ErrorKind, SeekFrom};

use crate::datatype::ExecutionStatus;
use crate::executor::{pipes::Pipe, shmem::*};
use crate::executor::{Executor, Tracer};
use std::os::unix::{
    io::{AsRawFd, RawFd},
    process::CommandExt,
};
use std::process::{Command, Stdio};

use crate::feedback::NewBitFilter;

pub struct BitmapTracer {
    pub bitmap: CommonUnixShMem,
    filter: NewBitFilter,
}

impl BitmapTracer {
    pub fn new(size: usize) -> Self {
        BitmapTracer {
            bitmap: CommonUnixShMemProvider::new()
                .unwrap()
                .new_map(size)
                .unwrap(),
            filter: NewBitFilter::new(size),
        }
    }

    pub fn clear(&mut self) {
        for b in self.bitmap.map_mut() {
            *b = 0;
        }
    }

    // One pass to collect all the new bits and update virgin bits
    pub fn construct_feedback(&mut self) -> Option<Vec<NewBit>> {
        let result = self.filter.filter(self.bitmap.map_mut());
        self.clear();
        result
    }

    pub fn construct_feedback_without_chaning_virgin(&mut self) -> Option<Vec<NewBit>> {
        let result = self
            .filter
            .filter_without_chaning_virgin(self.bitmap.map_mut());
        self.clear();
        result
    }

    pub fn update_bits(&mut self, bits: &Vec<NewBit>) {
        self.filter.update_bits(bits);
    }
}

impl<'a> Tracer<'a, &'a [u8]> for BitmapTracer {
    fn get_feedback(&'a self) -> &[u8] {
        self.bitmap.map()
    }
}

pub struct OutFile {
    file: File,
    file_name: String,
}

impl Drop for OutFile {
    fn drop(&mut self) {
        //drop(self.file);
        match remove_file(&self.file_name) {
            Ok(_) => (),
            Err(_) => {
                println!("Error in delete files");
            }
        }
    }
}

impl OutFile {
    pub fn new(file_name: &str) -> Result<Self, Error> {
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(file_name)?;
        Ok(Self {
            file: f,
            file_name: file_name.to_owned(),
        })
    }

    #[must_use]
    pub fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    pub fn write_buf(&mut self, buf: &[u8]) {
        self.rewind();
        // TODO: Potential opt1: Set len before write_all.
        // TODO: Potential opt2: Skip set_len if the current len is smaller.
        self.file.write_all(buf).unwrap();
        self.file.set_len(buf.len() as u64).unwrap();
        self.file.flush().unwrap();
        // Rewind again otherwise the target will not read stdin from the beginning
        //TODO: Potential opt3: Skip rewind if we don't use stdin for output fd.
        self.rewind();
    }

    pub fn rewind(&mut self) {
        self.file.seek(SeekFrom::Start(0)).unwrap();
    }
}

pub trait ConfigTarget {
    fn setsid(&mut self) -> &mut Self;
    fn setlimit(&mut self, memlimit: u64) -> &mut Self;
    fn setstdin(&mut self, fd: RawFd, use_stdin: bool) -> &mut Self;
    fn setpipe(
        &mut self,
        st_read: RawFd,
        st_write: RawFd,
        ctl_read: RawFd,
        ctl_write: RawFd,
    ) -> &mut Self;
}

const FORKSRV_FD: i32 = 198;
#[allow(clippy::cast_possible_wrap)]
const FS_OPT_ENABLED: i32 = 0x80000001u32 as i32;
#[allow(clippy::cast_possible_wrap)]
const FS_OPT_SHDMEM_FUZZ: i32 = 0x01000000u32 as i32;
// const SHMEM_FUZZ_HDR_SIZE: usize = 4;
// const MAX_FILE: usize = 1024 * 1024;

fn dup2(fd: i32, device: i32) -> Result<(), Error> {
    match unsafe { libc::dup2(fd, device) } {
        -1 => Err(Error::File(std::io::Error::last_os_error())),
        _ => Ok(()),
    }
}

impl ConfigTarget for Command {
    fn setsid(&mut self) -> &mut Self {
        let func = move || {
            unsafe {
                libc::setsid();
            };
            Ok(())
        };
        unsafe { self.pre_exec(func) }
    }

    fn setpipe(
        &mut self,
        st_read: RawFd,
        st_write: RawFd,
        ctl_read: RawFd,
        ctl_write: RawFd,
    ) -> &mut Self {
        let func = move || {
            match dup2(ctl_read, FORKSRV_FD) {
                Ok(_) => (),
                Err(_) => {
                    return Err(io::Error::last_os_error());
                }
            }

            match dup2(st_write, FORKSRV_FD + 1) {
                Ok(_) => (),
                Err(_) => {
                    return Err(io::Error::last_os_error());
                }
            }
            unsafe {
                libc::close(st_read);
                libc::close(st_write);
                libc::close(ctl_read);
                libc::close(ctl_write);
            }
            Ok(())
        };
        unsafe { self.pre_exec(func) }
    }

    fn setstdin(&mut self, fd: RawFd, use_stdin: bool) -> &mut Self {
        if use_stdin {
            let func = move || {
                match dup2(fd, libc::STDIN_FILENO) {
                    Ok(_) => (),
                    Err(_) => {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            };
            unsafe { self.pre_exec(func) }
        } else {
            self
        }
    }

    fn setlimit(&mut self, memlimit: u64) -> &mut Self {
        if memlimit == 0 {
            return self;
        }
        let func = move || {
            let memlimit: libc::rlim_t = (memlimit as libc::rlim_t) << 20;
            let r = libc::rlimit {
                rlim_cur: memlimit,
                rlim_max: memlimit,
            };
            let r0 = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };

            #[cfg(target_os = "openbsd")]
            let mut ret = unsafe { libc::setrlimit(libc::RLIMIT_RSS, &r) };
            #[cfg(not(target_os = "openbsd"))]
            let mut ret = unsafe { libc::setrlimit(libc::RLIMIT_AS, &r) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            ret = unsafe { libc::setrlimit(libc::RLIMIT_CORE, &r0) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        };
        unsafe { self.pre_exec(func) }
    }
}

struct ForkServer {
    st_pipe: Pipe,
    ctl_pipe: Pipe,
    child_pid: Pid,
    status: i32,
    last_run_timed_out: i32,
    // The forkserver fd in AFL/AFLPlusplus is 198/199, when the pipe fd is also 198/199,
    // the forkserver will fail to run. Therefore, we have to skip through these two numbers.
    // Currently I just open 10 files to increase the fd value. It is ugly but works.
    _useless_fds: Option<Vec<OutFile>>,
}

fn increase_fd_num(n: usize) -> Vec<OutFile> {
    let mut v = Vec::default();
    for _i in 0..n {
        let out_filename = format!("/tmp/use_less{}", datatype::get_id());
        v.push(OutFile::new(&out_filename).unwrap());
    }
    v
}

impl ForkServer {
    pub fn new(
        args: Vec<String>,
        out_filefd: RawFd,
        use_stdin: bool,
        memlimit: u64,
        bind_to_cpu: bool,
    ) -> Result<Self, Error> {
        let mut st_pipe = Pipe::new().unwrap();
        let mut ctl_pipe = Pipe::new().unwrap();
        let mut fds = None;

        // Try to avoid pipe fd being 198/199, which is FORKSERVER FD
        if st_pipe.read_end().unwrap() >= 190 && st_pipe.read_end().unwrap() <= 200 {
            fds = Some(increase_fd_num(10));
            st_pipe = Pipe::new().unwrap();
        }
        if ctl_pipe.read_end().unwrap() >= 190 && ctl_pipe.read_end().unwrap() <= 200 {
            fds = Some(increase_fd_num(10));
            ctl_pipe = Pipe::new().unwrap();
        }

        match Command::new(args[0].clone())
            .args(&args[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .env("LD_BIND_LAZY", "1")
            .setlimit(memlimit)
            .setsid()
            .setstdin(out_filefd, use_stdin)
            .setpipe(
                st_pipe.read_end().unwrap(),
                st_pipe.write_end().unwrap(),
                ctl_pipe.read_end().unwrap(),
                ctl_pipe.write_end().unwrap(),
            )
            .spawn()
        {
            Ok(child) => {
                println!("Pid: {}", child.id());
                if bind_to_cpu {
                    util::bind_to_cpu(child.id() as i32);
                }
            }
            Err(err) => {
                return Err(Error::ForkServer(format!(
                    "Could not spawn the forkserver: {:#?}",
                    err
                )));
            }
        };

        // Ctl_pipe.read_end and st_pipe.write_end are unnecessary for the parent, so we'll close them
        ctl_pipe.close_read_end();
        st_pipe.close_write_end();

        Ok(Self {
            st_pipe,
            ctl_pipe,
            child_pid: Pid::from_raw(0),
            status: 0,
            last_run_timed_out: 0,
            _useless_fds: fds,
        })
    }

    pub fn last_run_timed_out(&self) -> i32 {
        self.last_run_timed_out
    }

    #[allow(dead_code)]
    pub fn set_last_run_timed_out(&mut self, last_run_timed_out: i32) {
        self.last_run_timed_out = last_run_timed_out;
    }

    #[must_use]
    pub fn status(&self) -> i32 {
        self.status
    }

    pub fn set_status(&mut self, status: i32) {
        self.status = status;
    }

    //#[must_use]
    #[allow(dead_code)]
    pub fn child_pid(&self) -> Pid {
        self.child_pid
    }

    pub fn set_child_pid(&mut self, child_pid: Pid) {
        self.child_pid = child_pid;
    }

    pub fn read_st(&mut self) -> Result<(usize, i32), Error> {
        let mut buf: [u8; 4] = [0u8; 4];

        let rlen = self.st_pipe.read(&mut buf)?;
        let val: i32 = i32::from_ne_bytes(buf);

        Ok((rlen, val))
    }

    pub fn write_ctl(&mut self, val: i32) -> Result<usize, Error> {
        let slen = self.ctl_pipe.write(&val.to_ne_bytes())?;

        Ok(slen)
    }

    #[allow(dead_code)]
    pub fn read_st_timed(&mut self, timeout: &TimeSpec) -> Result<Option<i32>, Error> {
        let mut buf: [u8; 4] = [0u8; 4];
        let st_read = match self.st_pipe.read_end() {
            Some(fd) => fd,
            None => {
                return Err(Error::File(io::Error::new(
                    ErrorKind::BrokenPipe,
                    "Read pipe end was already closed",
                )));
            }
        };
        let pollfds = &mut [nix::poll::PollFd::new(
            st_read,
            nix::poll::PollFlags::POLLIN,
        )];
        let sret = nix::poll::poll(pollfds, timeout.num_milliseconds() as libc::c_int)?;
        /*
        let mut readfds = FdSet::new();
        readfds.insert(st_read);
        // We'll pass a copied timeout to keep the original timeout intact, because select updates timeout to indicate how much time was left. See select(2)
        let sret = pselect(
            Some(readfds.highest().unwrap() + 1),
            &mut readfds,
            None,
            None,
            Some(timeout),
            Some(&SigSet::empty()),
        )?;
        */
        if sret > 0 {
            if self.st_pipe.read_exact(&mut buf).is_ok() {
                let val: i32 = i32::from_ne_bytes(buf);
                Ok(Some(val))
            } else {
                Err(Error::ForkServer(
                    "Unable to communicate with fork server (OOM?)".to_string(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

pub struct ForkServerExecutor<T> {
    pub tracer: T,
    // args: Vec<String>,
    out_file: OutFile,
    forkserver: ForkServer,
    timeout: TimeSpec,
    signal: Signal,
}

impl<T> ForkServerExecutor<T> {
    pub fn set_timeout(&mut self, timeout: TimeSpec) -> TimeSpec {
        let old_timeout = self.timeout;
        self.timeout = timeout;
        old_timeout
    }
}

impl<'a> Executor<'a, BitmapTracer, &'a [u8], TestCase> for ForkServerExecutor<BitmapTracer> {
    fn new(
        tracer: BitmapTracer,
        arguments: Vec<String>,
        working_dir: Option<String>,
        timeout: u64,
        bind_to_cpu: bool,
    ) -> Result<Self, Error> {
        let mut args = Vec::<String>::new();
        let mut use_stdin = true;
        let working_dir = match working_dir {
            Some(working_dir) => working_dir,
            None => ".".to_string(),
        };
        let out_filename = format!("{}/.cur_input{}", working_dir, datatype::get_id());
        let out_file = OutFile::new(&out_filename)?;

        for item in arguments {
            if item == "@@" && use_stdin {
                use_stdin = false;
                args.push(out_filename.clone());
            } else {
                args.push(item.to_string());
            }
        }

        let mut forkserver = ForkServer::new(
            args.clone(),
            out_file.as_raw_fd(),
            use_stdin,
            0,
            bind_to_cpu,
        )?;
        let (rlen, status) = forkserver.read_st()?; // Initial handshake, read 4-bytes hello message from the forkserver.

        if rlen != 4 {
            return Err(Error::ForkServer(
                "Failed to start a forkserver".to_string(),
            ));
        }
        println!("All right - fork server is up.");
        let use_shmem_test_case = false;
        // If forkserver is responding, we then check if there's any option enabled.
        if status & FS_OPT_ENABLED == FS_OPT_ENABLED {
            if (status & FS_OPT_SHDMEM_FUZZ == FS_OPT_SHDMEM_FUZZ) & use_shmem_test_case {
                println!("Using SHARED MEMORY FUZZING feature.");
                let send_status = FS_OPT_ENABLED | FS_OPT_SHDMEM_FUZZ;

                let send_len = forkserver.write_ctl(send_status)?;
                if send_len != 4 {
                    return Err(Error::ForkServer(
                        "Writing to forkserver failed.".to_string(),
                    ));
                }
            }
        } else {
            println!("ForkServer Options are not available.");
        }

        Ok(Self {
            tracer,
            // args,
            out_file,
            forkserver,
            timeout: TimeSpec::milliseconds(timeout as i64),
            signal: Signal::SIGKILL,
        })
    }

    // Execute the test case and construct the feedback.
    fn execute(&mut self, test_case: TestCase) -> Feedback {
        let result = self.run_target(&test_case);
        assert!(result.is_ok());

        let status = result.unwrap();
        let mutation_info = test_case.extract_mutation_info();
        let mut feedback = Feedback::new(status).set_mutation_info(mutation_info);
        match status {
            ExecutionStatus::Ok => {
                if let Some(new_bits) = self.tracer.construct_feedback() {
                    feedback = Feedback::new(ExecutionStatus::Interesting)
                        .set_data(FeedbackData::new_coverage(new_bits))
                        .set_test_case(test_case);
                }
            }
            ExecutionStatus::Crash => {
                feedback = feedback.set_test_case(test_case);
            }
            ExecutionStatus::Timeout => {
                if let Some(new_bits) = self.tracer.construct_feedback_without_chaning_virgin() {
                    if !new_bits.is_empty() {
                        let old_timeout = self.set_timeout(TimeSpec::milliseconds(1000));
                        let result = self.run_target(&test_case);
                        self.tracer.clear();
                        self.set_timeout(old_timeout);
                        assert!(result.is_ok());
                        let mut result = result.unwrap();
                        if result != ExecutionStatus::Timeout {
                            if result == ExecutionStatus::Ok {
                                result = ExecutionStatus::Interesting;
                            }
                            self.tracer.update_bits(&new_bits);
                            feedback = Feedback::new(result)
                                .set_data(FeedbackData::new_coverage(new_bits))
                                .set_test_case(test_case);
                        }
                    }
                }
            }
            ExecutionStatus::Interesting => {
                unreachable!()
            }
        }

        feedback
    }

    fn run_target(&mut self, input: &TestCase) -> Result<ExecutionStatus, Error> {
        let mut exit_kind = ExecutionStatus::Ok;
        let last_run_timed_out = self.forkserver.last_run_timed_out();

        self.out_file.write_buf(input.get_buffer());

        let send_len = self.forkserver.write_ctl(last_run_timed_out)?;
        if send_len != 4 {
            return Err(Error::ForkServer(
                "Unable to request new process from fork server (OOM?)".to_string(),
            ));
        }

        let (recv_pid_len, pid) = self.forkserver.read_st()?;
        if recv_pid_len != 4 {
            return Err(Error::ForkServer(
                "Unable to request new process from fork server (OOM?)".to_string(),
            ));
        }

        if pid <= 0 {
            return Err(Error::ForkServer(
                "Fork server is misbehaving (OOM?)".to_string(),
            ));
        }

        self.forkserver.set_child_pid(Pid::from_raw(pid));

        if let Some(status) = self.forkserver.read_st_timed(&self.timeout)? {
            self.forkserver.set_status(status);
            if libc::WIFSIGNALED(self.forkserver.status()) {
                exit_kind = ExecutionStatus::Crash;
            }
        } else {
            self.forkserver.set_last_run_timed_out(1);

            // We need to kill the child in case he has timed out, or we can't get the correct pid in the next call to self.executor.forkserver_mut().read_st()?
            let _ = kill(self.forkserver.child_pid(), self.signal);
            let (recv_status_len, _) = self.forkserver.read_st()?;
            if recv_status_len != 4 {
                return Err(Error::ForkServer(
                    "Could not kill timed-out child".to_string(),
                ));
            }
            exit_kind = ExecutionStatus::Timeout;
        }

        self.forkserver.set_child_pid(Pid::from_raw(0));

        Ok(exit_kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;
    use serial_test::serial;

    #[test]
    #[serial]
    fn unlimited_num_of_forkserver_can_be_created() {
        const MAP_SIZE: usize = 65536;
        let mut v = Vec::default();
        for _i in 0..100 {
            let args = vec![util::get_test_bin_by_name("timeout_prog").unwrap()];
            let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
            bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();
            let executor = ForkServerExecutor::new(bitmap_tracer, args, None, 20, true);
            assert!(executor.is_ok());
            v.push(executor.unwrap());
        }
        assert_eq!(v.len(), 100);
    }

    #[test]
    #[serial]
    fn forkserver_handshake_fails_on_uninstrumented_binary() {
        const MAP_SIZE: usize = 65536;
        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![String::from("echo"), String::from("@@")];
        let executor = ForkServerExecutor::new(bitmap_tracer, args, None, 20, true);
        let result = match executor {
            Ok(_) => false,
            Err(e) => match e {
                Error::ForkServer(s) => s == "Failed to start a forkserver",
                _ => false,
            },
        };
        assert!(result);
    }

    #[test]
    #[serial]
    fn forkserver_timeout_correctly() {
        const MAP_SIZE: usize = 65536;
        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![util::get_test_bin_by_name("timeout_prog").unwrap()];
        let mut executor = ForkServerExecutor::new(bitmap_tracer, args, None, 20, true)
            .ok()
            .unwrap();

        let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);

        let result = match executor.run_target(&test_case) {
            Ok(ExecutionStatus::Timeout) => true,
            Ok(_) => false,
            Err(_e) => false,
        };
        assert!(result);

        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![util::get_test_bin_by_name("timeout_prog").unwrap()];
        let mut executor = ForkServerExecutor::new(bitmap_tracer, args, None, 200, true)
            .ok()
            .unwrap();

        let result = match executor.run_target(&test_case) {
            Ok(ExecutionStatus::Ok) => true,
            Ok(_) => false,
            Err(_e) => false,
        };
        assert!(result);
    }

    #[test]
    #[serial]
    fn forkserver_handshake_succeed_on_uninstrumented_binary() {
        const MAP_SIZE: usize = 65536;
        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let executor = ForkServerExecutor::new(bitmap_tracer, args, None, 20, true);
        assert!(executor.is_ok());
    }

    #[test]
    #[serial]
    fn forkserver_fork_and_produce_bitmap() {
        const MAP_SIZE: usize = 65536;
        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let mut executor = ForkServerExecutor::new(bitmap_tracer, args, None, 20, true)
            .ok()
            .unwrap();

        let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);

        let result = match executor.run_target(&test_case) {
            Ok(ExecutionStatus::Crash) => false,
            Ok(_) => true,
            Err(_e) => false,
        };
        assert!(result);

        let mut bitmap_counter = 0;
        for i in executor.tracer.get_feedback() {
            if *i != 0 {
                bitmap_counter += 1;
            }
        }
        println!("Bitmap counter: {}", bitmap_counter);
        assert!(bitmap_counter > 1);

        executor.tracer.clear();

        let test_case = TestCase::new(String::from("create table v(a);").as_bytes().to_vec(), 0);

        let result = match executor.run_target(&test_case) {
            Ok(ExecutionStatus::Crash) => false,
            Ok(_) => true,
            Err(_e) => false,
        };
        assert!(result);

        let mut bitmap_counter = 0;
        for i in executor.tracer.get_feedback() {
            if *i != 0 {
                bitmap_counter += 1;
            }
        }
        println!("Bitmap counter: {}", bitmap_counter);
        assert!(bitmap_counter > 1);
    }
    #[test]
    #[serial]
    fn executor_construct_feedback() {
        const MAP_SIZE: usize = 65536;
        let bitmap_tracer = BitmapTracer::new(MAP_SIZE);
        bitmap_tracer.bitmap.write_to_env("__AFL_SHM_ID").unwrap();

        let args = vec![util::get_test_bin_by_name("simple_prog").unwrap()];
        let mut executor = ForkServerExecutor::new(bitmap_tracer, args, None, 50, true)
            .ok()
            .unwrap();

        let test_case = TestCase::new(String::from("select 1;").as_bytes().to_vec(), 0);
        let mut feedback = executor.execute(test_case);
        assert_eq!(feedback.get_status(), ExecutionStatus::Interesting);
        assert!(feedback.contain_test_case());
        assert!(feedback.take_data().is_some());
    }
}
