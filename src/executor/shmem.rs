extern crate alloc;
use crate::Error;
use serde::{Deserialize, Serialize};
use std::env;

use core::fmt::{self, Debug, Display};
use core::{ptr, slice};

use libc::{c_int, c_uchar, shmat, shmctl, shmget};
use std::ptr::null_mut;

/// Description of a shared map.
/// May be used to restore the map by id.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct ShMemDescription {
    /// Size of this map
    pub size: usize,
    /// Id of this map
    pub id: ShMemId,
}

impl ShMemDescription {
    /// Create a description from a `id_str` and a `size`.
    #[must_use]
    pub fn from_string_and_size(id_str: &str, size: usize) -> Self {
        Self {
            size,
            id: ShMemId::from_string(id_str),
        }
    }
}

pub trait ShMem: Sized + Debug + Clone {
    /// Get the id of this shared memory mapping
    fn id(&self) -> ShMemId;

    /// Get the size of this mapping
    fn len(&self) -> usize;

    /// Check if the mapping is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the description of the shared memory mapping
    fn description(&self) -> ShMemDescription {
        ShMemDescription {
            size: self.len(),
            id: self.id(),
        }
    }

    /// The actual shared map, in memory
    fn map(&self) -> &[u8];

    /// The actual shared map, mutable
    fn map_mut(&mut self) -> &mut [u8];

    /// Write this map's config to env
    #[cfg(feature = "std")]
    fn write_to_env(&self, env_name: &str) -> Result<(), Error> {
        let map_size = self.len();
        let map_size_env = format!("{}_SIZE", env_name);
        env::set_var(env_name, self.id().to_string());
        env::set_var(map_size_env, format!("{}", map_size));
        Ok(())
    }
}

/// An id associated with a given shared memory mapping ([`ShMem`]), which can be used to
/// establish shared-mappings between proccesses.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Default)]
pub struct ShMemId {
    id: [u8; 20],
}

impl ShMemId {
    /// Create a new id from a fixed-size string
    #[must_use]
    pub fn from_slice(slice: &[u8; 20]) -> Self {
        Self { id: *slice }
    }

    /// Create a new id from an int
    #[must_use]
    pub fn from_int(val: i32) -> Self {
        Self::from_string(&val.to_string())
    }

    /// Create a new id from a string
    #[must_use]
    pub fn from_string(val: &str) -> Self {
        let mut slice: [u8; 20] = [0; 20];
        for (i, val) in val.as_bytes().iter().enumerate() {
            slice[i] = *val;
        }
        Self { id: slice }
    }

    /// Get the id as a fixed-length slice
    #[must_use]
    pub fn as_slice(&self) -> &[u8; 20] {
        &self.id
    }

    /// Returns the first null-byte in or the end of the buffer
    #[must_use]
    pub fn null_pos(&self) -> usize {
        self.id.iter().position(|&c| c == 0).unwrap()
    }

    /// Returns a `str` representation of this [`ShMemId`]
    #[must_use]
    pub fn as_str(&self) -> &str {
        alloc::str::from_utf8(&self.id[..self.null_pos()]).unwrap()
    }
}

impl From<ShMemId> for i32 {
    fn from(id: ShMemId) -> i32 {
        id.as_str().parse().unwrap()
    }
}

impl Display for ShMemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

pub trait ShMemProvider: Clone + Default + Debug {
    /// The actual shared map handed out by this [`ShMemProvider`].
    type Mem: ShMem;

    /// Create a new instance of the provider
    fn new() -> Result<Self, Error>;

    /// Create a new shared memory mapping
    fn new_map(&mut self, map_size: usize) -> Result<Self::Mem, Error>;

    /// Get a mapping given its id and size
    fn get_by_id_and_size(&mut self, id: ShMemId, size: usize) -> Result<Self::Mem, Error>;

    /// Get a mapping given a description
    fn get_by_description(&mut self, description: ShMemDescription) -> Result<Self::Mem, Error> {
        self.get_by_id_and_size(description.id, description.size)
    }

    /// Create a new sharedmap reference from an existing `id` and `len`
    fn clone_ref(&mut self, mapping: &Self::Mem) -> Result<Self::Mem, Error> {
        self.get_by_id_and_size(mapping.id(), mapping.len())
    }

    /// Reads an existing map config from env vars, then maps it
    #[cfg(feature = "std")]
    fn existing_from_env(&mut self, env_name: &str) -> Result<Self::Mem, Error> {
        let map_shm_str = env::var(env_name)?;
        let map_size = str::parse::<usize>(&env::var(format!("{}_SIZE", env_name))?)?;
        self.get_by_description(ShMemDescription::from_string_and_size(
            &map_shm_str,
            map_size,
        ))
    }

    /// This method should be called before a fork or a thread creation event, allowing the [`ShMemProvider`] to
    /// get ready for a potential reset of thread specific info, and for potential reconnects.
    /// Make sure to call [`Self::post_fork()`] after threading!
    fn pre_fork(&mut self) -> Result<(), Error> {
        // do nothing
        Ok(())
    }

    /// This method should be called after a fork or after cloning/a thread creation event, allowing the [`ShMemProvider`] to
    /// reset thread specific info, and potentially reconnect.
    /// Make sure to call [`Self::pre_fork()`] before threading!
    fn post_fork(&mut self, _is_child: bool) -> Result<(), Error> {
        // do nothing
        Ok(())
    }

    /// Release the resources associated with the given [`ShMem`]
    fn release_map(&mut self, _map: &mut Self::Mem) {
        // do nothing
    }
}

#[derive(Clone, Debug)]
pub struct CommonUnixShMem {
    id: ShMemId,
    map: *mut u8,
    map_size: usize,
}

unsafe impl Send for CommonUnixShMem {}

impl CommonUnixShMem {
    /// Create a new shared memory mapping, using shmget/shmat
    pub fn new(map_size: usize) -> Result<Self, Error> {
        unsafe {
            let os_id = shmget(
                libc::IPC_PRIVATE,
                map_size,
                libc::IPC_CREAT | libc::IPC_EXCL | libc::SHM_R | libc::SHM_W,
            );

            if os_id < 0_i32 {
                return Err(Error::Unknown(format!("Failed to allocate a shared mapping of size {} - check OS limits (i.e shmall, shmmax)", map_size)));
            }

            let map = shmat(os_id, ptr::null(), 0) as *mut c_uchar;

            if map as c_int == -1 || map.is_null() {
                shmctl(os_id, libc::IPC_RMID, ptr::null_mut());
                return Err(Error::Unknown(
                    "Failed to map the shared mapping".to_string(),
                ));
            }

            Ok(Self {
                id: ShMemId::from_int(os_id),
                map,
                map_size,
            })
        }
    }

    /// Get a [`UnixShMem`] of the existing shared memory mapping identified by id
    pub fn from_id_and_size(id: ShMemId, map_size: usize) -> Result<Self, Error> {
        unsafe {
            let id_int: i32 = id.into();
            let map = shmat(id_int, ptr::null(), 0) as *mut c_uchar;

            if map.is_null() || map == null_mut::<c_uchar>().wrapping_sub(1) {
                return Err(Error::Unknown(
                    "Failed to map the shared mapping".to_string(),
                ));
            }

            Ok(Self { id, map, map_size })
        }
    }
}

#[cfg(unix)]
impl ShMem for CommonUnixShMem {
    fn id(&self) -> ShMemId {
        self.id
    }

    fn len(&self) -> usize {
        self.map_size
    }

    fn map(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.map, self.map_size) }
    }

    fn map_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.map, self.map_size) }
    }
}

/// [`Drop`] implementation for [`UnixShMem`], which cleans up the mapping.
#[cfg(unix)]
impl Drop for CommonUnixShMem {
    fn drop(&mut self) {
        unsafe {
            let id_int: i32 = self.id.into();
            shmctl(id_int, libc::IPC_RMID, ptr::null_mut());
        }
    }
}

/// A [`ShMemProvider`] which uses `shmget`/`shmat`/`shmctl` to provide shared memory mappings.
#[cfg(unix)]
#[derive(Clone, Debug)]
pub struct CommonUnixShMemProvider {}

unsafe impl Send for CommonUnixShMemProvider {}

#[cfg(unix)]
impl Default for CommonUnixShMemProvider {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Implement [`ShMemProvider`] for [`UnixShMemProvider`].
#[cfg(unix)]
impl ShMemProvider for CommonUnixShMemProvider {
    type Mem = CommonUnixShMem;

    fn new() -> Result<Self, Error> {
        Ok(Self {})
    }
    fn new_map(&mut self, map_size: usize) -> Result<Self::Mem, Error> {
        CommonUnixShMem::new(map_size)
    }

    fn get_by_id_and_size(&mut self, id: ShMemId, size: usize) -> Result<Self::Mem, Error> {
        CommonUnixShMem::from_id_and_size(id, size)
    }
}
