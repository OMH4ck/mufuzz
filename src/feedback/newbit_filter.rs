use crate::datatype::NewBit;

const VIRGIN_BYTE: u8 = 0xff;

static COUNT_CLASS_LOOKUP: [u8; 256] = [
    0, 1, 2, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
];

const CLASSIFY_COUNT: bool = true;

// Deduplicate visited bits.
pub struct NewBitFilter {
    virgin_bits: Vec<u8>,
}

impl NewBitFilter {
    pub fn new(map_size: usize) -> Self {
        Self {
            virgin_bits: vec![VIRGIN_BYTE; map_size],
        }
    }

    pub fn byte_count(&self) -> usize {
        self.virgin_bits
            .iter()
            .filter(|&b| *b != VIRGIN_BYTE)
            .count()
    }

    fn has_new_bit(&self, bit: &NewBit) -> bool {
        self.has_new_bit_at(bit.index, bit.val)
    }

    fn has_new_bit_at(&self, index: usize, val: u8) -> bool {
        self.virgin_bits[index] & val != 0
    }

    // Try adding a new bit. If the bit has been visited, return false. Otherwise update the
    // virgin map and return true.
    fn try_update_bit(&mut self, bit: &NewBit) -> bool {
        if self.has_new_bit(bit) {
            self.virgin_bits[bit.index] &= !bit.val;
            true
        } else {
            false
        }
    }

    pub fn update_bits(&mut self, bits: &Vec<NewBit>) {
        for bit in bits {
            self.update_bit_at(bit.index, bit.val);
        }
    }

    fn update_bit_at(&mut self, idx: usize, val: u8) {
        self.virgin_bits[idx] &= !val;
    }

    // Try finding new bits and update the virgin map. If any, return true.
    pub fn try_update_bits(&mut self, bits: &[NewBit]) -> bool {
        let mut result = false;
        for bit in bits {
            result |= self.try_update_bit(bit);
        }
        result
    }

    // Filter all the visited bits and return the new ones.
    pub fn filter_old_bits(&self, new_bits: Vec<NewBit>) -> Option<Vec<NewBit>> {
        let result = new_bits
            .into_iter()
            .filter(|new_bit| self.has_new_bit(new_bit))
            .collect::<Vec<_>>();

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // Filter all the visited bits, record and return the new ones.
    pub fn filter_old_bits_mut(&mut self, new_bits: Vec<NewBit>) -> Option<Vec<NewBit>> {
        let mut result = Vec::new();
        for new_bit in new_bits.into_iter() {
            if self.try_update_bit(&new_bit) {
                result.push(new_bit);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // Filter all the visited bits in raw bitmap and return the new ones.
    // Notice: This is only used for criterion and will be removed in the future.
    pub fn filter_old_version(&mut self, raw_info: &[u8]) -> Option<Vec<NewBit>> {
        assert_eq!(raw_info.len(), self.virgin_bits.len());

        let mut result = Vec::new();
        for (idx, b) in raw_info.iter().enumerate() {
            let new_bits = self.virgin_bits[idx] & *b;
            if new_bits != 0 {
                self.update_bit_at(idx, new_bits);
                result.push(NewBit::new(idx, new_bits));
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // TODO: Why this is slower than filter + clear?
    pub fn filter_new_version(&mut self, raw_info: &mut [u8]) -> Option<Vec<NewBit>> {
        assert_eq!(raw_info.len(), self.virgin_bits.len());

        //TODO: This is ugly. How to fix this.
        #[cfg(target_pointer_width = "64")]
        const WORD_SIZE: usize = 8;
        #[cfg(target_pointer_width = "64")]
        let mut raw_info_ptr: *mut u64 = raw_info.as_ptr() as *mut u64;
        #[cfg(target_pointer_width = "64")]
        let mut virgin_bits_ptr: *const u64 = self.virgin_bits.as_ptr() as *const u64;

        #[cfg(target_pointer_width = "32")]
        const WORD_SIZE: usize = 4;
        #[cfg(target_pointer_width = "32")]
        let mut raw_info_ptr: *mut u32 = raw_info.as_ptr() as *mut u32;
        #[cfg(target_pointer_width = "32")]
        let mut virgin_bits_ptr: *const u32 = self.virgin_bits.as_ptr() as *const u32;

        let mut result = Vec::new();
        for i in (0..self.virgin_bits.len()).step_by(WORD_SIZE) {
            unsafe {
                if *raw_info_ptr != 0 {
                    if CLASSIFY_COUNT {
                        for j in 0..WORD_SIZE {
                            if raw_info[i + j] != 0 {
                                raw_info[i + j] = COUNT_CLASS_LOOKUP[raw_info[i + j] as usize];
                            }
                        }
                    }
                    if (*raw_info_ptr & *virgin_bits_ptr) != 0 {
                        for (j, v) in raw_info.iter().enumerate().skip(i).take(WORD_SIZE) {
                            let new_bit = v & self.virgin_bits[j];
                            if new_bit != 0 {
                                self.update_bit_at(j, new_bit);
                                result.push(NewBit::new(j, new_bit));
                            }
                        }
                    }
                    for j in 0..WORD_SIZE {
                        raw_info[i + j] = 0;
                    }
                }
                raw_info_ptr = raw_info_ptr.add(1);
                virgin_bits_ptr = virgin_bits_ptr.add(1);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // Filter all the visited bits in raw bitmap and return the new ones.
    pub fn filter(&mut self, raw_info: &mut [u8]) -> Option<Vec<NewBit>> {
        assert_eq!(raw_info.len(), self.virgin_bits.len());

        //TODO: This is ugly. How to fix this.
        #[cfg(target_pointer_width = "64")]
        const WORD_SIZE: usize = 8;
        #[cfg(target_pointer_width = "64")]
        let mut raw_info_ptr: *const u64 = raw_info.as_ptr() as *const u64;
        #[cfg(target_pointer_width = "64")]
        let mut virgin_bits_ptr: *const u64 = self.virgin_bits.as_ptr() as *const u64;

        #[cfg(target_pointer_width = "32")]
        const WORD_SIZE: usize = 4;
        #[cfg(target_pointer_width = "32")]
        let mut raw_info_ptr: *const u32 = raw_info.as_ptr() as *const u32;
        #[cfg(target_pointer_width = "32")]
        let mut virgin_bits_ptr: *const u32 = self.virgin_bits.as_ptr() as *const u32;

        let mut result = Vec::new();
        for i in (0..self.virgin_bits.len()).step_by(WORD_SIZE) {
            unsafe {
                if *raw_info_ptr != 0 {
                    if CLASSIFY_COUNT {
                        for j in 0..WORD_SIZE {
                            if raw_info[i + j] != 0 {
                                raw_info[i + j] = COUNT_CLASS_LOOKUP[raw_info[i + j] as usize];
                            }
                        }
                    }
                    if (*raw_info_ptr & *virgin_bits_ptr) != 0 {
                        for (j, v) in raw_info.iter().enumerate().skip(i).take(WORD_SIZE) {
                            let new_bit = v & self.virgin_bits[j];
                            if new_bit != 0 {
                                self.update_bit_at(j, new_bit);
                                result.push(NewBit::new(j, new_bit));
                            }
                        }
                    }
                }
                raw_info_ptr = raw_info_ptr.add(1);
                virgin_bits_ptr = virgin_bits_ptr.add(1);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    // Filter all the visited bits in raw bitmap and return the new ones.
    pub fn filter_without_chaning_virgin(&mut self, raw_info: &mut [u8]) -> Option<Vec<NewBit>> {
        assert_eq!(raw_info.len(), self.virgin_bits.len());

        //TODO: This is ugly. How to fix this.
        #[cfg(target_pointer_width = "64")]
        const WORD_SIZE: usize = 8;
        #[cfg(target_pointer_width = "64")]
        let mut raw_info_ptr: *const u64 = raw_info.as_ptr() as *const u64;
        #[cfg(target_pointer_width = "64")]
        let mut virgin_bits_ptr: *const u64 = self.virgin_bits.as_ptr() as *const u64;

        #[cfg(target_pointer_width = "32")]
        const WORD_SIZE: usize = 4;
        #[cfg(target_pointer_width = "32")]
        let mut raw_info_ptr: *const u32 = raw_info.as_ptr() as *const u32;
        #[cfg(target_pointer_width = "32")]
        let mut virgin_bits_ptr: *const u32 = self.virgin_bits.as_ptr() as *const u32;

        let mut result = Vec::new();
        for i in (0..self.virgin_bits.len()).step_by(WORD_SIZE) {
            unsafe {
                if *raw_info_ptr != 0 {
                    if CLASSIFY_COUNT {
                        for j in 0..WORD_SIZE {
                            if raw_info[i + j] != 0 {
                                raw_info[i + j] = COUNT_CLASS_LOOKUP[raw_info[i + j] as usize];
                            }
                        }
                    }
                    if (*raw_info_ptr & *virgin_bits_ptr) != 0 {
                        for (j, v) in raw_info.iter().enumerate().skip(i).take(WORD_SIZE) {
                            let new_bit = v & self.virgin_bits[j];
                            if new_bit != 0 {
                                result.push(NewBit::new(j, new_bit));
                            }
                        }
                    }
                }
                raw_info_ptr = raw_info_ptr.add(1);
                virgin_bits_ptr = virgin_bits_ptr.add(1);
            }
        }

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }

    pub fn visited_bits_num(&self) -> u32 {
        let mut result = 0;
        for i in &self.virgin_bits {
            if *i != VIRGIN_BYTE {
                result += 1;
            }
        }
        result
    }
}

#[test]
fn filter_correctly_classify_hit_count() {
    let map_size = 65536;
    let mut nbfilter = NewBitFilter::new(map_size);
    let mut raw_info = vec![0; map_size];
    raw_info[0x1] = 0xe0;
    raw_info[0x10] = 0x4;
    raw_info[0x100] = 0xe1;

    let result = nbfilter.filter(&mut raw_info).unwrap();
    assert_eq!(result.len(), 3);

    if CLASSIFY_COUNT {
        assert_eq!(result[0], NewBit::new(0x1, COUNT_CLASS_LOOKUP[0xe0]));
        assert_eq!(result[1], NewBit::new(0x10, COUNT_CLASS_LOOKUP[0x4]));
        assert_eq!(result[2], NewBit::new(0x100, COUNT_CLASS_LOOKUP[0xe1]));
    } else {
        assert_eq!(result[0], NewBit::new(0x1, 0xe0));
        assert_eq!(result[1], NewBit::new(0x10, 0x4));
        assert_eq!(result[2], NewBit::new(0x100, 0xe1));
    }

    raw_info[0x1] = 0xf0;
    raw_info[0x10] = 0x0f;
    raw_info[0x100] = 0x80;
    let result = nbfilter.filter(&mut raw_info).unwrap();
    if CLASSIFY_COUNT {
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], NewBit::new(0x10, COUNT_CLASS_LOOKUP[0xf]));
    } else {
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], NewBit::new(0x1, 0x10));
        assert_eq!(result[1], NewBit::new(0x10, 0xB));
    }
}
