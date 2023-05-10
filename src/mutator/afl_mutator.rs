use crate::datatype::{ExecutionStatus, Feedback, FeedbackData, TestCase};
use crate::mutator::Mutator;
use rand::prelude::*;
//use rand::rngs::MutationRng;
use super::MutationRng;
use bitflags::bitflags;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::WeightedAliasIndex;

//use super::MutatorFunc;
use core::mem::size_of;

// The mutation id for deterministic pass, which should be ignored during mutation feedback processing.
const INVALID_MUTATOR: u32 = 0xFFFFF;

#[derive(Clone)]
pub struct DeterministicMutatorFunc {
    func: fn(&TestCase, &mut MutationRng) -> Vec<TestCase>,
    flag: DeterministicMutationPass,
}

impl DeterministicMutatorFunc {
    pub fn new(
        func: fn(&TestCase, &mut MutationRng) -> Vec<TestCase>,
        flag: DeterministicMutationPass,
    ) -> Self {
        Self { func, flag }
    }

    pub fn is_match(&self, flag: DeterministicMutationPass) -> bool {
        flag.contains(self.flag)
    }

    pub fn mutate(&self, test_case: &TestCase, rng: &mut MutationRng) -> Vec<TestCase> {
        let mut new_test_cases = (self.func)(test_case, rng);
        for test_case in new_test_cases.iter_mut() {
            test_case.set_mutator_id(INVALID_MUTATOR);
        }
        new_test_cases
    }
}

pub struct MutatorFunc {
    pub func: fn(&TestCase, &mut MutationRng) -> Option<TestCase>,
    pub score: i64,
    pub id: u32,
}

impl MutatorFunc {
    pub fn new(
        id: u32,
        func: fn(&TestCase, &mut MutationRng) -> Option<TestCase>,
        score: i64,
    ) -> Self {
        Self { func, score, id }
    }

    pub fn mutate_once(
        &self,
        test_case: &TestCase,
        rng: &mut MutationRng,
        mutator_id: u32,
    ) -> Option<TestCase> {
        if let Some(mut test_case) = (self.func)(test_case, rng) {
            let mid = (mutator_id << 16) + self.id;
            test_case.set_mutator_id(mid);
            Some(test_case)
        } else {
            None
        }
    }
}

pub struct BitFlipMutator {
    mutated_test_cases: Vec<TestCase>,
    rng: MutationRng,
    mutator_funcs: Vec<MutatorFunc>,
    mutator_deterministic_pass_funcs: Vec<DeterministicMutatorFunc>,
    dist: WeightedAliasIndex<usize>,
    test_case_for_splicing: Option<TestCase>,
    //scores: Vec<i64>,
}

bitflags! {
    //TODO: Each determinitic pass should have a flag.
    #[derive(Default, Clone, Debug, Copy)]
    pub struct DeterministicMutationPass: u32 {
        const BITFLIP1 = 0b1;
        const BITFLIP2 = 0b10;
        const BITFLIP4 = 0b100;
        const BYTEFLIP1 = 0b1000;
    }
}

const DEFAULT_SCORE: i64 = 1000000000000;
const SCORE_UNINTERESTING: i64 = -8;
const SCORE_INTERESTING: i64 = 1600;
const SCORE_TIMEOUT: i64 = -80000;
const SCORE_CRASH: i64 = 10000;

// Marcos from AFL
const ARITH_MAX: i32 = 35;
//const HAVOC_BLK_XL: i32 = 32768;
const HAVOC_BLK_SMALL: usize = 32;
const HAVOC_BLK_MEDIUM: usize = 128;
const HAVOC_BLK_LARGE: usize = 1500;
const HAVOC_BLK_XL: usize = 32768;

// for arith op
const ARITH_SUB: i32 = 0;
const ARITH_ADD: i32 = 1;

const INTERESTING_8: [i8; 9] = [-128, -1, 0, 1, 16, 32, 64, 100, 127];
/// Interesting 16-bit values from AFL
const INTERESTING_16: [i16; 19] = [
    -128, -1, 0, 1, 16, 32, 64, 100, 127, -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
];
/// Interesting 32-bit values from AFL
const INTERESTING_32: [i32; 27] = [
    -128,
    -1,
    0,
    1,
    16,
    32,
    64,
    100,
    127,
    -32768,
    -129,
    128,
    255,
    256,
    512,
    1000,
    1024,
    4096,
    32767,
    -2147483648,
    -100663046,
    -32769,
    32768,
    65535,
    65536,
    100663045,
    2147483647,
];

macro_rules! interesting_mutator_impl {
    ($name: ident, $size: ty, $interesting: ident) => {
        /// Place a random interesting value.
        fn $name(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
            let old_buf = test_case.get_buffer();
            if old_buf.len() < size_of::<$size>() {
                return None;
            }

            let mut new_buf = old_buf.clone();
            let index = rng.gen_range(0..new_buf.len() - size_of::<$size>() + 1);

            let val = *(&$interesting).choose(rng).unwrap();

            let new_bytes = match (&[0, 1]).choose(rng) {
                Some(0) => val.to_be_bytes(),
                _ => val.to_le_bytes(),
            };
            let bytes = &mut new_buf[index..index + size_of::<$size>()];
            bytes.copy_from_slice(&new_bytes);
            Some(test_case.create_variant(new_buf))
        }
    };
}

macro_rules! arith_val_mutator_impl {
    ($name: ident, $size:ty, $op: ident) => {
        fn $name(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
            let old_buf = test_case.get_buffer();
            if old_buf.len() < size_of::<$size>() {
                return None;
            }

            let mut new_buf = old_buf.clone();
            let index = rng.gen_range(0..new_buf.len() - size_of::<$size>() + 1);
            let endian = (&[0, 1]).choose(rng);

            let mut new_val = match endian {
                Some(0) => <$size>::from_be_bytes(
                    new_buf[index..index + size_of::<$size>()]
                        .try_into()
                        .unwrap(),
                ),
                _ => <$size>::from_le_bytes(
                    new_buf[index..index + size_of::<$size>()]
                        .try_into()
                        .unwrap(),
                ),
            };
            new_val = match ($op) {
                ARITH_SUB => new_val.wrapping_sub((rng.gen_range(0..ARITH_MAX) + 1) as $size),
                _ => new_val.wrapping_add((rng.gen_range(0..ARITH_MAX) + 1) as $size),
            };

            let new_bytes = match endian {
                Some(0) => new_val.to_be_bytes(),
                _ => new_val.to_le_bytes(),
            };
            let bytes = &mut new_buf[index..index + size_of::<$size>()];
            bytes.copy_from_slice(&new_bytes);

            Some(test_case.create_variant(new_buf))
        }
    };
}

impl BitFlipMutator {
    pub fn new() -> Self {
        // TODO: There are lots of duplicated mutated inputs.
        let mutator_funcs = vec![
            Self::byte_change as fn(&TestCase, &mut MutationRng) -> Option<TestCase>,
            //Self::bit_flip,
            Self::byte_insert, // 2
            Self::byte_delete,
            Self::byte_shuffle,
            Self::afl_0_bit_flip,
            Self::afl_1_set_interesting_byte,
            Self::afl_2_set_interesting_word,
            Self::afl_3_set_interesting_dword,
            Self::afl_4_sub_arith_val,
            Self::afl_5_add_arith_val,
            Self::afl_6_sub_word_arith_val,
            Self::afl_7_add_word_arith_val,
            Self::afl_8_sub_dword_arith_val,
            Self::afl_9_add_dword_arith_val,
            Self::afl_11_12_remove_bytes,
            Self::afl_13_clone_or_insert_bytes,
            Self::afl_14_overwrite_bytes,
        ]
        .iter()
        .enumerate()
        .map(|(idx, &func)| MutatorFunc::new(idx as u32, func, DEFAULT_SCORE))
        .collect::<Vec<MutatorFunc>>();
        BitFlipMutator {
            mutated_test_cases: Vec::default(),
            rng: MutationRng::from_entropy(),
            //rng: MutationRng::seed_from_u64(crate::datatype::get_id() as u64),
            dist: WeightedAliasIndex::new(vec![DEFAULT_SCORE as usize; mutator_funcs.len()])
                .unwrap(),
            mutator_funcs,
            mutator_deterministic_pass_funcs: vec![DeterministicMutatorFunc::new(
                Self::byte_flip_pass,
                DeterministicMutationPass::BYTEFLIP1,
            )], // TODO
            test_case_for_splicing: None,
        }
    }

    /// Copied from AFL: This is a last-resort strategy triggered by a full round with no findings.
    /// It takes the current input file, randomly selects another input, and
    /// splices them together at some offset, then relies on the havoc
    /// code to mutate that blob.
    fn splice_two_test_cases(
        test_case1: &TestCase,
        test_case2: &TestCase,
        rng: &mut MutationRng,
    ) -> Option<TestCase> {
        let buf1 = test_case1.get_buffer();
        let buf2 = test_case2.get_buffer();
        let check_len = buf1.len().min(buf2.len());

        // Skip too short inputs.
        if check_len < 5 {
            return None;
        }

        let first_idx = buf1
            .iter()
            .zip(buf2.iter())
            .take(check_len)
            .position(|(a, b)| *a != *b)?;

        let last_idx = check_len
            - buf1
                .iter()
                .rev()
                .zip(buf2.iter().rev())
                .take(check_len - first_idx - 1)
                .position(|(a, b)| *a != *b)?;

        if last_idx == first_idx {
            return None;
        }
        let split_at = rng.gen_range(first_idx..last_idx);

        let mut new_buf = buf1.clone();
        new_buf[0..split_at].copy_from_slice(&buf2[0..split_at]);
        Some(test_case1.create_variant(new_buf))
    }

    fn byte_change(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        let mut new_buf = old_buf.clone();
        let idx = rng.gen_range(0..old_buf.len());
        // Avoid changing nothing.
        new_buf[idx] ^= rng.gen_range(0..255) + 1;
        Some(test_case.create_variant(new_buf))
    }

    fn byte_flip_pass(test_case: &TestCase, _rng: &mut MutationRng) -> Vec<TestCase> {
        let old_buf = test_case.get_buffer();
        (0..old_buf.len())
            .map(|idx| {
                let mut new_buf = old_buf.clone();
                new_buf[idx] ^= 0xFF;
                test_case.create_variant(new_buf)
            })
            .collect()
    }

    fn byte_insert(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        let mut new_buf = old_buf.clone();
        let byte_idx = rng.gen_range(0..old_buf.len()) + 1;
        let rand_byte = rng.gen::<u8>();
        new_buf.insert(byte_idx, rand_byte);
        Some(test_case.create_variant(new_buf))
    }

    fn byte_delete(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        if old_buf.is_empty() {
            return None;
        }
        let mut new_buf = old_buf.clone();
        let byte_idx = rng.gen_range(0..old_buf.len());
        new_buf.remove(byte_idx);
        Some(test_case.create_variant(new_buf))
    }

    // Shuffle a few bytes.
    fn byte_shuffle(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();

        if old_buf.len() < 4 {
            return None;
        }

        let mut new_buf = old_buf.clone();

        let start_idx = rng.gen_range(0..(new_buf.len() - 1).saturating_sub(8) + 1);
        let end_idx = new_buf.len().min(start_idx + 8);
        new_buf[start_idx..end_idx].shuffle(rng);
        Some(test_case.create_variant(new_buf))
    }

    fn afl_0_bit_flip(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        let mut new_buf = old_buf.clone();
        let rand_bit_idx = rng.gen_range(0..new_buf.len() * 8);
        new_buf[rand_bit_idx >> 3] ^= 128 >> (rand_bit_idx & 7);
        Some(test_case.create_variant(new_buf))
    }

    interesting_mutator_impl!(afl_1_set_interesting_byte, u8, INTERESTING_8);
    interesting_mutator_impl!(afl_2_set_interesting_word, u16, INTERESTING_16);
    interesting_mutator_impl!(afl_3_set_interesting_dword, u32, INTERESTING_32);
    arith_val_mutator_impl!(afl_4_sub_arith_val, u8, ARITH_SUB);
    arith_val_mutator_impl!(afl_5_add_arith_val, u8, ARITH_ADD);
    arith_val_mutator_impl!(afl_6_sub_word_arith_val, u16, ARITH_SUB);
    arith_val_mutator_impl!(afl_7_add_word_arith_val, u16, ARITH_ADD);
    arith_val_mutator_impl!(afl_8_sub_dword_arith_val, u32, ARITH_SUB);
    arith_val_mutator_impl!(afl_9_add_dword_arith_val, u32, ARITH_ADD);

    // afl_10 is same with byte_change()
    // should increase possibility for this mutation
    fn afl_11_12_remove_bytes(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        if old_buf.len() < 2 {
            return None;
        }

        let mut new_buf = old_buf.clone();
        let del_len = rng.gen_range(1..new_buf.len());
        let del_from = rng.gen_range(0..new_buf.len() - del_len + 1);
        new_buf.drain(del_from..del_from + del_len);
        Some(test_case.create_variant(new_buf))
    }

    fn choose_block_len(limit: usize, rng: &mut StdRng) -> usize {
        let (mut min_value, max_value) = match rng.gen_range(0..20u8) {
            0..=7 => (1, HAVOC_BLK_SMALL),
            8..=15 => (HAVOC_BLK_SMALL, HAVOC_BLK_MEDIUM),
            16..=18 => (HAVOC_BLK_MEDIUM, HAVOC_BLK_LARGE),
            _ => (1, HAVOC_BLK_SMALL),
            //_ => (HAVOC_BLK_LARGE, HAVOC_BLK_XL),
        };

        if min_value >= limit {
            min_value = 1
        }

        min_value + rng.gen_range(0..(max_value.min(limit) - min_value + 1))
    }

    /// Clone bytes (75%) or insert a block of random bytes (25%).
    fn afl_13_clone_or_insert_bytes(
        test_case: &TestCase,
        rng: &mut MutationRng,
    ) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        let actually_clone = rng.gen_range(0..4u8);

        let clone_from;
        let clone_len;
        if actually_clone != 0 {
            clone_len = Self::choose_block_len(old_buf.len(), rng);
            clone_from = rng.gen_range(0..old_buf.len() - clone_len + 1);
        } else {
            clone_len = Self::choose_block_len(HAVOC_BLK_XL, rng);
            clone_from = 0;
        }
        assert_ne!(clone_len, 0);
        let clone_to = rng.gen_range(0..old_buf.len());
        let mut new_buf = vec![0; clone_len + old_buf.len()];

        // Head
        if clone_to != 0 {
            new_buf[0..clone_to].copy_from_slice(&old_buf[0..clone_to]);
        }

        // Inserted part
        if actually_clone != 0 {
            new_buf[clone_to..clone_to + clone_len]
                .copy_from_slice(&old_buf[clone_from..clone_from + clone_len]);
        } else {
            // Fill with constant byte.
            let some_const_byte = if rng.gen::<bool>() {
                rng.gen::<u8>()
            } else {
                old_buf[rng.gen_range(0..old_buf.len())]
            };

            new_buf[clone_to..clone_to + clone_len].fill(some_const_byte);
        }

        // Tail
        assert_eq!(
            old_buf.len() - clone_to,
            new_buf.len() - clone_to - clone_len
        );
        new_buf[clone_to + clone_len..].copy_from_slice(&old_buf[clone_to..]);

        Some(test_case.create_variant(new_buf))
    }

    fn afl_14_overwrite_bytes(test_case: &TestCase, rng: &mut MutationRng) -> Option<TestCase> {
        let old_buf = test_case.get_buffer();
        if old_buf.len() < 2 {
            return None;
        }
        let mut new_buf = old_buf.clone();
        let copy_len = Self::choose_block_len(old_buf.len() - 1, rng);
        let copy_from = rng.gen_range(0..old_buf.len() - copy_len + 1);
        let copy_to = rng.gen_range(0..old_buf.len() - copy_len + 1);

        if rng.gen_range(0..4u8) != 0 && copy_from != copy_to {
            new_buf[copy_to..copy_to + copy_len]
                .copy_from_slice(&old_buf[copy_from..copy_from + copy_len]);
        } else {
            let some_const_byte = if rng.gen::<bool>() {
                rng.gen::<u8>()
            } else {
                old_buf[rng.gen_range(0..old_buf.len())]
            };
            new_buf[copy_to..copy_to + copy_len].fill(some_const_byte);
        }
        Some(test_case.create_variant(new_buf))
    }

    pub fn update_mutator_func_score(&mut self, scores: Vec<(u32, i64)>) {
        for (mutation_id, score) in scores {
            if self.get_id() != (mutation_id >> 16) {
                continue;
            }
            self.mutator_funcs[(mutation_id & 0xFFFF) as usize].score = score;
        }
        self.dist = WeightedAliasIndex::new(
            self.mutator_funcs
                .iter()
                .map(|mutator_func| mutator_func.score as usize)
                .collect(),
        )
        .unwrap();
    }

    pub fn get_mutator_funcs(&self) -> &Vec<MutatorFunc> {
        &self.mutator_funcs
    }

    fn pick_mutator_function(&mut self) -> &MutatorFunc {
        &self.mutator_funcs[self.dist.sample(&mut self.rng)]
    }

    pub fn collect_pass_function(
        &self,
        flag: DeterministicMutationPass,
    ) -> Vec<DeterministicMutatorFunc> {
        self.mutator_deterministic_pass_funcs
            .iter()
            .filter(|x| x.is_match(flag))
            .cloned()
            .collect()
    }
}

impl Default for BitFlipMutator {
    fn default() -> Self {
        Self::new()
    }
}

impl Mutator for BitFlipMutator {
    fn get_rng(&mut self) -> &mut MutationRng {
        &mut self.rng
    }

    fn get_result_pool(&mut self) -> &mut Vec<TestCase> {
        &mut self.mutated_test_cases
    }

    fn get_id(&self) -> u32 {
        1
    }

    fn process_mutation_feedback(&mut self, feedbacks: &[Feedback]) {
        let self_id = self.get_id();
        for feedback in feedbacks.iter() {
            let mutation_info = feedback.get_mutation_info().unwrap();
            let mutator_id = mutation_info.get_mutator_id2();
            let mutator_func_id = mutation_info.get_mutator_func_id();
            if mutator_id != self_id {
                continue;
            }
            let counter = match feedback.borrow_data() {
                Some(FeedbackData::Counter(c)) => *c as i64,
                _ => {
                    unreachable!();
                }
            };

            let score_change = counter
                * match feedback.get_status() {
                    ExecutionStatus::Ok => SCORE_UNINTERESTING,
                    ExecutionStatus::Timeout => SCORE_TIMEOUT,
                    ExecutionStatus::Crash => SCORE_CRASH,
                    ExecutionStatus::Interesting => SCORE_INTERESTING,
                };

            self.mutator_funcs[mutator_func_id as usize].score += score_change;
            assert!(self.mutator_funcs[mutator_func_id as usize].score > 0);
        }

        self.dist = WeightedAliasIndex::new(
            self.mutator_funcs
                .iter()
                .map(|mutator_func| mutator_func.score as usize)
                .collect(),
        )
        .unwrap();
    }

    fn mutate(&mut self, test_case: &TestCase, round: usize) {
        if test_case.get_buffer().is_empty() {
            return;
        }

        let mut rng = self.get_rng().clone();
        if let Some(meta) = test_case.borrow_meta() {
            let pass = meta.get_todo_pass();
            if !pass.is_empty() {
                let mutation_funcs = self.collect_pass_function(pass);
                self.get_result_pool().extend(
                    mutation_funcs
                        .iter()
                        .flat_map(|function| function.mutate(test_case, &mut rng)),
                );
                return;
            }
        }

        // TODO: Filter some of the mutator functions based on meta (done_pass).
        let (do_splicing, round) =
            if self.test_case_for_splicing.is_some() && rng.gen_range(0..4u8) == 0 {
                (true, round >> 2)
            } else {
                (false, round)
            };

        let id = self.get_id();
        for _i in 0..round {
            let mut mutated_test_case = test_case.clone();
            assert!(!mutated_test_case.get_buffer().is_empty());
            for _j in 0..(rng.gen_range(2..16u8)) {
                let mutator_func = self.pick_mutator_function();
                if let Some(tmp_test_case) =
                    mutator_func.mutate_once(&mutated_test_case, &mut rng, id)
                {
                    if !tmp_test_case.get_buffer().is_empty() {
                        mutated_test_case = tmp_test_case;
                    }
                }
                assert!(!mutated_test_case.get_buffer().is_empty())
            }
            self.get_result_pool().push(mutated_test_case);
        }

        // Splice two test cases.
        if do_splicing {
            let splice_test_case = self.test_case_for_splicing.take().unwrap();
            if let Some(spliced_test_case) =
                Self::splice_two_test_cases(test_case, &splice_test_case, &mut rng)
            {
                for _i in 0..round {
                    let mutator_func = self.pick_mutator_function();
                    if let Some(test_case) =
                        mutator_func.mutate_once(&spliced_test_case, &mut rng, id)
                    {
                        self.get_result_pool().push(test_case);
                    }
                }
            }
        }

        if self.test_case_for_splicing.is_none() && self.get_rng().gen_range(0..10u8) == 0 {
            self.test_case_for_splicing = Some(test_case.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::{FeedbackData, MutationInfo, TestCase};

    #[test]
    fn mutator_pick_mutator_function_base_on_weight() {
        let mut mutator = BitFlipMutator::new();
        let mutator_function_num = mutator.get_mutator_funcs().len();
        let mut counters = vec![0; mutator_function_num];
        mutator.dist = WeightedAliasIndex::new((1..=mutator_function_num).collect()).unwrap();
        let total_weight: usize = (1..=mutator_function_num).sum();

        const REPEAT_TIMES: usize = 100000;
        for _i in 0..REPEAT_TIMES {
            counters[mutator.pick_mutator_function().id as usize] += 1;
        }

        for (idx, count) in counters.iter().enumerate() {
            let expected_ratio = (idx + 1) as f64 / total_weight as f64;
            let experiment_ratio = *count as f64 / REPEAT_TIMES as f64;
            assert!((expected_ratio - experiment_ratio).abs() < 0.005);
        }
    }

    #[test]
    fn mutator_process_feedback_and_ajust_mutator_function_weight() {
        let mut mutator = BitFlipMutator::new();
        let mut feedbacks = Vec::new();

        let mut score_change = 0;

        feedbacks.extend((0..10).map(|i| {
            Feedback::new(ExecutionStatus::Timeout)
                .set_mutation_info(MutationInfo::new(0, (1 << 16) + i))
                .set_data(FeedbackData::Counter(2))
        }));
        score_change += 2 * SCORE_TIMEOUT;

        feedbacks.extend((0..10).map(|i| {
            Feedback::new(ExecutionStatus::Crash)
                .set_mutation_info(MutationInfo::new(0, (1 << 16) + i))
                .set_data(FeedbackData::Counter(2))
        }));
        score_change += 2 * SCORE_CRASH;

        feedbacks.extend((0..10).map(|i| {
            Feedback::new(ExecutionStatus::Ok)
                .set_mutation_info(MutationInfo::new(0, (1 << 16) + i))
                .set_data(FeedbackData::Counter(2))
        }));
        score_change += 2 * SCORE_UNINTERESTING;

        feedbacks.extend((0..10).map(|i| {
            Feedback::new(ExecutionStatus::Interesting)
                .set_mutation_info(MutationInfo::new(0, (1 << 16) + i))
                .set_data(FeedbackData::Counter(1))
        }));
        score_change += SCORE_INTERESTING;

        mutator.process_mutation_feedback(&feedbacks);

        for i in 0..10 {
            assert_eq!(
                mutator.get_mutator_funcs()[i].score,
                DEFAULT_SCORE + score_change
            );
        }
    }

    #[test]
    fn test_basic_mutation_only_change_one_byte_at_a_time() {
        let test_case = TestCase::new(vec![1, 2, 3], 0);
        let mut m = BitFlipMutator::new();
        let new_test_case = BitFlipMutator::afl_0_bit_flip(&test_case, m.get_rng()).unwrap();
        m.mutated_test_cases.push(new_test_case);
        let new_test_case = BitFlipMutator::byte_change(&test_case, m.get_rng()).unwrap();
        m.mutated_test_cases.push(new_test_case);
        assert_eq!(m.mutated_test_cases.len(), 2);
        m.mutated_test_cases.iter().for_each(|mutated_test_case| {
            assert!(
                mutated_test_case
                    .get_buffer()
                    .iter()
                    .zip(test_case.get_buffer().iter())
                    .filter(|(a, b)| *a != *b)
                    .count()
                    == 1
            )
        });
    }

    #[test]
    fn test_byte_shuffle() {
        let test_case = TestCase::new((0..100).collect(), 0);
        let old_buf = test_case.get_buffer();
        let mut m = BitFlipMutator::new();
        for _i in 0..50 {
            if let Some(tc) = BitFlipMutator::byte_shuffle(&test_case, m.get_rng()) {
                m.mutated_test_cases.push(tc);
            }
        }

        let mut diff_counter = 0;

        m.get_mutated_test_cases(None)
            .into_iter()
            .for_each(|new_test_case| {
                let new_buf = new_test_case.get_buffer();
                assert_eq!(new_buf.len(), old_buf.len());

                let diff_byte_num = new_buf
                    .iter()
                    .zip(old_buf.iter())
                    .filter(|(a, b)| *a != *b)
                    .count();
                assert!(diff_byte_num <= 8);
                if diff_byte_num != 0 {
                    diff_counter += 1;
                }
            });

        assert_ne!(diff_counter, 0);
    }

    #[test]
    fn test_basic_mutation_insert_one_byte() {
        let test_case = TestCase::new(vec![1, 2, 3, 4, 5], 0);
        let mut m = BitFlipMutator::new();
        let old_len = test_case.get_buffer().len();

        for _i in 0..50 {
            if let Some(tc) = BitFlipMutator::byte_insert(&test_case, m.get_rng()) {
                m.mutated_test_cases.push(tc);
            }
        }
        //test if rest of the content is still the same between old test_case & mutated test_case
        m.mutated_test_cases.iter().for_each(|mutated_test_case| {
            assert!(mutated_test_case.get_buffer().len() == old_len + 1);
            let old_buf = test_case.get_buffer().clone();
            let mut new_buf = mutated_test_case.get_buffer().clone();
            for i in 0..old_buf.len() {
                if old_buf[i] != new_buf[i] {
                    new_buf.remove(i);
                }
            }
            if old_buf.len() != new_buf.len() {
                new_buf.remove(new_buf.len() - 1);
            }
            assert_eq!(new_buf, old_buf);
        });
    }

    #[test]
    fn test_basic_mutation_delete_one_byte() {
        let test_case = TestCase::new(vec![1, 2, 3, 4, 5], 0);
        let mut m = BitFlipMutator::new();
        let old_len = test_case.get_buffer().len();

        let empty_test_case = TestCase::new(Vec::default(), 0);
        BitFlipMutator::byte_delete(&empty_test_case, m.get_rng());
        assert_eq!(m.get_mutated_test_cases(None).len(), 0);

        for _i in 0..50 {
            if let Some(tc) = BitFlipMutator::byte_delete(&test_case, m.get_rng()) {
                m.mutated_test_cases.push(tc);
            }
        }
        //test if length is increased by 1
        //test if rest of the content is still the same between old test_case & mutated test_case
        m.mutated_test_cases.iter().for_each(|mutated_test_case| {
            assert!(mutated_test_case.get_buffer().len() == old_len - 1);
            let mut old_buf = test_case.get_buffer().clone();
            let new_buf = mutated_test_case.get_buffer().clone();
            for i in 0..new_buf.len() {
                if old_buf[i] != new_buf[i] {
                    old_buf.remove(i);
                }
            }
            if old_buf.len() != new_buf.len() {
                old_buf.remove(old_buf.len() - 1);
            }
            assert_eq!(new_buf, old_buf);
        });
    }
    #[test]
    fn test_get_mutated_test_cases() {
        let test_case = TestCase::new(vec![1, 2, 3, 4, 5], 0);
        let mut m = BitFlipMutator::new();
        m.mutate(&test_case, 200);
        assert_eq!(m.get_mutated_test_cases(Some(2)).len(), 2);
        assert!(m.get_mutated_test_cases(None).len() > 25);
        assert!(m.get_mutated_test_cases(Some(1)).is_empty());
    }
    #[test]
    fn change_interesting_value_32_replace_4_bytes() {
        // 13 is not an interesing value.
        const UNINTERESTING_VALUE: u8 = 13;
        let test_case = TestCase::new(vec![UNINTERESTING_VALUE; 10], 0);
        let old_buf = test_case.get_buffer();
        let mut m = BitFlipMutator::new();

        for _i in 0..50 {
            if let Some(tc) = BitFlipMutator::afl_3_set_interesting_dword(&test_case, m.get_rng()) {
                m.mutated_test_cases.push(tc);
            }
        }

        m.get_mutated_test_cases(None)
            .into_iter()
            .for_each(|new_test_case| {
                let new_buf = new_test_case.get_buffer();
                assert_eq!(new_buf.len(), old_buf.len());

                // We change 4 bytes.
                let diff_count = new_buf
                    .iter()
                    .zip(old_buf.iter())
                    .filter(|(&a, &b)| a != b)
                    .count();
                assert_eq!(diff_count, 4);

                // The value comes from INTERESTING_32.
                let idx = new_buf
                    .iter()
                    .position(|&x| x != UNINTERESTING_VALUE)
                    .unwrap();
                let le_value: i32 = i32::from_le_bytes(new_buf[idx..idx + 4].try_into().unwrap());
                let be_value: i32 = i32::from_be_bytes(new_buf[idx..idx + 4].try_into().unwrap());
                assert!(INTERESTING_32
                    .iter()
                    .any(|&x| x == be_value || x == le_value));
            });
    }

    #[test]
    fn mutator_add_info_to_generated_test_case() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![1, 2, 3, 4, 5], 0);
        const BYTE_INSERT_IDX: u32 = 2;
        let mut rng = m.get_rng().clone();
        let mutator_id = m.get_id();
        let mutator_func = &m.get_mutator_funcs()[BYTE_INSERT_IDX as usize];
        let mutated_test_cases = (0..50)
            .filter_map(|_i| mutator_func.mutate_once(&test_case, &mut rng, mutator_id))
            .collect::<Vec<TestCase>>();
        assert!(!mutated_test_cases.is_empty());

        for test_case in mutated_test_cases {
            assert_eq!(
                test_case.get_mutator_id(),
                (m.get_id() << 16) + BYTE_INSERT_IDX
            );
        }
    }

    #[test]
    fn afl_0_bit_flip_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![1, 2, 3, 4, 5], 0);
        let new_test_case = BitFlipMutator::afl_0_bit_flip(&test_case, m.get_rng()).unwrap();
        assert_eq!(
            test_case.get_buffer().len(),
            new_test_case.get_buffer().len()
        );

        let mut diff_byte_num = 0;
        for _i in 0..new_test_case.get_buffer().len() {
            if new_test_case.get_buffer()[_i] != test_case.get_buffer()[_i] {
                diff_byte_num += 1;
            }
        }

        assert_eq!(diff_byte_num, 1);
        println!("{:?}", new_test_case.get_buffer());
    }

    #[test]
    fn afl_1_set_interesting_byte_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![13; 10], 0);
        let new_test_case =
            BitFlipMutator::afl_1_set_interesting_byte(&test_case, m.get_rng()).unwrap();
        assert_eq!(
            test_case.get_buffer().len(),
            new_test_case.get_buffer().len()
        );

        let mut diff_byte_num = 0;
        for i in 0..new_test_case.get_buffer().len() {
            if new_test_case.get_buffer()[i] != test_case.get_buffer()[i] {
                diff_byte_num += 1;
                assert!(INTERESTING_8
                    .iter()
                    .any(|x| x == &(new_test_case.get_buffer()[i] as i8)));
            }
        }

        assert_eq!(diff_byte_num, 1);
        println!("{:?}", new_test_case.get_buffer());
    }

    #[test]
    fn afl_2_set_interesting_word_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![13; 10], 0);
        let new_test_case =
            BitFlipMutator::afl_2_set_interesting_word(&test_case, m.get_rng()).unwrap();
        assert_eq!(
            test_case.get_buffer().len(),
            new_test_case.get_buffer().len()
        );

        let mut le_val = 0;
        let mut be_val = 0;
        for i in 0..new_test_case.get_buffer().len() {
            if new_test_case.get_buffer()[i] != test_case.get_buffer()[i] {
                le_val =
                    i16::from_le_bytes(new_test_case.get_buffer()[i..i + 2].try_into().unwrap());
                be_val =
                    i16::from_be_bytes(new_test_case.get_buffer()[i..i + 2].try_into().unwrap());
                break;
            }
        }
        assert!(INTERESTING_16.iter().any(|x| x == &le_val || x == &be_val));
    }

    #[test]
    fn afl_3_set_interesting_dword_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![13; 10], 0);
        let new_test_case =
            BitFlipMutator::afl_3_set_interesting_dword(&test_case, m.get_rng()).unwrap();
        assert_eq!(
            test_case.get_buffer().len(),
            new_test_case.get_buffer().len()
        );

        let mut le_val = 0;
        let mut be_val = 0;
        for i in 0..new_test_case.get_buffer().len() {
            if new_test_case.get_buffer()[i] != test_case.get_buffer()[i] {
                le_val =
                    i32::from_le_bytes(new_test_case.get_buffer()[i..i + 4].try_into().unwrap());
                be_val =
                    i32::from_be_bytes(new_test_case.get_buffer()[i..i + 4].try_into().unwrap());
                break;
            }
        }
        assert!(INTERESTING_32.iter().any(|x| x == &le_val || x == &be_val));
    }

    #[test]
    fn afl_4_5_arith_op_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![13; 10], 0);
        let new_test_case1 = BitFlipMutator::afl_4_sub_arith_val(&test_case, m.get_rng()).unwrap();
        let new_test_case2 = BitFlipMutator::afl_5_add_arith_val(&test_case, m.get_rng()).unwrap();

        for i in 0..new_test_case1.get_buffer().len() {
            if new_test_case1.get_buffer()[i] != test_case.get_buffer()[i] {
                let le_val_num1 =
                    i8::from_le_bytes(test_case.get_buffer()[i..i + 1].try_into().unwrap())
                        .wrapping_sub(i8::from_le_bytes(
                            new_test_case1.get_buffer()[i..i + 1].try_into().unwrap(),
                        ));
                let be_val_num1 =
                    i8::from_be_bytes(test_case.get_buffer()[i..i + 1].try_into().unwrap())
                        .wrapping_sub(i8::from_be_bytes(
                            new_test_case1.get_buffer()[i..i + 1].try_into().unwrap(),
                        ));
                assert!(
                    (le_val_num1 > 0 && le_val_num1 < 36) || (be_val_num1 > 0 && be_val_num1 < 36)
                );
                break;
            }
        }

        for i in 0..new_test_case2.get_buffer().len() {
            if new_test_case2.get_buffer()[i] != test_case.get_buffer()[i] {
                let le_val_num2 =
                    i8::from_le_bytes(new_test_case2.get_buffer()[i..i + 1].try_into().unwrap())
                        .wrapping_sub(i8::from_le_bytes(
                            test_case.get_buffer()[i..i + 1].try_into().unwrap(),
                        ));
                let be_val_num2 =
                    i8::from_be_bytes(new_test_case2.get_buffer()[i..i + 1].try_into().unwrap())
                        .wrapping_sub(i8::from_be_bytes(
                            test_case.get_buffer()[i..i + 1].try_into().unwrap(),
                        ));
                assert!(
                    (le_val_num2 > 0 && le_val_num2 < 36) || (be_val_num2 > 0 && be_val_num2 < 36)
                );
                break;
            }
        }
    }

    // todo: merge test
    #[test]
    fn afl_6_7_arith_op_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![0; 10], 0);
        let test_case2 = TestCase::new(vec![255; 10], 0);
        let new_test_case1 =
            BitFlipMutator::afl_6_sub_word_arith_val(&test_case, m.get_rng()).unwrap();
        let new_test_case2 =
            BitFlipMutator::afl_7_add_word_arith_val(&test_case2, m.get_rng()).unwrap();

        let mut is_modified1 = false;
        let mut is_modified2 = false;

        for i in 0..new_test_case1.get_buffer().len() - 1 {
            if new_test_case1.get_buffer()[i] != test_case.get_buffer()[i] {
                let le_val_num1 =
                    i16::from_le_bytes(test_case.get_buffer()[i..i + 2].try_into().unwrap())
                        .wrapping_sub(i16::from_le_bytes(
                            new_test_case1.get_buffer()[i..i + 2].try_into().unwrap(),
                        ));

                let be_val_num1 =
                    i16::from_be_bytes(test_case.get_buffer()[i..i + 2].try_into().unwrap())
                        .wrapping_sub(i16::from_be_bytes(
                            new_test_case1.get_buffer()[i..i + 2].try_into().unwrap(),
                        ));

                assert!(
                    (le_val_num1 > 0 && le_val_num1 < 36) || (be_val_num1 > 0 && be_val_num1 < 36)
                );
                is_modified1 = true;
                break;
            }
        }
        assert!(is_modified1);

        for i in 0..new_test_case2.get_buffer().len() - 1 {
            if new_test_case2.get_buffer()[i] != test_case2.get_buffer()[i] {
                let le_val_num2 =
                    i16::from_le_bytes(new_test_case2.get_buffer()[i..i + 2].try_into().unwrap())
                        .wrapping_sub(i16::from_le_bytes(
                            test_case2.get_buffer()[i..i + 2].try_into().unwrap(),
                        ));

                let be_val_num2 =
                    i16::from_be_bytes(new_test_case2.get_buffer()[i..i + 2].try_into().unwrap())
                        .wrapping_sub(i16::from_be_bytes(
                            test_case2.get_buffer()[i..i + 2].try_into().unwrap(),
                        ));

                assert!(
                    (le_val_num2 > 0 && le_val_num2 < 36) || (be_val_num2 > 0 && be_val_num2 < 36)
                );
                is_modified2 = true;
                break;
            }
        }
        assert!(is_modified2);
    }

    #[test]
    fn afl_8_9_arith_op_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![0; 10], 0);
        let test_case2 = TestCase::new(vec![255; 10], 0);
        let new_test_case1 =
            BitFlipMutator::afl_8_sub_dword_arith_val(&test_case, m.get_rng()).unwrap();
        let new_test_case2 =
            BitFlipMutator::afl_9_add_dword_arith_val(&test_case2, m.get_rng()).unwrap();

        let mut is_modified1 = false;
        let mut is_modified2 = false;

        for i in 0..new_test_case1.get_buffer().len() - 3 {
            if new_test_case1.get_buffer()[i] != test_case.get_buffer()[i] {
                let le_val_num1 =
                    i32::from_le_bytes(test_case.get_buffer()[i..i + 4].try_into().unwrap())
                        .wrapping_sub(i32::from_le_bytes(
                            new_test_case1.get_buffer()[i..i + 4].try_into().unwrap(),
                        ));

                let be_val_num1 =
                    i32::from_be_bytes(test_case.get_buffer()[i..i + 4].try_into().unwrap())
                        .wrapping_sub(i32::from_be_bytes(
                            new_test_case1.get_buffer()[i..i + 4].try_into().unwrap(),
                        ));

                assert!(
                    (le_val_num1 > 0 && le_val_num1 < 36) || (be_val_num1 > 0 && be_val_num1 < 36)
                );
                is_modified1 = true;
                break;
            }
        }
        assert!(is_modified1);

        for i in 0..new_test_case2.get_buffer().len() - 3 {
            if new_test_case2.get_buffer()[i] != test_case2.get_buffer()[i] {
                let le_val_num2 =
                    i32::from_le_bytes(new_test_case2.get_buffer()[i..i + 4].try_into().unwrap())
                        .wrapping_sub(i32::from_le_bytes(
                            test_case2.get_buffer()[i..i + 4].try_into().unwrap(),
                        ));

                let be_val_num2 =
                    i32::from_be_bytes(new_test_case2.get_buffer()[i..i + 4].try_into().unwrap())
                        .wrapping_sub(i32::from_be_bytes(
                            test_case2.get_buffer()[i..i + 4].try_into().unwrap(),
                        ));

                assert!(
                    (le_val_num2 > 0 && le_val_num2 < 36) || (be_val_num2 > 0 && be_val_num2 < 36)
                );
                is_modified2 = true;
                break;
            }
        }
        assert!(is_modified2);
    }

    #[test]
    fn afl_11_12_remove_bytes_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new(vec![13; 10], 0);
        let new_test_case =
            BitFlipMutator::afl_11_12_remove_bytes(&test_case, m.get_rng()).unwrap();

        assert!(
            new_test_case.get_buffer().len() < test_case.get_buffer().len()
                && !new_test_case.get_buffer().is_empty()
        );
    }

    #[test]
    fn afl_13_clone_or_insert_bytes_mutate_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new((0..100).collect(), 0);

        for _i in 0..50000 {
            let new_test_case =
                BitFlipMutator::afl_13_clone_or_insert_bytes(&test_case, m.get_rng()).unwrap();
            let increased_len = new_test_case.get_buffer().len() - test_case.get_buffer().len();
            let old_buf = test_case.get_buffer();
            let new_buf = new_test_case.get_buffer();

            assert!(
                increased_len > 0 && increased_len <= HAVOC_BLK_XL,
                "len: {increased_len}"
            );
            let head_end = new_buf
                .iter()
                .zip(old_buf.iter())
                .take(old_buf.len())
                .position(|(a, b)| a != b)
                .unwrap_or(old_buf.len());
            let tail_start = new_buf
                .iter()
                .rev()
                .zip(old_buf.iter().rev())
                .take(old_buf.len() - head_end)
                .position(|(a, b)| a != b)
                .unwrap_or(old_buf.len() - head_end);
            assert_eq!(head_end + tail_start, old_buf.len());

            let mut const_byte = new_buf[head_end];

            // Not super strict check.
            for i in &new_buf[head_end..new_buf.len() - tail_start] {
                if *i != const_byte {
                    assert_eq!(*i, const_byte + 1);
                    const_byte += 1;
                } else {
                    assert_eq!(*i, const_byte);
                }
            }
        }
    }

    // TODO: The test is sloppy now, just checking whether the content has changed.
    #[test]
    fn afl_14_overwrite_bytes_test() {
        let mut m = BitFlipMutator::new();
        let test_case = TestCase::new((0..100).collect(), 0);

        let mut diff_counter = 0;
        for _i in 0..50000 {
            let new_test_case =
                BitFlipMutator::afl_14_overwrite_bytes(&test_case, m.get_rng()).unwrap();
            assert_eq!(
                new_test_case.get_buffer().len(),
                test_case.get_buffer().len()
            );
            let old_buf = test_case.get_buffer();
            if !old_buf.eq(new_test_case.get_buffer()) {
                diff_counter += 1;
            }
        }

        assert!(
            diff_counter > 45000,
            "Only {} of the new test cases are really new!",
            diff_counter
        );
    }

    #[test]
    fn afl_splicing_merge_two_test_cases() {
        let mut m = BitFlipMutator::new();
        let test_case1 = TestCase::new(vec![0x33; 0x100], 0);
        let test_case2 = TestCase::new(vec![0x33; 0x300], 0);
        assert!(
            BitFlipMutator::splice_two_test_cases(&test_case1, &test_case2, m.get_rng()).is_none()
        );
        let test_case2 = TestCase::new(vec![0x43; 0x300], 0);

        let mut new_buf = vec![0x43; 0x100];
        new_buf[1..0xFF].copy_from_slice(&test_case1.get_buffer()[1..0xFF]);
        println!("New buf {:?}", new_buf);
        let test_case1 = TestCase::new(new_buf, 0);
        for _i in 0..5000 {
            let new_test_case =
                BitFlipMutator::splice_two_test_cases(&test_case1, &test_case2, m.get_rng());
            assert!(new_test_case.is_some());
            let new_test_case = new_test_case.unwrap();

            assert_eq!(new_test_case.get_buffer().len(), 0x100);
            let splice_at = new_test_case
                .get_buffer()
                .iter()
                .position(|x| *x == 0x33)
                .unwrap();

            assert_ne!(splice_at, 0);
            assert_ne!(splice_at, 0xFF);
            for x in new_test_case.get_buffer()[0..splice_at].iter() {
                assert_eq!(*x, 0x43);
            }
            for x in new_test_case.get_buffer()[splice_at..0xFF].iter() {
                assert_eq!(*x, 0x33);
            }
        }
    }
}
