pub mod afl_mutator;
pub mod frontend;
pub mod rpc;

use crate::datatype::{Feedback, TestCase};

type MutationRng = rand::rngs::StdRng;
/*
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
            test_case.set_mutator_id(mid as u32);
            Some(test_case)
        } else {
            None
        }
    }
}
*/

pub trait Mutator {
    //fn get_mutator_funcs(&self) -> &Vec<MutatorFunc>;

    // Each mutator should have its own unique id.
    // TODO: We should automatically assign this id.
    fn get_id(&self) -> u32;

    fn get_rng(&mut self) -> &mut MutationRng;
    //fn get_dist(&self) -> &WeightedAliasIndex<usize>;
    //fn pick_mutator_function(&mut self) -> &MutatorFunc;
    fn get_result_pool(&mut self) -> &mut Vec<TestCase>;
    fn mutate(&mut self, test_case: &TestCase, round: usize);
    /* {
        if test_case.get_buffer().is_empty() {
            return;
        }

        let mut rng = self.get_rng().clone();
        let id = self.get_id();
        for _i in 0..round {
            let mutator_func = self.pick_mutator_function();
            if let Some(test_case) = mutator_func.mutate_once(test_case, &mut rng, id) {
                self.get_result_pool().push(test_case);
            }
        }
    }
    */

    fn get_mutated_test_cases(&mut self, num: Option<u32>) -> Vec<TestCase> {
        let n_new_test_case = self.get_result_pool().len();
        match num {
            Some(sz) => {
                let mut nsize = sz as usize;
                nsize = nsize.min(n_new_test_case);
                self.get_result_pool().split_off(n_new_test_case - nsize)
            }
            None => self.get_result_pool().split_off(0),
        }
    }

    /// Check the feedbacks about the mutation and fine-tune the scheduling of mutator functions.
    fn process_mutation_feedback(&mut self, feedbacks: &[Feedback]);
}
