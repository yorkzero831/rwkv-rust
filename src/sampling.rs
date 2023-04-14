use std::collections::HashMap;
use futures::SinkExt;
use quicksort::{quicksort, quicksort_by};
use rulinalg::utils;
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub fn sample_logits(logits: *mut f32, logits_length: usize, temperature: f32, top_p: f32) -> usize {
    unsafe {
        let mut rebuilt = Vec::from_raw_parts(logits, logits_length, logits_length);
        let mut soft_max_vec = sort_max_v1(rebuilt);
        sample_probs(soft_max_vec, temperature, top_p)
    }
}

pub fn sample_probs(mut probs: Vec<f32>, temperature: f32, top_p: f32) -> usize {
    let local_top_p = if top_p == 0.0f32 {
        1.0f32
    } else {
        top_p
    };

    if temperature == 0.0f32 {
        let result = utils::argmax(&probs);
        return result.0;
    }

    if local_top_p < 1.0f32 {
        let mut sorted_probs = probs.clone();
        quicksort_by(sorted_probs.as_mut_slice(),|a, b| b.partial_cmp(a).unwrap());
        let cumulative_probs = cum_sum_v1(&sorted_probs);
        let mut index = 0;
        for (i, one) in cumulative_probs.iter().enumerate()  {
            if *one > local_top_p {
                index = i;
                break
            }
        }
        let cutoff = sorted_probs[index];
        probs.iter_mut().for_each(|x| if *x < cutoff { *x = 0f32 });

    }

    if temperature != 1.0f32 {
        let p = 1.0f32 / temperature;
        probs.iter_mut().for_each(|x| *x = (*x).powf(p));
    }

    let sum_of_props: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x = (*x)/sum_of_props);

    let dist = WeightedIndex::new(&probs).unwrap();
    let mut rng = thread_rng();
    let result_index: usize = dist.sample(&mut rng);

    result_index
}

fn sort_max_v1(input: Vec<f32>) -> Vec<f32> {
    let sum_exp: f32 = input.iter().map(|&x| x.exp()).sum();
    input.iter().map(|&x| x.exp() / sum_exp).collect()
}

fn cum_sum_v1(input: &Vec<f32>) -> Vec<f32> {
    let mut cur = 0f32;
    input.iter().map(|&x| {
        cur += x;
        cur
    }).collect()
}

#[test]
fn test() {
    let v1 = vec!(-3.9332f32, 0.7909f32, 0.8927f32);
    let v2 = sort_max_v1(v1);
    assert_eq!(v2, vec!(0.00419590343f32, 0.472580731f32, 0.5232234f32));

    let v3 = vec!(1f32, 2f32, 3f32, 4f32, 5f32);
    let v4 = sort_max_v1(v3);
    assert_eq!(v4, vec!(0.01165623f32, 0.03168492f32, 0.08612854f32, 0.23412164f32, 0.6364086f32));

    let v5 = vec!(1f32, 2f32, 3f32, 4f32, 5f32, 6f32);
    let v6 = cum_sum_v1(&v5);
    assert_eq!(v6, vec!(1f32, 3f32, 6f32, 10f32, 15f32, 21f32));

    let mut v7 = vec!(1f32, 2f32, 3f32, 4f32, 5f32, 6f32);
    v7.iter_mut().for_each(|x| if *x <= 4f32 { *x = 0f32 });
    assert_eq!(v7, vec!(0f32, 0f32, 0f32, 0f32, 5f32, 6f32));

    let mut v8 = vec!(1f32, 2f32, 3f32, 4f32, 5f32, 6f32);
    v8.iter_mut().for_each(|x| *x = (*x).powf(2.0f32));
    assert_eq!(v8, vec!(1f32, 4f32, 9f32, 16f32, 25f32, 36f32));

    let ap = 0.1f32.exp();
    let t = vec!(1f32, 5f32, 3f32, 4f32, 2f32);
    let t1  = sort_max_v1(t);
    let a = sample_probs(t1, 0.2f32, 0.4f32);
    assert_eq!(a, 1)
}
